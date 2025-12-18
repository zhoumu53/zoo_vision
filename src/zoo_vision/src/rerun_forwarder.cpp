// This file is part of zoo_vision.
//
// zoo_vision is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// zoo_vision is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// zoo_vision. If not, see <https://www.gnu.org/licenses/>.

#include "zoo_vision/rerun_forwarder.hpp"

#include "zoo_msgs/msg/track_state.h"
#include "zoo_vision/utils.hpp"

#include <nlohmann/json.hpp>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/time.hpp>

#include <chrono>
#include <cinttypes>
#include <format>
#include <fstream>

using namespace std::chrono_literals;
using json = nlohmann::json;

using CImage12m = zoo_msgs::msg::Image12m;
using CImage4m = zoo_msgs::msg::Image4m;
using CDetection = zoo_msgs::msg::Detection;
using CTrackState = zoo_msgs::msg::TrackState;

extern "C" {
extern uint32_t zoo_rs_init(void **zoo_rs_handle, char const *const data_path, char const *const config_json);
extern uint32_t zoo_rs_test_me(void *zoo_rs_handle, char const *const frame_id);
extern uint32_t zoo_rs_image_callback(void *zoo_rs_handle, char const *const cameraTopic, char const *const channel,
                                      const CImage12m *);
extern uint32_t zoo_rs_detection_callback(void *zoo_rs_handle, char const *const cameraTopic, char const *const channel,
                                          const CDetection *);
extern uint32_t zoo_rs_track_state_callback(void *zoo_rs_handle, char const *const cameraTopic,
                                            char const *const channel, const CTrackState *);

// Dummy definitions to disable the zoo_vision_rs library
// uint32_t zoo_rs_init(void **, char const *const, char const *const) { return 0; }
// uint32_t zoo_rs_test_me(void *, char const *const) { return 0; }
// uint32_t zoo_rs_image_callback(void *, char const *const, char const *const, const CImage12m *) { return 0; }
// uint32_t zoo_rs_detection_callback(void *, char const *const, char const *const, const CDetection *) { return 0; }
// uint32_t zoo_rs_track_state_callback(void *, char const *const, char const *const, const CTrackState *) { return 0; }
}
namespace zoo {

auto timeFromRosTime(const builtin_interfaces::msg::Time &stamp) {
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds(stamp.sec) +
                                                                        std::chrono::nanoseconds(stamp.nanosec));
  return std::chrono::system_clock::time_point{duration};
}

RerunForwarder::RerunForwarder(const rclcpp::NodeOptions &options) : Node("rerun_forwarder", options) {

  // Load config
  const std::vector<std::string> cameraNames = [&]() {
    std::vector<std::string> names;
    json config = getConfig();
    for (auto &item : config["cameras"].items()) {
      names.push_back(item.key());
    }
    return names;
  }();
  {
    std::string cameraNamesDescr;
    {
      for (const auto &name : cameraNames) {
        if (!cameraNamesDescr.empty()) {
          cameraNamesDescr.append(", ");
        }
        cameraNamesDescr.append(name);
      }
      cameraNamesDescr = "[" + cameraNamesDescr + "]";
    }
    RCLCPP_INFO(get_logger(), "Configured cameras=%s", cameraNamesDescr.c_str());
  }

  // All rerun calls are thread-safe so this node is can accept calls in different threads simultaneously
  rclcpp::SubscriptionOptions imageOptions;
  imageOptions.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions otherOptions;
  otherOptions.callback_group = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  const auto reliableQoS = rclcpp::QoS(rclcpp::KeepAll{}).durability_volatile().reliable();

  auto subscribeImage = [&](std::string cameraName, std::string channel) {
    auto subscription = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
        *this, channel, reliableQoS,
        [this, cameraName, channel](std::shared_ptr<const zoo_msgs::msg::Image12m> msg) {
          this->onImage(cameraName, channel, std::move(msg));
        },
        imageOptions);
    RCLCPP_INFO(get_logger(), "Subscribed to detection image %s (loans=%d)", channel.c_str(),
                subscription->can_loan_messages());
    imageSubscribers_.push_back(std::move(subscription));
  };

  auto subscribeDetection = [&](std::string cameraName, std::string channel) {
    auto subscription = rclcpp::create_subscription<zoo_msgs::msg::Detection>(
        *this, channel, reliableQoS,
        [this, cameraName, channel](std::shared_ptr<const zoo_msgs::msg::Detection> msg) {
          this->onDetection(cameraName, channel, *msg);
        },
        otherOptions);
    RCLCPP_INFO(get_logger(), "Subscribed to detection results %s (loans=%d)", channel.c_str(),
                subscription->can_loan_messages());
    detectionSubscribers_.push_back(std::move(subscription));
  };

  auto subscribeTrackState = [&](std::string cameraName, std::string channel) {
    auto subscription = rclcpp::create_subscription<zoo_msgs::msg::TrackState>(
        *this, channel, reliableQoS,
        [this, cameraName, channel](std::shared_ptr<const zoo_msgs::msg::TrackState> msg) {
          this->onTrackState(cameraName, channel, *msg);
        },
        otherOptions);
    RCLCPP_INFO(get_logger(), "Subscribed to track state results %s (loans=%d)", channel.c_str(),
                subscription->can_loan_messages());
    trackStateSubscribers_.push_back(std::move(subscription));
  };

  // Subscribe to all cameras
  for (const auto &name : cameraNames) {
    imageCaches_.emplace(name, std::make_unique<CameraData>());

    // subscribeImage(name, name + "/detections/image"); // Detection image, full res but in sync with detections
    subscribeImage(name, name + "/image"); // Full-res image from camera
    subscribeDetection(name, name + "/detections");
    subscribeTrackState(name, name + "/track_state");
  }

  zoo_rs_init(&rsHandle_, getDataPath().c_str(), getConfig().dump().c_str());
}

void RerunForwarder::onImage(const std::string &cameraName, const std::string & /*channel*/,
                             std::shared_ptr<const zoo_msgs::msg::Image12m> msg) {
  nvtx3::scoped_range nvtxLabel{"rerun_image"};
  // RCLCPP_INFO(get_logger(), "Received image %s (id=%lu)", cameraName.c_str(), msg->header.frame_id);

  // Store the image in the cache to use when we get the detections
  CameraData &cameraData = *imageCaches_[cameraName];

  auto dataLock = std::unique_lock<std::mutex>{cameraData.mutex};
  if (cameraData.images.isFull()) {
    while (cameraData.lastDetectionFrameId < cameraData.images.front()->header.frame_id) {
      // The queue is full and the oldest frame in it is newer than the last detection.
      // Block and wait until the detections catch up.
      dataLock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      // Check if the app is shutting down so we don't spin forever
      const bool shouldQuit = !rclcpp::contexts::get_global_default_context()->shutdown_reason().empty();
      if (shouldQuit) {
        return;
      }
      dataLock.lock();
    }
  }
  cameraData.images.pushImage(std::move(msg));
}

void RerunForwarder::onDetection(const std::string &cameraName, const std::string &channel,
                                 const zoo_msgs::msg::Detection &msg) {
  nvtx3::scoped_range nvtxLabel{"rerun_detection"};

  // Find image
  std::shared_ptr<const zoo_msgs::msg::Image12m> image;
  {
    CameraData &cameraData = *imageCaches_[cameraName];
    auto dataLock = std::unique_lock<std::mutex>{cameraData.mutex};
    cameraData.lastDetectionFrameId = msg.header.frame_id;
    image = cameraData.images.popImage(msg.header.frame_id);
  }
  if (image != nullptr) {
    // Image with same frame id found
    // Forward and clear cache
    zoo_rs_image_callback(rsHandle_, cameraName.c_str(), channel.c_str(), image.get());
  }

  // Forward to rerun
  zoo_rs_detection_callback(rsHandle_, cameraName.c_str(), channel.c_str(), &msg);
}

void RerunForwarder::onTrackState(const std::string &cameraName, const std::string &channel,
                                  const zoo_msgs::msg::TrackState &msg) {
  nvtx3::scoped_range nvtxLabel{"rerun_track_state"};

  // Forward to rerun
  zoo_rs_track_state_callback(rsHandle_, cameraName.c_str(), channel.c_str(), &msg);
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::RerunForwarder)