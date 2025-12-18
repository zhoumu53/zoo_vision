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

#include "zoo_vision/zoo_camera.hpp"

#include "zoo_vision/utils.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>

#include <chrono>

using namespace std::chrono_literals;

namespace {
const std::string DEFAULT_VIDEO_NAME = "sample_video.mp4";
}
namespace zoo {

ZooCamera::ZooCamera(const rclcpp::NodeOptions &options, int nameIndex)
    : Node(std::format("input_camera_{}", nameIndex), options) {
  this->cameraName_ = declare_parameter<std::string>("camera_name");
  RCLCPP_INFO(get_logger(), "Starting zoo_camera for %s", cameraName_.c_str());

  const nlohmann::json &config = getConfig();
  const bool useLiveStream = config["live_stream"].get<bool>();

  if (useLiveStream) {
    const auto &streamConfig = config["cameras"][cameraName_]["stream"];
    const std::string protocol = streamConfig["protocol"].get<std::string>();
    const std::string address = streamConfig["ip"].get<std::string>();
    const std::string url = streamConfig["url"].get<std::string>();
    // TODO: oh god, not hardcoded, no :(
    const std::string username = "daniel";
    const std::string pwd = "PwU82-!MnG";

    videoUrl_ = std::format("{}://{}:{}@{}/{}", protocol, username, pwd, address, url);
    RCLCPP_INFO(get_logger(), "Connecting to camera %s at %s://%s/%s", cameraName_.c_str(), protocol.c_str(),
                address.c_str(), url.c_str());
  } else {
    videoUrl_ = getDataPath() / "cameras" / cameraName_ / "sample.mp4";
    RCLCPP_INFO(get_logger(), "Loading video from %s", videoUrl_.c_str());
  }

  openCamera();

  publisher_ = rclcpp::create_publisher<zoo_msgs::msg::Image12m>(*this, cameraName_ + "/image", 10);
  timer_ = create_wall_timer(30ms, [this]() { this->onTimer(); });
}

void ZooCamera::openCamera() {
  if (cvStream_.isOpened()) {
    cvStream_.release();
  }
  bool ok = cvStream_.open(videoUrl_);
  if (ok) {
    frameWidth_ = cvStream_.get(cv::CAP_PROP_FRAME_WIDTH);
    frameHeight_ = cvStream_.get(cv::CAP_PROP_FRAME_HEIGHT);
    RCLCPP_INFO(get_logger(), "Opened video (%dx%d)", frameWidth_, frameHeight_);
  } else {
    RCLCPP_ERROR(get_logger(), "Failed to open camera %s", cameraName_.c_str());
    frameWidth_ = frameHeight_ = 500;
  }
  assert(frameHeight_ * frameWidth_ * 3 <= zoo_msgs::msg::Image12m::DATA_MAX_SIZE);
  frameIndex_ = 0;
}

void ZooCamera::onTimer() {
  auto msg = std::make_unique<zoo_msgs::msg::Image12m>();
  msg->header.stamp = now();
  setMsgString(msg->encoding, "rgb8");
  msg->width = frameWidth_;
  msg->height = frameHeight_;
  msg->is_bigendian = false;
  msg->step = msg->width * 3 * sizeof(char);

  cv::Mat3b image(frameHeight_, frameWidth_, reinterpret_cast<cv::Vec3b *>(&msg->data));

  if (!cvStream_.isOpened()) {
    openCamera();
    if (!cvStream_.isOpened()) {
      return;
    }
  }

  cvStream_ >> image;
  if (image.empty()) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "Failed to get image from camera %s", cameraName_.c_str());
    // Maybe camera disconnected or maybe video ran out, reopen to try and fix it
    openCamera();
    return;
  }

  // TODO: converting BGR->RGB like this is inefficient!
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  msg->header.frame_id = frameIndex_;

  frameIndex_++;
  publisher_->publish(std::move(msg));
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::ZooCamera)