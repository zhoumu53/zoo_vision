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

#include "zoo_vision/video_loader.hpp"

#include "zoo_vision/utils.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <chrono>
#include <fstream>
#include <spanstream>

using namespace std::chrono_literals;

namespace zoo {
namespace {

std::chrono::system_clock::time_point parseTime(std::string_view timeStr) {
  // using Clock = std::chrono::system_clock;
  std::tm parsedTime = {};
  std::ispanstream ss(timeStr);

  constexpr auto DATE_FORMAT = "%Y-%m-%dT%H:%M:%S";
  ss >> std::get_time(&parsedTime, DATE_FORMAT);

  std::chrono::system_clock::time_point parsedPoint = std::chrono::system_clock::from_time_t(std::mktime(&parsedTime));

  return parsedPoint;
}

std::chrono::system_clock::duration parseDuration(std::string_view timeStr) {
  using Clock = std::chrono::system_clock;
  std::tm t = {};
  std::ispanstream ss(timeStr);

  constexpr auto DATE_FORMAT = "%H:%M";
  ss >> std::get_time(&t, DATE_FORMAT);

  Clock::duration dur = std::chrono::hours(t.tm_hour) + std::chrono::hours(t.tm_min);
  return dur;
}

} // namespace

VideoLoader::VideoLoader(const rclcpp::NodeOptions &options) : Node("video_loader", options) {
  RCLCPP_INFO(get_logger(), "Starting video_loader");

  const nlohmann::json &config = getConfig();

  // Load database
  std::vector<std::string> enabledCameras = config["enabled_cameras"];
  std::filesystem::path videoDatabase = config["video_db"];
  loadVideoDatabase(videoDatabase, enabledCameras);

  // Advance replay clock
  replayNow_ += parseDuration(config["video_start_time"].get<std::string>());
  RCLCPP_INFO(get_logger(), "Replay start time: %s", std::format("{:%Y-%m-%d %T}", replayNow_).c_str());

  // Load videos
  for (auto &[cameraName, cameraData] : cameras_) {
    cameraData.publisher_ = rclcpp::create_publisher<zoo_msgs::msg::Image12m>(*this, cameraName + "/image", 10);
    loadVideo(cameraData, replayNow_);
  }

  timer_ = create_wall_timer(40ms, [this]() { this->onTimer(); });
}

void VideoLoader::loadVideoDatabase(const std::filesystem::path &database,
                                    const std::span<const std::string> enabledCameras) {
  const auto videoRootPath = database.parent_path();

  std::ifstream f(database);
  nlohmann::json databaseJson = nlohmann::json::parse(f);

  // Set replay clock to database start time
  const std::string databaseStartTime = databaseJson["start_time"];
  replayNow_ = parseTime(databaseStartTime);

  // Read all videos
  for (const std::string_view camera : enabledCameras) {
    nlohmann::json cameraJson = databaseJson["cameras"][camera];

    const auto pair = cameras_.emplace(std::make_pair(camera, CameraData()));
    CameraData &cameraData = pair.first->second;

    for (auto [videoJson, startTimeJson, endTimeJson] :
         std::ranges::views::zip(cameraJson["videos"], cameraJson["start_times"], cameraJson["end_times"])) {
      cameraData.videoList_.emplace_back(videoRootPath / videoJson.get<std::string>(),
                                         parseTime(startTimeJson.get<std::string>()),
                                         parseTime(endTimeJson.get<std::string>()));
    }
  }
}

void VideoLoader::loadVideo(CameraData &cameraData, const Clock::time_point time) {
  for (const auto &[videoFile, startTime, endTime] : cameraData.videoList_) {
    if (startTime <= time && time < endTime) {
      cv::VideoCapture cvVideo;
      const bool ok = cvVideo.open(videoFile);
      if (ok) {
        cameraData.frameSize = cv::Size2i{static_cast<int>(cvVideo.get(cv::CAP_PROP_FRAME_WIDTH)),
                                          static_cast<int>(cvVideo.get(cv::CAP_PROP_FRAME_HEIGHT))};
        cameraData.videoStartTime_ = startTime;
        cameraData.videoStream_ = std::move(cvVideo);
        RCLCPP_INFO(get_logger(), "Loaded video %s (%dx%d) start time %s", videoFile.filename().c_str(),
                    cameraData.frameSize.width, cameraData.frameSize.height,
                    std::format("{:%Y-%m-%d %T}", *cameraData.videoStartTime_).c_str());

        // Adjust time
        constexpr float FPS = 25;
        const auto offsetMs = std::chrono::duration_cast<std::chrono::milliseconds>(startTime - time).count();
        const auto offsetFrames = FPS * offsetMs / 1000;
        cameraData.videoStream_->set(cv::CAP_PROP_POS_FRAMES, offsetFrames);

      } else {
        RCLCPP_ERROR(get_logger(), "Failed to open video %s", videoFile.c_str());
      }
    }
  }
}

void VideoLoader::onTimer() {
  for (auto &[cameraName, cameraData] : cameras_) {
    if (!cameraData.videoStartTime_.has_value() || *cameraData.videoStartTime_ > replayNow_) {
      continue;
    }

    auto msg = std::make_unique<zoo_msgs::msg::Image12m>();
    msg->header.stamp = now();
    setMsgString(msg->header.frame_id, std::to_string(frameIndex_).c_str());
    setMsgString(msg->encoding, sensor_msgs::image_encodings::RGB8);
    msg->width = cameraData.frameSize.width;
    msg->height = cameraData.frameSize.height;
    msg->is_bigendian = false;
    msg->step = msg->width * 3 * sizeof(char);

    cv::Mat3b image = wrapMat3bFromMsg(*msg);

    assert(cameraData.videoStream_.has_value());
    auto &cvVideo = *cameraData.videoStream_;

    cvVideo >> image;
    if (image.empty()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "Video for %s EOF", cameraName.c_str());

      cameraData.videoStartTime_.reset();
      cameraData.videoStream_.reset();
      loadVideo(cameraData, replayNow_);
    }

    // TODO: converting BGR->RGB like this is inefficient!
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    if (image.empty()) {
      setMsgString(msg->header.frame_id, "error");
      image.setTo(cv::Vec3b(0, 0, 255));
    }

    cameraData.publisher_->publish(std::move(msg));
  }

  replayNow_ += std::chrono::milliseconds(40);
  frameIndex_++;
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::VideoLoader)