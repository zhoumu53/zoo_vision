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

#include <date/chrono_io.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>

#include <chrono>
#include <fstream>
#include <spanstream>

using namespace std::chrono_literals;

namespace zoo {

VideoLoader::VideoLoader(const rclcpp::NodeOptions &options) : Node("video_loader", options) {
  RCLCPP_INFO(get_logger(), "Starting video_loader");

  const nlohmann::json &config = getConfig();

  std::vector<std::string> videoFiles = config["videos"];
  std::vector<std::string> enabledCameras = config["enabled_cameras"];
  loadVideos(videoFiles, enabledCameras);

  // Init replay clock
  replayStart_ = replayNow_ = Clock::now();
  RCLCPP_INFO(get_logger(), "Replay start time: %s", std::format("{:%Y-%m-%d %T}", replayNow_).c_str());

  timer_ = create_wall_timer(40ms, [this]() { this->onTimer(); });
}

void VideoLoader::loadVideos(const std::span<const std::string> videoFiles,
                             const std::span<const std::string> enabledCameras) {
  CHECK_EQ(videoFiles.size(), enabledCameras.size());
  // Read all videos
  for (auto const &[cameraName, videoFile] : std::views::zip(enabledCameras, videoFiles)) {
    const auto pair = cameras_.emplace(std::make_pair(cameraName, CameraData()));
    CameraData &cameraData = pair.first->second;
    cameraData.videoFile = videoFile;

    cv::VideoCapture cvVideo;
    const bool ok = cvVideo.open(videoFile);
    if (ok) {
      if (!gCameraLimiters.empty()) {
        cameraData.rateLimiter = gCameraLimiters[cameraName].get();
      }
      cameraData.publisher_ = rclcpp::create_publisher<zoo_msgs::msg::Image12m>(*this, cameraName + "/image", 10);
      cameraData.frameSize = cv::Size2i{static_cast<int>(cvVideo.get(cv::CAP_PROP_FRAME_WIDTH)),
                                        static_cast<int>(cvVideo.get(cv::CAP_PROP_FRAME_HEIGHT))};
      cameraData.videoStream_ = std::move(cvVideo);
      RCLCPP_INFO(get_logger(), "Loaded video %s", videoFile.c_str());
      RCLCPP_INFO(get_logger(), "Resolution=%dx%d, now=%s, start time=%s", cameraData.frameSize.width,
                  cameraData.frameSize.height, std::format("{:%Y-%m-%d %T}", replayNow_).c_str(),
                  std::format("{:%Y-%m-%d %T}", replayNow_).c_str());
    } else {
      using namespace std::literals;
      throw std::runtime_error("Failed to open video "s + videoFile);
    }
  }
}

void VideoLoader::loadImage(CameraData &cameraData, cv::Mat3b &image) {
  if (!cameraData.videoStream_.has_value()) {
    image = cv::Mat3b{};
    return;
  }

  auto &cvVideo = *cameraData.videoStream_;

  cvVideo >> image;
  if (image.empty()) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "Video for %s EOF", cameraData.videoFile.c_str());
    cameraData.videoStream_.reset();
  }
}

void VideoLoader::onTimer() {
  std::optional<Clock::time_point> newReplayTime;
  bool framePublished = false;

  for (auto &[cameraName, cameraData] : cameras_) {
    if (!cameraData.videoStream_.has_value()) {
      // No video to load, ignore camera from now on
      continue;
    }

    auto msg = std::make_unique<zoo_msgs::msg::Image12m>();
    msg->header.stamp =
        rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(replayNow_.time_since_epoch()).count());
    setMsgString(msg->header.video_filename, cameraData.videoFile.stem().c_str());
    setMsgString(msg->header.frame_id, std::to_string(frameIndex_).c_str());
    setMsgString(msg->encoding, "rgb8");
    msg->width = cameraData.frameSize.width;
    msg->height = cameraData.frameSize.height;
    msg->is_bigendian = false;
    msg->step = msg->width * 3 * sizeof(char);

    cv::Mat3b image = wrapMat3bFromMsg(*msg);
    loadImage(cameraData, image);
    if (image.empty()) {
      continue;
    }

    if (!newReplayTime.has_value() && cameraData.videoStream_.has_value()) {
      // Calculate replay time based on how much we've advanced in the video
      const auto offsetMs = cameraData.videoStream_->get(cv::CAP_PROP_POS_MSEC);
      newReplayTime = replayStart_ + std::chrono::milliseconds(static_cast<int64_t>(offsetMs));
      // assert(newReplayTime.value() >= replayNow_);
    }

    // TODO: converting BGR->RGB like this is inefficient!
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cameraData.publisher_->publish(std::move(msg));
    framePublished = true;

    if (cameraData.rateLimiter != nullptr) {
      cameraData.rateLimiter->waitForProcessing();
    }
  }
  if (framePublished) {
    frameIndex_ += 1;
  }

  if (!newReplayTime.has_value()) {
    assert(framePublished == false);
    RCLCPP_INFO(get_logger(), "End of all videos");
    std::exit(0);
  }

  assert(newReplayTime.has_value());
  replayNow_ = *newReplayTime;

  static int64_t minutesLastLog = 0;
  const int64_t minutesNow = std::chrono::duration_cast<std::chrono::minutes>(replayNow_.time_since_epoch()).count();
  if (abs(minutesNow - minutesLastLog) > 5) {
    minutesLastLog = minutesNow;
    RCLCPP_INFO(get_logger(), "Replay time: %s", std::format("{}", replayNow_).c_str());
  }
}

} // namespace zoo
