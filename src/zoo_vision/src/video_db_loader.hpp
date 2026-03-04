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
#pragma once

#include "zoo_msgs/msg/image12m.hpp"
#include "zoo_vision/image_rate_limiter.hpp"
#include "zoo_vision/profiler.hpp"
#include "zoo_vision/stats.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <rclcpp/rclcpp.hpp>

#include <filesystem>
#include <memory>

namespace zoo {

class VideoDBLoader : public rclcpp::Node {
public:
  explicit VideoDBLoader(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  using Clock = std::chrono::system_clock;

  struct VideoInfo {
    std::filesystem::path videoFile;
    Clock::time_point startTime;
    Clock::time_point endTime;
  };

  struct CameraData {
    std::vector<VideoInfo> videoList_;

    cv::Size2i frameSize;
    std::vector<VideoInfo>::const_iterator currentVideo_;
    std::optional<Clock::time_point> videoStartTime_;
    std::unique_ptr<cv::VideoCapture> videoStream_;

    ImageRateLimiter *rateLimiter;
    std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::Image12m>> publisher_;
  };

  void loadVideoDatabase(const std::filesystem::path &database, std::span<const std::string> enabledCameras);
  void loadVideo(const std::string &cameraName, CameraData &cameraData, const Clock::time_point time);
  void loadNextVideo(const std::string &cameraName, CameraData &cameraData);

  void openVideo(const std::string &cameraName, CameraData &cameraData, const VideoInfo &info);

  std::optional<Clock::time_point> findNextValidReplayTime() const;

  cv::Mat3b loadImage(CameraData &cameraData, zoo_msgs::msg::Image12m &msg);
  void onTimer();

  int skipFrameCount_;
  Clock::time_point replayNow_;
  std::filesystem::path videoRootPath_;

  std::unordered_map<std::string, CameraData> cameras_;

  rclcpp::CallbackGroup::SharedPtr timerCbGroup_;
  rclcpp::TimerBase::SharedPtr timer_;

  RunningStats imageCountStats_;

  std::stack<ProfilerSectionData *> profilerStack_;
  ProfileTicOnly profileTic_;
};
} // namespace zoo
