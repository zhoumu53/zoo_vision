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

#include "rclcpp/rclcpp.hpp"
#include "zoo_msgs/msg/image12m.hpp"
#include <opencv2/highgui/highgui.hpp>

#include <filesystem>
#include <memory>

namespace zoo {

class VideoLoader : public rclcpp::Node {
public:
  explicit VideoLoader(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

private:
  using Clock = std::chrono::system_clock;

  struct CameraData {
    std::filesystem::path videoFile;

    cv::Size2i frameSize;
    std::optional<cv::VideoCapture> videoStream_;
    std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::Image12m>> publisher_;
  };

  void loadVideos(std::span<const std::string> videoFiles, std::span<const std::string> enabledCameras);

  void loadImage(CameraData &cameraData, cv::Mat3b &img);
  void onTimer();

  Clock::time_point replayStart_;
  Clock::time_point replayNow_;
  size_t frameIndex_;

  std::unordered_map<std::string, CameraData> cameras_;
  rclcpp::TimerBase::SharedPtr timer_;
};
} // namespace zoo