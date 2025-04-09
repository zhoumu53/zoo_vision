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
#include "zoo_msgs/msg/detection.hpp"
#include "zoo_msgs/msg/image12m.hpp"
#include "zoo_msgs/msg/track_state.hpp"
#include "zoo_vision/image_queue.hpp"

#include <image_transport/image_transport.hpp>

#include <deque>
#include <unordered_map>

namespace zoo {
class RerunForwarder : public rclcpp::Node {
public:
  explicit RerunForwarder(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  void onImage(const std::string &cameraTopic, const std::string &channel,
               std::shared_ptr<const zoo_msgs::msg::Image12m> msg);
  void onDetection(const std::string &cameraTopic, const std::string &channel, const zoo_msgs::msg::Detection &msg);
  void onTrackState(const std::string &cameraTopic, const std::string &channel, const zoo_msgs::msg::TrackState &msg);

  std::unordered_map<std::string, std::unique_ptr<ImageQueue>> imageCaches_;

  void *rsHandle_;
  std::vector<std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Image12m>>> imageSubscribers_;
  std::vector<std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Detection>>> detectionSubscribers_;
  std::vector<std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::TrackState>>> trackStateSubscribers_;
};
} // namespace zoo