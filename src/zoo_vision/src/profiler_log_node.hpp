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

#include <rclcpp/node.hpp>
#include <rclcpp/timer.hpp>

namespace zoo {

class ProfilerLogNode : public rclcpp::Node {
public:
  explicit ProfilerLogNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  void onTimer();

private:
  std::shared_ptr<rclcpp::WallTimer<rclcpp::VoidCallbackType>> timer0_;
  std::shared_ptr<rclcpp::WallTimer<rclcpp::VoidCallbackType>> timer_;
};
} // namespace zoo