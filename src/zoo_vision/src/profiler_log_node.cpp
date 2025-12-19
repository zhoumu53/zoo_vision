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
#include "zoo_vision/profiler_log_node.hpp"

#include "zoo_vision/profiler.hpp"

#include <rclcpp/create_timer.hpp>

namespace zoo {

ProfilerLogNode::ProfilerLogNode(const rclcpp::NodeOptions &options) : rclcpp::Node("profiler_log_node", options) {
  timer0_ = create_wall_timer(std::chrono::seconds(10), rclcpp::VoidCallbackType([this]() { this->onTimer(); }));
  timer_ = create_wall_timer(std::chrono::minutes(5), rclcpp::VoidCallbackType([this]() { this->onTimer(); }));
}

void ProfilerLogNode::onTimer() {
  if (timer0_) {
    timer0_->cancel();
    timer0_ = nullptr;
  }
  std::cout << "Profiling timings:\n" << Profiler::Instance() << std::endl;
  Profiler::Instance().reset();
}
} // namespace zoo