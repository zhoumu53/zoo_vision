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

#include <deque>
#include <mutex>
#include <unordered_map>

namespace zoo {
class ImageQueue {
public:
  using SharedPtrImage = std::shared_ptr<const zoo_msgs::msg::Image12m>;
  ImageQueue();

  void pushImage(std::shared_ptr<const zoo_msgs::msg::Image12m> msg);
  SharedPtrImage popImage(uint64_t id);

  SharedPtrImage &front() { return queue_.front(); }
  bool isFull() const { return queue_.size() >= maxCacheSize_; }

private:
  size_t maxCacheSize_ = 20;

  std::mutex queueMutex_;
  std::deque<SharedPtrImage> queue_;
};
} // namespace zoo