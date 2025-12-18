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
#include "zoo_vision/image_rate_limiter.hpp"

#include <rclcpp/contexts/default_context.hpp>

#include <iostream>
#include <thread>

namespace zoo {
namespace {
bool shouldQuit() { return !rclcpp::contexts::get_global_default_context()->shutdown_reason().empty(); }
} // namespace
void ImageRateLimiter::addToQueue() {
  std::unique_lock<std::mutex> lock(mutex_);
  count++;
}

void ImageRateLimiter::waitForProcessing() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (count >= MAX_QUEUE_SIZE) {
    // std::cout << "Image rate limiter waiting (count=" << count << ")..." << std::endl;
    // Wait for most images to be processed
    condition_.wait(lock, [this] { // Check if the app is shutting down so we don't spin forever
      return count < 2 || shouldQuit();
    });
    if (shouldQuit()) {
      return;
    }
    // std::cout << "Image rate limiter wait done" << std::endl;
  }
}

void ImageRateLimiter::signalProcessingComplete() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    count--;
    // std::cout << "Signal process complete (count=" << count << ")..." << std::endl;
  }
  condition_.notify_one(); // Notify one waiting consumer
}

std::unordered_map<std::string, std::unique_ptr<ImageRateLimiter>> gCameraLimiters;

} // namespace zoo