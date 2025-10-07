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
#include <thread>

#include <iostream>

namespace zoo {
void ImageRateLimiter::waitForProcessing() {
  std::unique_lock<std::mutex> lock(mutex_);
  std::cout << "Waiting..." << std::endl;
  count++;

  condition_.wait(lock, [this] { return count == 0; }); // Wait all images are processed
  std::cout << "Wait done" << std::endl;
}

void ImageRateLimiter::signalProcessingComplete() {
  std::cout << "Signal process complete..." << std::endl;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    count--;
  }
  condition_.notify_one(); // Notify one waiting consumer
}

std::unordered_map<std::string, std::unique_ptr<ImageRateLimiter>> gCameraLimiters;

} // namespace zoo