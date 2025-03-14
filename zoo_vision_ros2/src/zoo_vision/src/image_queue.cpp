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

#include "zoo_vision/image_queue.hpp"

#include "zoo_vision/utils.hpp"

namespace zoo {

ImageQueue::ImageQueue() {}
// ImageQueue::ImageQueue(size_t maxCacheSize) : maxCacheSize_{maxCacheSize} {}

void ImageQueue::pushImage(std::shared_ptr<const zoo_msgs::msg::Image12m> msg) {
  // Store the image in the cache to use when we get the detections
  std::lock_guard<std::mutex> lock(queueMutex_);
  if (queue_.size() >= maxCacheSize_) {
    queue_.pop_front();
  }
  queue_.push_back(std::move(msg));
}

auto ImageQueue::popImage(std::string_view id) -> SharedPtrImage {
  // Find image
  SharedPtrImage image = nullptr;
  {
    std::lock_guard<std::mutex> lock(queueMutex_);

    auto matchingImageIt = queue_.begin();
    for (; matchingImageIt != queue_.end(); ++matchingImageIt) {
      const auto &image_i = **matchingImageIt;
      std::string_view id_i = getMsgString(image_i.header.frame_id);

      if (id.compare(id_i) == 0) {
        break;
      }
    }

    if (matchingImageIt != queue_.end()) {
      image = *matchingImageIt;

      // We can drop all images before this one
      auto eraseIt = queue_.begin();
      while (*eraseIt != image) {
        eraseIt = queue_.erase(eraseIt);
      }
      queue_.erase(eraseIt);
    } else {
      std::cout << std::format("Could not find image {} in cache.", id) << std::endl;
    }
  }

  return image;
}

} // namespace zoo
