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

#include "zoo_vision/segmenter_interface.hpp"
#include "zoo_vision/timings.hpp"

#include <nlohmann/json_fwd.hpp>
#include <rclcpp/rclcpp.hpp>

#include <filesystem>

namespace zoo {

class SegmenterFake : public ISegmenter {
public:
  explicit SegmenterFake(int nameIndex);
  ~SegmenterFake();

  void setImageSize(Vector2i size) override { detectionImageSize_ = size; }
  Vector2i getDetectionImageSize() const override { return detectionImageSize_; }

  void onImage(SegmenterResult &result, const at::Tensor &imageGpu,
               const cv::Mat &imageCpu /*TODO: remove imageCpu*/) override;

private:
  const rclcpp::Logger &get_logger() const { return logger_; }

  std::string name_;
  rclcpp::Logger logger_;

  Vector2i detectionImageSize_;
};
} // namespace zoo