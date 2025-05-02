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

#include "zoo_vision/types.hpp"

#include <ATen/core/Tensor.h>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

namespace zoo {

struct SegmenterResult {
  at::Tensor masks; // This must be initialized with the correct dimensions. It can be a mapped location into the output
                    // message buffer.
  std::vector<Eigen::AlignedBox2f> bboxesInDetection;
};

class ISegmenter {
public:
  virtual ~ISegmenter() = default;
  virtual Vector2i getDetectionImageSize() const = 0;
  virtual void onImage(SegmenterResult &result, const at::Tensor &imageGpu,
                       const cv::Mat &imageCpu /*TODO: remove imageCpu*/) = 0;
};

} // namespace zoo