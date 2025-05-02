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
  std::vector<Eigen::AlignedBox2f> boxes;
};

class ISegmenter {
public:
  virtual ~ISegmenter() = default;

  virtual void readConfig(const nlohmann::json &config) = 0;
  virtual void onImage(const cv::Mat &imageCpu, const at::Tensor &imageGpu) = 0;
};
} // namespace zoo