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

#include "zoo_msgs/msg/bounding_box2_d.hpp"

#include <ATen/Tensor.h>
#include <Eigen/Dense>

#include <span>

namespace zoo {

using float32_t = float;

class PatchCropper {
public:
  explicit PatchCropper();

  void extractCrops(at::Tensor &patches, const at::Tensor &imageGpu, const Eigen::Vector2f &scale_image_from_detection,
                    const std::span<const zoo_msgs::msg::BoundingBox2D> bboxes);
};
} // namespace zoo