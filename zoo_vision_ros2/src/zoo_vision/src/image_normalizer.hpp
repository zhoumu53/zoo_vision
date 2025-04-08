
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

#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>

namespace zoo {

class ImageNormalizer {
public:
  explicit ImageNormalizer();

  at::Tensor normalize(const at::Tensor &image_u8) const {
    at::Tensor image_f32 = image_u8.to(at::kFloat);
    at::Tensor imageNorm = (image_f32 - preprocessMean_) / preprocessStd_;
    return imageNorm;
  }

  at::Tensor denormalize(const at::Tensor &image_f32) const {
    return (image_f32 * preprocessStd_ + preprocessMean_).to(at::kByte);
  }

private:
  at::Tensor preprocessMean_;
  at::Tensor preprocessStd_;
};
} // namespace zoo