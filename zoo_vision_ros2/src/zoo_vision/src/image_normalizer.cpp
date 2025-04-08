
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

#include "zoo_vision/image_normalizer.hpp"

#include <torch/torch.h>

namespace zoo {

ImageNormalizer::ImageNormalizer() {
  auto preprocessMeanData = std::array<float32_t, 3>({0.48500001430511475f, 0.4560000002384186f, 0.4059999883174896f});
  auto preprocessStdData = std::array<float32_t, 3>({0.2290000021457672f, 0.2239999920129776f, 0.22499999403953552f});

  preprocessMean_ =
      (at::from_blob(preprocessMeanData.data(), {3, 1, 1}, at::TensorOptions().dtype(at::kFloat)) * 255.0f)
          .to(torch::kCUDA);
  preprocessStd_ = (at::from_blob(preprocessStdData.data(), {3, 1, 1}, at::TensorOptions().dtype(at::kFloat)) * 255.0f)
                       .to(torch::kCUDA);
}

} // namespace zoo