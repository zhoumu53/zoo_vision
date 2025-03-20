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

#include "zoo_vision/patch_cropper.hpp"

#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/utils.hpp"

#include <rclcpp/time.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace at::indexing;

namespace zoo {

PatchCropper::PatchCropper() {}

void PatchCropper::extractCrops(torch::Tensor &patches, const torch::Tensor &imageGpu,
                                const float scale_image_from_detection,
                                const std::span<const zoo_msgs::msg::BoundingBox2D> bboxes) {
  constexpr int CROP_SIZE = 256;
  const int detectionCount = bboxes.size();
  const int channels = 3;

  patches = at::zeros({detectionCount, channels, CROP_SIZE, CROP_SIZE}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));

  // Extract crops
  for (const auto &[i, bbox] : std::views::enumerate(bboxes)) {
    const float32_t bboxAspect = static_cast<float32_t>(bbox.half_size[0]) / static_cast<float32_t>(bbox.half_size[1]);
    Eigen::Vector2f bboxCenter = Eigen::Vector2f{bbox.center[0], bbox.center[1]};
    Eigen::Vector2f bboxHalfSize = Eigen::Vector2f{bbox.half_size[0], bbox.half_size[1]};

    Eigen::Vector2f center = bboxCenter * scale_image_from_detection;
    Eigen::Vector2f half_size = bboxHalfSize * scale_image_from_detection;

    Eigen::Vector2f corner0 = center - half_size;
    Eigen::Vector2f corner1 = center + half_size;
    const auto bboxPatch = imageGpu.index({
        None,
        Slice(),
        Slice(corner0[1], corner1[1]),
        Slice(corner0[0], corner1[0]),
    });

    const auto rescaleSize = (bboxAspect >= 1.0f) ? Eigen::Vector2i(CROP_SIZE, std::round(CROP_SIZE / bboxAspect))
                                                  : Eigen::Vector2i(std::round(CROP_SIZE * bboxAspect), CROP_SIZE);

    namespace F = torch::nn::functional;
    const auto interpolateOpts = F::InterpolateFuncOptions()
                                     .size({{rescaleSize.y(), rescaleSize.x()}})
                                     .mode(torch::kBilinear)
                                     .antialias(true)
                                     .align_corners(false);
    assert(std::holds_alternative<torch::enumtype::kBilinear>(interpolateOpts.mode()));
    auto rescaledPatch = at::ones({1, channels, rescaleSize.y(), rescaleSize.x()});
    rescaledPatch = F::interpolate(bboxPatch, interpolateOpts);

    const Eigen::Vector2i c0 = (Eigen::Vector2i(CROP_SIZE, CROP_SIZE) - rescaleSize) / 2;
    const Eigen::Vector2i c1 = c0 + rescaleSize;

    auto patch_i = patches[i].index({Slice(), Slice(c0.y(), c1.y()), Slice(c0.x(), c1.x())});
    patch_i.copy_(rescaledPatch[0]);
  }
}

} // namespace zoo
