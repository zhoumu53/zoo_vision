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
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace at::indexing;

namespace zoo {

PatchCropper::PatchCropper() {}

void PatchCropper::extractCrops(torch::Tensor &patches, const torch::Tensor &image,
                                const Eigen::Vector2f &scale_image_from_detection,
                                const std::span<const zoo_msgs::msg::BoundingBox2D> bboxes) {
  constexpr int CROP_SIZE = 512;
  constexpr float CONTEXT_FACTOR = 1.1f;
  const int detectionCount = bboxes.size();
  const int channels = 3;

  const std::array<int64_t, 2> imageSize = {/*width*/ image.size(2), /*height*/ image.size(1)};
  patches = at::zeros({detectionCount, channels, CROP_SIZE, CROP_SIZE},
                      at::TensorOptions(image.device()).dtype(image.dtype()));

  // Extract crops
  for (const auto &[i, bbox] : std::views::enumerate(bboxes)) {
    const float32_t bboxAspect = static_cast<float32_t>(bbox.half_size[0]) / static_cast<float32_t>(bbox.half_size[1]);
    const Eigen::Vector2f bboxCenter = Eigen::Vector2f{bbox.center[0], bbox.center[1]};
    const Eigen::Vector2f bboxHalfSize = Eigen::Vector2f{bbox.half_size[0], bbox.half_size[1]};

    const Eigen::Vector2f center = bboxCenter.cwiseProduct(scale_image_from_detection);
    const Eigen::Vector2f half_size = bboxHalfSize.cwiseProduct(scale_image_from_detection) * CONTEXT_FACTOR;

    Eigen::Vector2f corner0 = center - half_size;
    Eigen::Vector2f corner1 = center + half_size;
    // Make sure corners are inside the image (because we increased patch size by CONTEXT_FACTOR)
    for (Eigen::Vector2f *pcorner : {&corner0, &corner1}) {
      auto &corner = *pcorner;
      for (int i : {0, 1}) {
        if (corner[i] < 0) {
          corner[i] = 0;
        } else if (corner[i] >= imageSize[i]) {
          corner[i] = imageSize[i] - 1;
        }
      }
    }
    const auto bboxPatch = image.index({
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
