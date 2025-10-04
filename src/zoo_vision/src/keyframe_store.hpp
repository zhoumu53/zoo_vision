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
#include <Eigen/Dense>

#include <span>

namespace zoo {

using TKeyframeIndex = uint32_t;

class KeyframeStore {
public:
  constexpr static uint32_t MAX_KEYFRAME_COUNT = 40;

  explicit KeyframeStore();

  uint32_t getCount() const { return keyframeCount_; }
  at::Tensor getKeyframeImage(TKeyframeIndex i) const { return keyframeImages_[i]; }
  std::optional<TKeyframeIndex> maybeAddKeyframe(const at::Tensor &image_u8, const at::Tensor &embedding);

private:
  using TSimilaritiesVector = Eigen::Vector<float32_t, MAX_KEYFRAME_COUNT>;

  void replaceKeyframe(TKeyframeIndex replaceIdx, const at::Tensor &image_u8, const at::Tensor &embedding,
                       const at::Tensor &embeddingsNorm, const TSimilaritiesVector &newSimilarities);
  void findMostSimilarKeyframe();

  at::DeviceType device_ = at::kCPU;

  std::array<at::Tensor, MAX_KEYFRAME_COUNT> keyframeImages_;

  uint32_t keyframeCount_ = 0;
  at::Tensor keyframeEmbeddings_;     // [MAX_KEYFRAME_COUNT, EMBEDDING_FLAT_COUNT]
  at::Tensor keyframeEmbeddingNorms_; // [MAX_KEYFRAME_COUNT]

  Eigen::MatrixXf similarities_;
  float32_t mostSimilarScore_ = 1;            // Start with maximum similarity case we don't have keyframes
  TKeyframeIndex mostSimilarKeyframeIdx_ = 0; // Start by replacing the first (non-existent) keyframe
};

} // namespace zoo