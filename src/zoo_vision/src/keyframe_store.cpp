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

#include "zoo_vision/keyframe_store.hpp"

#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/linalg_norm.h>

#include <cmath>
#include <ranges>

using namespace at::indexing;

namespace zoo {

namespace {
constexpr static uint32_t EMBEDDING_FLAT_COUNT =
    197 * 768; // ViT Embedding shape from huggingface google/vit-base-patch16-224
} // namespace

KeyframeStore::KeyframeStore()
    : keyframeEmbeddings_{at::zeros({MAX_KEYFRAME_COUNT, EMBEDDING_FLAT_COUNT},
                                    at::TensorOptions(at::kCUDA).dtype(at::kFloat))},
      keyframeEmbeddingNorms_{at::zeros({MAX_KEYFRAME_COUNT}, at::TensorOptions(at::kCUDA).dtype(at::kFloat))},
      similarities_{Eigen::MatrixXf::Ones(MAX_KEYFRAME_COUNT, MAX_KEYFRAME_COUNT)} {}

std::optional<TKeyframeIndex> KeyframeStore::maybeAddKeyframe(const at::Tensor &image_u8,
                                                              const at::Tensor &embeddings) {
  assert(image_u8.size(0) == 3);                // Channels
  assert(image_u8.size(1) == image_u8.size(1)); // Square aspect ratio

  at::Tensor embeddingsFlat = embeddings.reshape({-1});
  assert(embeddingsFlat.size(0) == EMBEDDING_FLAT_COUNT);

  at::Tensor embeddingsNorm = at::linalg_norm(embeddingsFlat);

  // Calculate similarities between new image and keyframes
  TSimilaritiesVector newSimilarities;
  newSimilarities.setZero();

  float32_t maxSimilarity = 0;
  if (keyframeCount_ > 0) {
    // Calculate similarities on gpu
    // but only for valid keyframes
    const at::Tensor validKeyframeEmbeddings = keyframeEmbeddings_.index({Slice(0, keyframeCount_)});
    const at::Tensor validKeyframeEmbeddingNorms = keyframeEmbeddingNorms_.index({Slice(0, keyframeCount_)});

    const at::Tensor newSimilaritiesGpu = at::sum((validKeyframeEmbeddings * embeddingsFlat.unsqueeze(0)), 1) /
                                          (embeddingsNorm * validKeyframeEmbeddingNorms);
    const at::Tensor newSimilaritiesCpu = newSimilaritiesGpu.cpu();
    for (uint32_t i = 0; i < keyframeCount_; ++i) {
      const float32_t value = newSimilaritiesCpu[i].item<float32_t>();
      if (value > maxSimilarity) {
        maxSimilarity = value;
      }
      newSimilarities[i] = value;
    }
  }
  // Complete newSimilarities for non-existing keyframes
  for (uint32_t i = keyframeCount_; i < MAX_KEYFRAME_COUNT; ++i) {
    newSimilarities[i] = 0;
  }

  constexpr float32_t SIMILARITY_THRESHOLD = 0.8f;
  if (maxSimilarity > SIMILARITY_THRESHOLD) {
    // New image is too similar to existing keyframes
    return std::nullopt;
  }

  if (maxSimilarity >= mostSimilarScore_) {
    // New image is more similar than existing keyframes
    return std::nullopt;
  }

  // Replace keyframe
  const auto replaceIdx = mostSimilarKeyframeIdx_;
  replaceKeyframe(replaceIdx, image_u8, embeddingsFlat, embeddingsNorm, newSimilarities);
  return replaceIdx;
}

void KeyframeStore::replaceKeyframe(TKeyframeIndex replaceIdx, const at::Tensor &image_u8, const at::Tensor &embeddings,
                                    const at::Tensor &embeddingsNorm, const TSimilaritiesVector &newSimilarities) {
  if (replaceIdx >= keyframeCount_) {
    assert(replaceIdx == keyframeCount_);
    keyframeCount_ += 1;
  }

  keyframeEmbeddings_[replaceIdx].copy_(embeddings);
  keyframeEmbeddingNorms_[replaceIdx].copy_(embeddingsNorm);

  // Update similarities
  for (uint32_t i = 0; i < replaceIdx; ++i) {
    similarities_(replaceIdx, i) = newSimilarities[i];
    similarities_(i, replaceIdx) = newSimilarities[i];
  }
  for (uint32_t i = replaceIdx + 1; i < MAX_KEYFRAME_COUNT; ++i) {
    similarities_(replaceIdx, i) = newSimilarities[i - 1];
    similarities_(i, replaceIdx) = newSimilarities[i - 1];
  }
  similarities_(replaceIdx, replaceIdx) = 0;

  findMostSimilarKeyframe();

  // Update image for ui
  keyframeImages_[replaceIdx] = image_u8.to(at::kByte);
}

void KeyframeStore::findMostSimilarKeyframe() {
  Eigen::Index index0, index1;
  mostSimilarScore_ = similarities_.maxCoeff(&index0, &index1);

  // Compare the l2-norm of both indices to find the most similar keyframe
  const float32_t sqNnorm0 = similarities_.col(index0).squaredNorm();
  const float32_t sqNnorm1 = similarities_.col(index1).squaredNorm();
  mostSimilarKeyframeIdx_ = (sqNnorm0 > sqNnorm1) ? index0 : index1;
}

} // namespace zoo
