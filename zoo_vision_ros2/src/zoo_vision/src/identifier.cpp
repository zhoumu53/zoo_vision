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

#include "zoo_vision/identifier.hpp"

#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/utils.hpp"

#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>
#include <rclcpp/time.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string.h>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

Identifier::Identifier(int nameIndex, std::string cameraName, TrackMatcher &trackMatcher,
                       at::cuda::CUDAStream cudaStream)
    : name_{std::format("identifier_{}", nameIndex)}, logger_{rclcpp::get_logger(name_)}, cudaStream_{cudaStream},
      cameraName_{cameraName}, trackMatcher_{trackMatcher} {
  at::InferenceMode inferenceGuard;

  RCLCPP_INFO(get_logger(), "Starting identifier for %s", cameraName_.c_str());

  readConfig(getConfig());
}

void Identifier::readConfig(const nlohmann::json &config) {
  // Load model
  const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / config["models"]["identity"]);
  loadModel(modelPath);
  isStatefulModel_ = (modelPath.filename().string().contains("gru"));
  RCLCPP_INFO(get_logger(), "Is model stateful: %d", int(isStatefulModel_));
}

void Identifier::loadModel(const std::filesystem::path &modelPath) {
  RCLCPP_INFO(get_logger(), "Loading identity model from %s", modelPath.c_str());

  try {
    if (!std::filesystem::exists(modelPath)) {
      throw std::runtime_error("Model does not exist");
    }
    identityNetwork_ = torch::jit::load(modelPath, torch::kCUDA);
    identityNetwork_.eval();
  } catch (const std::exception &ex) {
    std::cout << "Error loading model from " << modelPath << std::endl;
    std::cout << "Exception: " << ex.what() << std::endl;
    std::terminate();
  }
  // DEBUG print model info
}

float calculateWeight(const std::span<const int> identities, const Eigen::MatrixXf &prior) {
  if (identities.size() < 2) {
    // Trivial weight when there is nothing to compare
    return 1;
  } else if (identities.size() == 2) {
    // Only two items, use prior directly
    return prior(identities[0], identities[1]);
  } else {
    // Many items, approximate prior with geometric mean
    float weight = 1;
    for (size_t i = 0; i < identities.size(); ++i) {
      for (size_t j = i + 1; j < identities.size(); ++j) {
        weight *= prior(i, j);
      }
    }
    return std::pow(weight, identities.size() - 1);
  }
}

TIdentity selectOptimalIdentities(const at::Tensor &probabilities) {
  assert(probabilities.dim() == 1);

  constexpr float32_t CONFIDENCE_THRESHOLD = 0.4f;

  // Force to max probability
  const auto &[maxProbs, maxIndices] = probabilities.topk(/*k*/ 2, /*dim*/ 0);
  const float32_t top1 = maxProbs[0].item<float32_t>();
  const float32_t top2 = maxProbs[1].item<float32_t>();
  if ((top1 - top2) >= CONFIDENCE_THRESHOLD) {
    const int maxIdx = maxIndices[0].item<int>();
    return maxIdx + 1; // Add one because class 0 is invalid
  } else {
    return INVALID_IDENTITY;
  }
}

void Identifier::callStatefulModel(at::Tensor &logitsGpu, const torch::Tensor &patches,
                                   const std::span<const TrackId> trackIds) {
  const int detectionCount = trackIds.size();
  constexpr size_t HIDDEN_LAYER_COUNT = 3;
  constexpr size_t HIDDEN_STATE_SIZE = 128;
  auto trackState0 = at::zeros({HIDDEN_LAYER_COUNT, detectionCount, HIDDEN_STATE_SIZE},
                               at::TensorOptions(at::kCUDA).dtype(at::kFloat));
  for (const auto &[i, trackId] : std::views::enumerate(trackIds)) {
    // Copy track state
    TrackData &track = trackMatcher_.getTrackData(trackId);
    if (track.identityState.has_value()) {
      trackState0.index({Slice(), i, Ellipsis}).copy_(track.identityState.value());
    }
  }

  auto patchesTime = patches.unsqueeze(1);
  at::Tensor trackState;
  {
    c10::IValue modelResult = identityNetwork_.forward({patchesTime, trackState0});
    c10::Dict<c10::IValue, c10::IValue> modelResultDict = modelResult.toGenericDict();

    logitsGpu = modelResultDict.at("logits").toTensor();
    logitsGpu = logitsGpu.squeeze(1); // Remove dummy time dimension

    trackState = modelResultDict.at("gru_state").toTensor();
  }

  // Remember track state
  for (const auto &[i, trackId] : std::views::enumerate(trackIds)) {
    TrackData &track = trackMatcher_.getTrackData(trackId);
    track.identityState.emplace(trackState.index({Slice(), i, Ellipsis}));
  }
}

void Identifier::callStatelessModel(at::Tensor &logitsGpu, const torch::Tensor &patches) {
  c10::IValue modelResult = identityNetwork_.forward({patches});
  if (modelResult.isTensor()) {
    logitsGpu = modelResult.toTensor();
  } else {
    auto dict = modelResult.toGenericDict();
    logitsGpu = dict.at("logits").toTensor();
    logitsGpu = logitsGpu.squeeze(1); // Remove dummy time dimension
  }
}

void Identifier::onKeyframe(TKeyframeIndex keyframeIndex, const torch::Tensor &patch_f32, TrackData &track) {
  at::InferenceMode inferenceGuard;
  std::optional<nvtx3::scoped_range> nvtxLabel{"id_before (" + cameraName_ + ")"};

  assert(patch_f32.device().is_cuda());

  // Send to model
  at::Tensor identityLogitsGpu;
  nvtxLabel.emplace("id_net (" + cameraName_ + ")");
  at::cuda::CUDAEvent eventBeforeNetwork{cudaEventDefault}, eventAfterNetwork{cudaEventDefault};
  eventBeforeNetwork.record();
  if (isStatefulModel_) {
    // callStatefulModel(identityLogitsGpu, patches, trackIds);
    throw std::runtime_error("Stateful doesn't make sense any more");
  } else {
    callStatelessModel(identityLogitsGpu, patch_f32.unsqueeze(0));
    identityLogitsGpu = identityLogitsGpu.squeeze(0); // Remove dummy batch dimension
  }
  eventAfterNetwork.record();
  nvtxLabel.emplace("id_after (" + cameraName_ + ")");

  at::Tensor identityLogits = identityLogitsGpu.to(at::kCPU); // Dims: [identity]
  at::Tensor identityProbs =
      torch::nn::functional::softmax(identityLogits, torch::nn::functional::SoftmaxFuncOptions(/*dim*/ 0));
  const size_t identityCount = identityLogits.size(0);

  // TODO: sort tracks based on score
  // const TIdentity identity = selectOptimalIdentities(identityProbs);

  track.identityHistogram.resize(identityCount); // TODO: initialize histogram with size

  // Did we have a keyframe at this index before?
  if (keyframeIndex < track.identityProbsByKeyframe.size()) {
    // Remove vote from old keyframe
    at::Tensor &oldIdentityProbs = track.identityProbsByKeyframe[keyframeIndex];
    for (const auto id : std::views::iota(0uz, identityCount)) {
      track.identityHistogram.removeVote(id, oldIdentityProbs[id].item<float32_t>());
    }
    oldIdentityProbs = identityProbs;
  } else {
    // No, resize so we can remember from now on
    assert(keyframeIndex == track.identityProbsByKeyframe.size());
    track.identityProbsByKeyframe.push_back(identityProbs);
  }
  // Add to histogram
  for (const auto id : std::views::iota(0uz, identityCount)) {
    track.identityHistogram.addVote(id, identityProbs[id].item<float32_t>());
  }

  // Store selected identity in track data
  track.selectedIdentity = INVALID_IDENTITY;
  {
    auto [bestIdentity, bestVoteCount, firstToSecondRatio] = track.identityHistogram.getHighest();
    constexpr uint64_t VOTE_COUNT_THRESHOLD = 3;
    constexpr float32_t RATIO_THRESHOLD = 0.65f;
    if (bestVoteCount >= VOTE_COUNT_THRESHOLD && firstToSecondRatio > RATIO_THRESHOLD) {
      track.selectedIdentity = bestIdentity + 1;
    }
  }

  // constexpr auto MS_TO_NS = 1e6f;
  // cudaStreamSynchronize(cudaStream_);
  // addRosKeyValue(msg.timings.items_ns, "id_net", eventBeforeNetwork.elapsed_time(eventAfterNetwork) * MS_TO_NS);
}

} // namespace zoo
