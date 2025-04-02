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
#include <sensor_msgs/image_encodings.hpp>
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

std::vector<int> selectOptimalIdentities(const at::Tensor &probabilities, const Eigen::MatrixXf &prior) {
  const size_t trackCount = probabilities.size(0);
  const size_t identityCount = probabilities.size(1);

  // Iterate over all permutations
  const size_t optimTrackCount = std::min(trackCount, identityCount);
  std::vector<uint8_t> bitmask(optimTrackCount, 1);
  bitmask.resize(identityCount, 0);

  float bestProb = -std::numeric_limits<float>::infinity();
  std::vector<int> bestIdentities(optimTrackCount);

  std::vector<int> permIdentities(optimTrackCount);
  // print integers and permute bitmask
  do {
    int j = 0;
    for (size_t i = 0; i < bitmask.size(); ++i) // [0..N-1] integers
    {
      if (bitmask[i]) {
        permIdentities[j] = static_cast<int>(i);
        j += 1;
      }
    }

    // Iterate over all permutations of the selected identities
    do {
      const float weight = calculateWeight(permIdentities, prior);
      if (weight == 0.0f) {
        continue;
      }
      float prob = 0;
      for (size_t k = 0; k < optimTrackCount; ++k) {
        prob += probabilities[k][permIdentities[k]].item<float>();
      }

      // Remember best
      if (prob > bestProb) {
        bestProb = prob;
        bestIdentities = permIdentities;
      }
    } while (std::next_permutation(permIdentities.begin(), permIdentities.end()));

  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

  // TODO: fix this
  bestIdentities.resize(trackCount, 0);

  constexpr float32_t CONFIDENCE_THRESHOLD = 0.4f;

  // Force to max probability
  const auto &[maxProbs, maxIndices] = probabilities.topk(/*k*/ 2, /*dim*/ 1);
  for (int i = 0; i < static_cast<int>(trackCount); ++i) {
    const float32_t top1 = maxProbs.index({i, 0}).item<float32_t>();
    const float32_t top2 = maxProbs.index({i, 1}).item<float32_t>();
    if ((top1 - top2) >= CONFIDENCE_THRESHOLD) {
      const int maxIdx = maxIndices.index({i, 0}).item<int>();
      bestIdentities[i] = maxIdx + 1; // Add one because class 0 is invalid
    } else {
      bestIdentities[i] = 0;
    }
  }

  return bestIdentities;
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

void Identifier::onDetection(zoo_msgs::msg::Detection &msg, const torch::Tensor &patches,
                             const std::span<const TrackId> trackIds) {
  at::InferenceMode inferenceGuard;
  std::optional<nvtx3::scoped_range> nvtxLabel{"id_before (" + cameraName_ + ")"};

  assert(patches.device().is_cuda());

  // Send to model
  at::Tensor identityLogitsGpu;
  nvtxLabel.emplace("id_net (" + cameraName_ + ")");
  at::cuda::CUDAEvent eventBeforeNetwork{cudaEventDefault}, eventAfterNetwork{cudaEventDefault};
  eventBeforeNetwork.record();
  if (isStatefulModel_) {
    callStatefulModel(identityLogitsGpu, patches, trackIds);
  } else {
    callStatelessModel(identityLogitsGpu, patches);
  }
  eventAfterNetwork.record();
  nvtxLabel.emplace("id_after (" + cameraName_ + ")");

  at::Tensor identityLogits = identityLogitsGpu.to(at::kCPU); // Dims: [track, identity]
  at::Tensor identityProbs =
      torch::nn::functional::softmax(identityLogits, torch::nn::functional::SoftmaxFuncOptions(1));

  const size_t identityCount = identityLogits.size(1);
  Eigen::MatrixXf prior;
  prior.resize(identityCount, identityCount);
  prior.setConstant(1.0f);
  for (size_t i = 0; i < identityCount; ++i) {
    prior(i, i) = 0; // Same identity cannot happen twice
  }
  constexpr int kChandra = 0;
  constexpr int kIndi = 1;
  constexpr int kFahra = 2;
  constexpr int kPanang = 3;

  prior(kChandra, kFahra) = 0;
  prior(kChandra, kPanang) = 0;
  prior(kIndi, kFahra) = 0;
  prior(kIndi, kPanang) = 0;
  prior(kFahra, kChandra) = 0;
  prior(kFahra, kIndi) = 0;
  prior(kPanang, kChandra) = 0;
  prior(kPanang, kIndi) = 0;

  // TODO: sort tracks based on score
  std::vector<int> identities = selectOptimalIdentities(identityProbs, prior);
  assert(identities.size() == trackIds.size());

  // Add to histogram
  constexpr int INVALID_ID = 0;
  constexpr uint64_t VOTE_COUNT_THRESHOLD = 10;
  for (auto [trackId, identity] : std::views::zip(trackIds, identities)) {
    auto &trackData = trackMatcher_.getTrackData(trackId);
    trackData.identityHistogram.resize(identityCount); // TODO: initialize histogram with size

    if (identity == INVALID_ID) {
      continue;
    }

    if (identity >= identityCount + 1) {
      throw std::runtime_error(std::format("identity out of range, {} >= {}", identity, identityCount));
    }

    trackData.identityHistogram.addVote(identity - 1);

    auto [bestIdentity, bestVoteCount] = trackData.identityHistogram.getHighest();
    if (bestVoteCount < VOTE_COUNT_THRESHOLD) {
      identity = INVALID_ID;
    } else {
      identity = bestIdentity + 1;
    }
  }

  // Forward logits for display
  for (const auto i : std::views::iota(0u, trackIds.size())) {
    auto &track = trackMatcher_.getTrackData(trackIds[i]);

    msg.identity_ids[i] = identities[i];
    for (const auto j : std::views::iota(0u, identityCount)) {
      // msg.identity_logits[i * identityCount + j] = identityLogits[i][j].item<float>();
      // msg.identity_logits[i * identityCount + j] = identityProbs[i][j].item<float>();
      msg.identity_logits[i * identityCount + j] = track.identityHistogram.getVotes()[j];
    }
  }

  constexpr auto MS_TO_NS = 1e6f;
  cudaStreamSynchronize(cudaStream_);
  addRosKeyValue(msg.timings.items_ns, "id_net", eventBeforeNetwork.elapsed_time(eventAfterNetwork) * MS_TO_NS);
}

} // namespace zoo
