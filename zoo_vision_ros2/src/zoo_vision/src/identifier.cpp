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
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

Identifier::Identifier(TrackMatcher &trackMatcher, const rclcpp::NodeOptions &options, int nameIndex)
    : Node(std::format("identifier_{}", nameIndex), options), trackMatcher_{trackMatcher} {
  at::InferenceMode inferenceGuard;

  cameraName_ = declare_parameter<std::string>("camera_name");
  RCLCPP_INFO(get_logger(), "Starting segmenter for %s", cameraName_.c_str());

  readConfig(getConfig());

  // Subscribe to receive images from camera
  // Publish results
}

void Identifier::readConfig(const nlohmann::json &config) {
  // Settings
  recordTracks_ = config["record_tracks"].get<bool>();

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

void saveTensorImage(const at::Tensor &imgTensor, const std::string &name) {
  const auto region0 = (imgTensor.permute({1, 2, 0}).to(at::kCPU) * 255).toType(at::kByte).contiguous();
  assert(region0.stride(1) == 3);
  assert(region0.stride(2) == 1);
  auto img = cv::Mat(region0.size(0), region0.size(1), CV_8UC3, region0.data_ptr(), region0.stride(0));
  cv::Mat imgRgb;
  cv::cvtColor(img, imgRgb, cv::COLOR_RGB2BGR);
  cv::imwrite(name.c_str(), imgRgb);
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
  bestIdentities.resize(trackCount, identityCount + 1);

  // Force to max probability
  at::Tensor minProbs = probabilities.argmax(1);
  for (size_t i = 0; i < trackCount; ++i) {
    bestIdentities[i] = minProbs[i].item<int>();
  }

  return bestIdentities;
}

void Identifier::extractCrops(torch::Tensor &patches, const torch::Tensor &imageGpu,
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

void Identifier::callStatefulModel(at::Tensor &identityLogitsGpu, const torch::Tensor &patches,
                                   const std::span<const TrackId> trackIds) {
  const int detectionCount = trackIds.size();
  constexpr size_t HIDDEN_LAYER_COUNT = 3;
  constexpr size_t HIDDEN_STATE_SIZE = 128;
  auto trackState0 = at::zeros({HIDDEN_LAYER_COUNT, detectionCount, HIDDEN_STATE_SIZE},
                               at::TensorOptions(at::kCUDA).dtype(at::kFloat));
  for (const auto &[i, trackId] : std::views::enumerate(trackIds)) {
    // Copy track state
    TrackData *track = trackMatcher_.getTrackData(trackId);
    assert(track != nullptr);
    if (track->identityState.has_value()) {
      trackState0.index({Slice(), i, Ellipsis}).copy_(track->identityState.value());
    }
  }

  auto patchesTime = patches.unsqueeze(1);
  at::Tensor trackState;
  {
    c10::IValue modelResult = identityNetwork_.forward({patchesTime, trackState0});
    c10::Dict<c10::IValue, c10::IValue> modelResultDict = modelResult.toGenericDict();

    identityLogitsGpu = modelResultDict.at("logits").toTensor();
    identityLogitsGpu = identityLogitsGpu.squeeze(1); // Remove dummy time dimension

    trackState = modelResultDict.at("gru_state").toTensor();
  }

  // Remember track state
  for (const auto &[i, trackId] : std::views::enumerate(trackIds)) {
    TrackData *track = trackMatcher_.getTrackData(trackId);
    assert(track != nullptr);
    track->identityState.emplace(trackState.index({Slice(), i, Ellipsis}));
  }
}

void Identifier::callStatelessModel(at::Tensor &identityLogitsGpu, const torch::Tensor &patches) {
  c10::IValue modelResult = identityNetwork_.forward({patches});
  identityLogitsGpu = modelResult.toTensor();
}

void Identifier::onDetection(const at::cuda::CUDAStream &cudaStream_, const torch::Tensor &imageGpu,
                             const float scale_image_from_detection, const std::span<const TrackId> trackIds,
                             const std::span<const zoo_msgs::msg::BoundingBox2D> bboxes,
                             zoo_msgs::msg::Detection &msg) {
  at::cuda::CUDAStreamGuard streamGuard{cudaStream_};
  at::InferenceMode inferenceGuard;
  std::optional<nvtx3::scoped_range> nvtxLabel{"id_before (" + cameraName_ + ")"};

  assert(imageGpu.device().is_cuda());

  // Prepare
  at::Tensor patches;
  extractCrops(patches, imageGpu, scale_image_from_detection, bboxes);

  // Save images for this track
  if (recordTracks_) {
    for (auto &&[idx, trackId] : std::views::enumerate(trackIds)) {
      TrackData *track = trackMatcher_.getTrackData(trackId);
      saveTensorImage(patches[idx], std::format("debug/{}_t{}_{}.png", cameraName_, trackId, track->trackLength));
    }
  }

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
  const std::vector<int> identities = selectOptimalIdentities(identityProbs, prior);
  assert(identities.size() == bboxes.size());

  for (const auto i : std::views::iota(0u, bboxes.size())) {
    msg.identity_ids[i] = identities[i] + 1; // Add one because 0 is background
    for (const auto j : std::views::iota(0u, identityCount)) {
      msg.identity_logits[i * identityCount + j] = identityLogits[i][j].item<float>();
      // msg.identity_logits[i * identityCount + j] = identityProbs[i][j].item<float>();
    }
  }

  constexpr auto MS_TO_NS = 1e6f;
  cudaStreamSynchronize(cudaStream_);
  addRosKeyValue(msg.timings.items_ns, "id_net", eventBeforeNetwork.elapsed_time(eventAfterNetwork) * MS_TO_NS);
}

} // namespace zoo

// #include "rclcpp_components/register_node_macro.hpp"

// RCLCPP_COMPONENTS_REGISTER_NODE(zoo::Identifier)