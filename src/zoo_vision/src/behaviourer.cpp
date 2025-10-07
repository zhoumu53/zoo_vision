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

#include "zoo_vision/behaviourer.hpp"

#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/compute_device.hpp"
#include "zoo_vision/utils.hpp"

#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>
#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

Behaviourer::Behaviourer(int nameIndex, std::string cameraName, std::optional<at::cuda::CUDAStream> cudaStream)
    : name_{std::format("behaviourer_{}", nameIndex)}, logger_{rclcpp::get_logger(name_)}, cudaStream_{cudaStream},
      cameraName_{cameraName} {
  at::InferenceMode inferenceGuard;

  RCLCPP_INFO(get_logger(), "Starting identifier for %s", cameraName_.c_str());

  readConfig(getConfig());
}

void Behaviourer::readConfig(const nlohmann::json &config) {
  // Settings

  // Load model
  const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / config["models"]["behaviour"]);
  loadModel(modelPath);
}

void Behaviourer::loadModel(const std::filesystem::path &modelPath) {
  RCLCPP_INFO(get_logger(), "Loading behaviour model from %s", modelPath.c_str());

  try {
    if (!std::filesystem::exists(modelPath)) {
      throw ZooVisionError("Model does not exist");
    }
    model_ = torch::jit::load(modelPath, torch::kCUDA);
    model_.eval();
  } catch (const std::exception &ex) {
    std::cout << "Error loading model from " << modelPath << std::endl;
    std::cout << "Exception: " << ex.what() << std::endl;
    std::terminate();
  }
  // DEBUG print model info
}

void Behaviourer::callStatelessModel(at::Tensor &logitsGpu, const torch::Tensor &patches) {
  c10::IValue modelResult = model_.forward({patches});
  if (modelResult.isTensor()) {
    logitsGpu = modelResult.toTensor();
  } else {
    auto dict = modelResult.toGenericDict();
    logitsGpu = dict.at("logits").toTensor();
    logitsGpu = logitsGpu.squeeze(1); // Remove dummy time dimension
  }
}

void Behaviourer::onDetection(zoo_msgs::msg::Detection &msg, const torch::Tensor &patches) {
  at::InferenceMode inferenceGuard;
  std::optional<nvtx3::scoped_range> nvtxLabel;

  // Send to model
  at::Tensor logitsGpu;
  nvtxLabel.emplace("id_net (" + cameraName_ + ")");
  at::cuda::CUDAEvent eventBeforeNetwork{cudaEventDefault}, eventAfterNetwork{cudaEventDefault};
  eventBeforeNetwork.record();
  callStatelessModel(logitsGpu, patches);
  eventAfterNetwork.record();
  nvtxLabel.emplace("id_after (" + cameraName_ + ")");

  at::Tensor logits = logitsGpu.to(g_computeDevice); // Dims: [track, identity]
  at::Tensor probabilities = torch::nn::functional::softmax(logits, torch::nn::functional::SoftmaxFuncOptions(1));

  const size_t patchCount = logits.size(0);
  const size_t classCount = logits.size(1);

  // TODO: sort tracks based on score
  std::vector<int> bestClasses;
  bestClasses.resize(patchCount);
  constexpr float32_t CONFIDENCE_THRESHOLD = 0.2f;

  // Force to max probability
  const auto &[maxProbs, maxIndices] = probabilities.topk(/*k*/ 2, /*dim*/ 1);
  for (int i = 0; i < static_cast<int>(patchCount); ++i) {
    const float32_t top1 = maxProbs.index({i, 0}).item<float32_t>();
    const float32_t top2 = maxProbs.index({i, 1}).item<float32_t>();
    if ((top1 - top2) >= CONFIDENCE_THRESHOLD) {
      const int maxIdx = maxIndices.index({i, 0}).item<int>();
      bestClasses[i] = maxIdx;
    } else {
      bestClasses[i] = 0;
    }
  }

  assert(bestClasses.size() == patchCount);

  for (const auto i : std::views::iota(0u, patchCount)) {
    msg.behaviour_ids[i] = bestClasses[i];
    for (const auto j : std::views::iota(0u, classCount)) {
      msg.behaviour_logits[i * classCount + j] = probabilities[i][j].item<float>();
    }
  }

  constexpr auto MS_TO_NS = 1e6f;
  if (cudaStream_.has_value()) {
    cudaStreamSynchronize(*cudaStream_);
  }
  addRosKeyValue(msg.timings.items_ns, "beh_net", eventBeforeNetwork.elapsed_time(eventAfterNetwork) * MS_TO_NS);
}

} // namespace zoo
