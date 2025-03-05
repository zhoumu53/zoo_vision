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

Identifier::Identifier(const rclcpp::NodeOptions &options, int nameIndex)
    : Node(std::format("identifier_{}", nameIndex), options) {
  at::InferenceMode inferenceGuard;

  cameraName_ = declare_parameter<std::string>("camera_name");
  RCLCPP_INFO(get_logger(), "Starting segmenter for %s", cameraName_.c_str());

  readConfig(getConfig());

  // Subscribe to receive images from camera
  // Publish results
}

void Identifier::readConfig(const nlohmann::json &config) {
  // Load model
  const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / config["models"]["identity"]);
  loadModel(modelPath);
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
  std::cout << "Size: " << imgTensor.sizes() << std::endl;
  const auto region0 = (imgTensor.permute({1, 2, 0}).to(at::kCPU) * 255).toType(at::kByte).contiguous();
  assert(region0.stride(1) == 3);
  assert(region0.stride(2) == 1);
  auto img = cv::Mat(region0.size(0), region0.size(1), CV_8UC3, region0.data_ptr(), region0.stride(0));
  cv::Mat imgRgb;
  cv::cvtColor(img, imgRgb, cv::COLOR_RGB2BGR);
  cv::imwrite(name.c_str(), imgRgb);
}

void Identifier::onDetection(const at::cuda::CUDAStream &cudaStream_, const torch::Tensor &imageGpu,
                             const std::span<const zoo_msgs::msg::BoundingBox2D> bboxes,
                             std::span<uint32_t> outputIdentities) {
  at::cuda::CUDAStreamGuard streamGuard{cudaStream_};
  at::InferenceMode inferenceGuard;
  std::optional<nvtx3::scoped_range> nvtxLabel{"id_before (" + cameraName_ + ")"};

  assert(imageGpu.device().is_cuda());

  constexpr int CROP_SIZE = 256;
  const int detectionCount = bboxes.size();
  const int channels = 3;

  auto inputRegions =
      at::zeros({detectionCount, channels, CROP_SIZE, CROP_SIZE}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));
  // Extract crops
  for (const auto &[i, bbox] : std::views::enumerate(bboxes)) {
    const float32_t bboxAspect = static_cast<float32_t>(bbox.half_size[0]) / static_cast<float32_t>(bbox.half_size[1]);
    const auto bboxPatch = imageGpu.index({
        None,
        Slice(),
        Slice(bbox.center[1] - bbox.half_size[1], bbox.center[1] + bbox.half_size[1]),
        Slice(bbox.center[0] - bbox.half_size[0], bbox.center[0] + bbox.half_size[0]),
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

    auto destRegion = inputRegions[i].index({Slice(), Slice(c0.y(), c1.y()), Slice(c0.x(), c1.x())});
    destRegion.copy_(rescaledPatch[0]);
  }

  // Send to model
  at::Tensor identityResultGpu;
  {
    nvtxLabel.emplace("id_net (" + cameraName_ + ")");
    identityResultGpu = identityNetwork_.forward({inputRegions}).toTensor();
  }
  nvtxLabel.emplace("id_after (" + cameraName_ + ")");

  // std::cout << identityResultGpu.to(torch::kCPU);
  // for (auto i : std::views::iota(0, detectionCount)) {
  //   saveTensorImage(inputRegions[i], std::format("debug/region{}.png", i));
  // }
  // throw std::runtime_error("err");

  at::Tensor identityTensor = identityResultGpu.to(at::kCPU).argmax(1);
  for (const auto i : std::views::iota(0u, bboxes.size())) {
    outputIdentities[i] = identityTensor[i].item<int>() + 1; // Add one because 0 is background
  }
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::Identifier)