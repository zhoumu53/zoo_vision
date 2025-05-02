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

#include "zoo_vision/segmenter_yolo.hpp"

#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/utils.hpp"

#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <nlohmann/json.hpp>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>
#include <seg/YOLO11Seg.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

SegmenterYolo::SegmenterYolo(int nameIndex, std::string cameraName, at::cuda::CUDAStream cudaStream)
    : name_{std::format("segmenter_yolo_{}", nameIndex)}, logger_{rclcpp::get_logger(name_)}, cudaStream_{cudaStream},
      cameraName_{cameraName} {
  at::InferenceMode inferenceGuard;
  RCLCPP_INFO(get_logger(), "Starting segmenter_yolo for %s", cameraName_.c_str());

  readConfig(getConfig());
}

SegmenterYolo::~SegmenterYolo() = default;

void SegmenterYolo::readConfig(const nlohmann::json &config) {
  // Load model
  const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / config["models"]["segmentation"]);
  loadModel(modelPath);
  scoreThreshold_ = config["models"]["score_threshold"].get<float>();
}

void SegmenterYolo::loadModel(const std::filesystem::path &modelPath) {
  RCLCPP_INFO(get_logger(), "Loading segmentation model from %s", modelPath.c_str());

  const auto namesPath = std::filesystem::canonical(getDataPath() / "../models/segmentation/yolo/class_names.txt");

  try {
    if (!std::filesystem::exists(modelPath)) {
      throw std::runtime_error("Model does not exist");
    }
    model_ = std::make_unique<YOLOv11SegDetector>(modelPath.string(), namesPath.string(), /*useGPU*/ true);
  } catch (const std::exception &ex) {
    std::cout << "Error loading model from " << modelPath << std::endl;
    std::cout << "Exception: " << ex.what() << std::endl;
    std::terminate();
  }
  // DEBUG print model info
}

auto SegmenterYolo::callYolo(const cv::Mat &image) -> SegmentationResult {
  std::vector<Segmentation> result = model_->segment(image);
  RCLCPP_INFO(get_logger(), "Detected %i objects", int(result.size()));
  return SegmentationResult{};
}

void SegmenterYolo::onImage(zoo_msgs::msg::Detection &detectionMsg, std::vector<Eigen::AlignedBox2f> & /*boxes*/,
                            const cv::Mat &image) {
  // RCLCPP_INFO(get_logger(), "Segmenter received id: %s", detectionMsg.header.frame_id.data.data());

  std::optional<nvtx3::scoped_range> nvtxLabel{"seg_before (" + cameraName_ + ")"};

  at::cuda::CUDAEvent eventBeforeNetwork{cudaEventDefault}, eventAfterNetwork{cudaEventDefault};

  ////////////////////////////////////////////////////////////
  // Execute segmentation network
  SegmentationResult segmentationResult;
  {
    torch::jit::GraphOptimizerEnabledGuard optGuard(false);

    nvtxLabel.emplace("seg_network (" + cameraName_ + ")");

    eventBeforeNetwork.record();
    segmentationResult = callYolo(image);
    eventAfterNetwork.record();
  }

  ////////////////////////////////////////////////////////////
  // Post-process segmentation results

  nvtxLabel.emplace("seg_after (" + cameraName_ + ")");

  cudaStreamSynchronize(cudaStream_);

  constexpr auto MS_TO_NS = 1e6f;
  addRosKeyValue(detectionMsg.timings.items_ns, "seg_net",
                 eventBeforeNetwork.elapsed_time(eventAfterNetwork) * MS_TO_NS);
}

} // namespace zoo
