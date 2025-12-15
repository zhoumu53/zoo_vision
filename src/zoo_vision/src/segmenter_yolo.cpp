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

#include "zoo_vision/compute_device.hpp"
#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/utils.hpp"

#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <nlohmann/json.hpp>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/time.hpp>
#include <seg/YOLO11Seg.hpp>

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

SegmenterYolo::SegmenterYolo(int nameIndex, std::string cameraName, std::optional<at::cuda::CUDAStream> cudaStream)
    : name_{std::format("segmenter_yolo_{}", nameIndex)}, logger_{rclcpp::get_logger(name_)}, cudaStream_{cudaStream},
      cameraName_{cameraName} {
  at::InferenceMode inferenceGuard;
  RCLCPP_INFO(get_logger(), "Starting segmenter_yolo for %s", cameraName_.c_str());

  readConfig(getConfig());
}

SegmenterYolo::~SegmenterYolo() = default;

void SegmenterYolo::readConfig(const nlohmann::json &config) {
  // Load model
  const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / config["detection"]["model"]);
  loadModel(modelPath);
  scoreThreshold_ = config["detection"]["score_threshold"].get<float>();
}

void SegmenterYolo::loadModel(const std::filesystem::path &modelPath) {
  RCLCPP_INFO(get_logger(), "Loading segmentation model from %s", modelPath.c_str());

  const auto namesPath = std::filesystem::canonical(getDataPath() / "../models/segmentation/yolo/class_names.txt");

  try {
    if (!std::filesystem::exists(modelPath)) {
      throw ZooVisionError("Model does not exist");
    }
    model_ = std::make_unique<YOLOv11SegDetector>(modelPath.string(), namesPath.string(),
                                                  /*useGPU*/ g_computeDevice == c10::kCUDA);
  } catch (const std::exception &ex) {
    std::cout << "Error loading model from " << modelPath << std::endl;
    std::cout << "Exception: " << ex.what() << std::endl;
    std::terminate();
  }
}

AlignedBox2f eigenBboxFromYoloBbox(const BoundingBox &bbox) {
  Vector2f min = {bbox.x, bbox.y};
  Vector2f size = {bbox.width, bbox.height};
  return AlignedBox2f(min, min + size);
}
void SegmenterYolo::onImage(SegmenterResult &result, const at::Tensor & /*imageGpu*/, const cv::Mat &imageCpu) {
  // RCLCPP_INFO(get_logger(), "Segmenter received id: %s", detectionMsg.header.frame_id.data.data());

  std::optional<nvtx3::scoped_range> nvtxLabel{"seg_before (" + cameraName_ + ")"};

  at::cuda::CUDAEvent eventBeforeNetwork{cudaEventDefault}, eventAfterNetwork{cudaEventDefault};

  ////////////////////////////////////////////////////////////
  // Execute segmentation network
  std::vector<Segmentation> resultsYolo;
  {
    nvtxLabel.emplace("seg_network (" + cameraName_ + ")");

    if (g_computeDevice == at::kCUDA) {
      eventBeforeNetwork.record();
    }
    resultsYolo = model_->segment(imageCpu);
    if (g_computeDevice == at::kCUDA) {
      eventAfterNetwork.record();
    }
  }

  ////////////////////////////////////////////////////////////
  // Post-process segmentation results

  nvtxLabel.emplace("seg_after (" + cameraName_ + ")");

  for (const auto &[i, resultYolo] : std::views::enumerate(resultsYolo)) {
    CHECK_EQ(detectionImageSize_[0], resultYolo.mask.cols);
    CHECK_EQ(detectionImageSize_[1], resultYolo.mask.rows);

    result.bboxesInDetection.push_back(eigenBboxFromYoloBbox(resultYolo.box));
    result.scores.push_back(resultYolo.conf);
    cv::Mat1b maskMap = wrapCvFromTensor1b(result.masks[i]);
    resultYolo.mask.copyTo(maskMap);
  }
}

} // namespace zoo
