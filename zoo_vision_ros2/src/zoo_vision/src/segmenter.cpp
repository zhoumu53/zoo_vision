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

#include "zoo_vision/segmenter.hpp"

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
#include <sensor_msgs/image_encodings.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

Segmenter::Segmenter(int nameIndex, std::string cameraName, TrackMatcher &trackMatcher, at::cuda::CUDAStream cudaStream)
    : name_{std::format("segmenter_{}", nameIndex)}, logger_{rclcpp::get_logger(name_)}, cudaStream_{cudaStream},
      cameraName_{cameraName}, trackMatcher_{trackMatcher} {
  at::InferenceMode inferenceGuard;
  RCLCPP_INFO(get_logger(), "Starting segmenter for %s", cameraName_.c_str());

  readConfig(getConfig());
}

void Segmenter::readConfig(const nlohmann::json &config) {
  // Camera calibration
  H_mapFromWorld2_ = config["map"]["T_map_from_world2"];
  H_world2FromCamera_ = config["cameras"][cameraName_]["H_world2_from_camera"];

  // Load model
  const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / config["models"]["segmentation"]);
  loadModel(modelPath);
  elephant_label_id_ = config["models"]["elephant_label_id"].get<int>();
  scoreThreshold_ = config["models"]["score_threshold"].get<float>();
}

void Segmenter::loadModel(const std::filesystem::path &modelPath) {
  RCLCPP_INFO(get_logger(), "Loading segmentation model from %s", modelPath.c_str());

  try {
    if (!std::filesystem::exists(modelPath)) {
      throw std::runtime_error("Model does not exist");
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

void copyBboxToRos(zoo_msgs::msg::BoundingBox2D &outBbox, const Eigen::AlignedBox2f &in) {
  const Eigen::Vector2f center = in.center();
  const Eigen::Vector2f halfSize = in.sizes() / 2;
  outBbox.center[0] = center[0];
  outBbox.center[1] = center[1];
  outBbox.half_size[0] = halfSize[0];
  outBbox.half_size[1] = halfSize[1];
}

auto Segmenter::callMaskrcnn(const at::Tensor &image) -> SegmentationResult {
  // TorchScript models require a List[IValue] as input
  at::List<at::Tensor> imageList({image});
  at::IValue detectionResult = model_.forward({imageList});

  const auto detections = detectionResult.toTuple()->elements()[1].toList()[0].get().toGenericDict();

  // Results in gpu
  const at::Tensor masks_f32 = detections.at("masks").toTensor().squeeze(1);
  const at::Tensor boxes = detections.at("boxes").toTensor();
  const at::Tensor scores = detections.at("scores").toTensor();
  const at::Tensor labels = detections.at("labels").toTensor();

  // const torch::Tensor scores = detections.at("scores").toTensor().to(torch::kCPU);

  // Masks to u8
  const at::Tensor masks_u8 = masks_f32.mul(255).clamp(0, 255).to(at::kByte);

  return SegmentationResult{masks_u8, boxes, scores, labels};
}

auto Segmenter::callMask2Former(const at::Tensor &image) -> SegmentationResult {
  // TorchScript models require a List[IValue] as input
  const at::IValue detectionResult = model_.forward({image.unsqueeze(0)});

  const auto detections = detectionResult.toGenericDict();

  // Results in gpu
  const at::Tensor masks_u8 = detections.at("masks").toTensor().squeeze(1);
  const at::Tensor scores = detections.at("scores").toTensor();
  const at::Tensor boxes = detections.at("boxes").toTensor();
  const at::Tensor labels = detections.at("labels").toTensor();

  return SegmentationResult{masks_u8, boxes, scores, labels};
}
void Segmenter::onImage(zoo_msgs::msg::Detection &detectionMsg, const at::Tensor &imageTensor) {
  // RCLCPP_INFO(get_logger(), "Segmenter received id: %s", detectionMsg.header.frame_id.data.data());

  // at::InferenceMode inferenceGuard; // Runtime error: Global alloc not supported yet
  at::NoGradGuard nograd;
  std::optional<nvtx3::scoped_range> nvtxLabel{"seg_before (" + cameraName_ + ")"};

  at::cuda::CUDAEvent eventBeforeNetwork{cudaEventDefault}, eventAfterNetwork{cudaEventDefault};

  // Resize
  const int DETECTION_HEIGHT = 600;
  const int DETECTION_WIDTH = 1060;
  at::Tensor detectionImage;
  {
    namespace F = torch::nn::functional;
    const auto interpolateOpts = F::InterpolateFuncOptions()
                                     .size({{DETECTION_HEIGHT, DETECTION_WIDTH}})
                                     .mode(torch::kBilinear)
                                     .antialias(true)
                                     .align_corners(false);
    detectionImage = F::interpolate(imageTensor.unsqueeze(0), interpolateOpts)[0];
  }

  ////////////////////////////////////////////////////////////
  // Execute segmentation network
  SegmentationResult segmentationResult;
  {
    torch::jit::GraphOptimizerEnabledGuard optGuard(false);

    nvtxLabel.emplace("seg_network (" + cameraName_ + ")");

    eventBeforeNetwork.record();
    segmentationResult = callMask2Former(detectionImage);
    eventAfterNetwork.record();
  }

  ////////////////////////////////////////////////////////////
  // Post-process segmentation results

  nvtxLabel.emplace("seg_after (" + cameraName_ + ")");

  // Check dimensions
  assert(segmentationResult.scores.dim() == 1);
  assert(segmentationResult.labels.dim() == 1);
  assert(segmentationResult.boxes.dim() == 2);
  assert(segmentationResult.masks_u8.dim() == 3);

  const int64_t modelDetectionCount = segmentationResult.scores.sizes()[0];
  assert(segmentationResult.scores.size(0) == modelDetectionCount);
  assert(segmentationResult.labels.size(0) == modelDetectionCount);
  assert(segmentationResult.boxes.size(0) == modelDetectionCount);
  assert(segmentationResult.masks_u8.size(0) == modelDetectionCount);

  const int64_t maskHeight = segmentationResult.masks_u8.sizes()[1];
  const int64_t maskWidth = segmentationResult.masks_u8.sizes()[2];
  detectionMsg.scalex_image_from_detection = static_cast<float>(imageTensor.size(2)) / maskWidth;
  detectionMsg.scaley_image_from_detection = static_cast<float>(imageTensor.size(1)) / maskHeight;

  const int64_t MAX_DETECTION_COUNT = zoo_msgs::msg::Detection::MAX_DETECTION_COUNT;
  detectionMsg.masks.sizes[0] = MAX_DETECTION_COUNT;
  detectionMsg.masks.sizes[1] = maskHeight;
  detectionMsg.masks.sizes[2] = maskWidth;
  at::Tensor masksMap = mapRosTensor(detectionMsg.masks);

  // Move all to cpu
  const at::Tensor scores = segmentationResult.scores.to(at::kCPU, true);
  const at::Tensor labels = segmentationResult.labels.to(at::kCPU, true);
  const at::Tensor boxesNet = segmentationResult.boxes.to(at::kCPU, true);
  const at::Tensor masks = segmentationResult.masks_u8.to(at::kCPU, true);
  cudaStreamSynchronize(cudaStream_);

  // Sort scores
  const auto [sortedScores, sortedIndices] = torch::sort(scores, /*dim*/ 0, /*descending*/ true);

  constexpr auto MS_TO_NS = 1e6f;
  addRosKeyValue(detectionMsg.timings.items_ns, "seg_net",
                 eventBeforeNetwork.elapsed_time(eventAfterNetwork) * MS_TO_NS);

  Eigen::Map<Eigen::Matrix3Xf> worldPositionsMap{detectionMsg.world_positions.data(), 3, MAX_DETECTION_COUNT};

  detectionMsg.detection_count = 0;

  const auto world_from_world2 = [](const Eigen::Vector2f &x2) { return Eigen::Vector3f{x2[0], x2[1], 0.0f}; };
  std::vector<Eigen::AlignedBox2f> boxes;
  for (int i = 0; i < modelDetectionCount && detectionMsg.detection_count < MAX_DETECTION_COUNT; ++i) {
    const auto inputIndex = sortedIndices[i].item<int>();

    const float score = sortedScores[i].item<float>();
    if (score < scoreThreshold_) {
      continue;
    }

    const int label = labels[inputIndex].item<int>();
    if (label != elephant_label_id_) {
      continue;
    }

    const auto outputIndex = detectionMsg.detection_count;
    detectionMsg.detection_count += 1;

    // Track ids are decided later

    // Mask
    const at::Tensor mask = masks[inputIndex];
    masksMap[outputIndex].copy_(mask);

    // Bbox
    const at::Tensor bbox = boxesNet[inputIndex];
    const Eigen::Vector2f x0{bbox[0].item<float32_t>(), bbox[1].item<float32_t>()};
    const Eigen::Vector2f x1{bbox[2].item<float32_t>(), bbox[3].item<float32_t>()};
    const auto bboxEigen = Eigen::AlignedBox2f(x0, x1);
    boxes.push_back(bboxEigen);
    copyBboxToRos(detectionMsg.bboxes[outputIndex], bboxEigen);

    // Project to world
    const Eigen::Vector2f imagePosition =
        Eigen::Vector2f{bboxEigen.center()[0] * detectionMsg.scalex_image_from_detection,
                        bboxEigen.max()[1] * detectionMsg.scaley_image_from_detection};
    const Eigen::Vector3f worldPosition =
        world_from_world2((H_world2FromCamera_ * imagePosition.homogeneous()).hnormalized());
    worldPositionsMap.col(outputIndex) = worldPosition;
  }
  detectionMsg.masks.sizes[0] = detectionMsg.detection_count;

  auto msgBboxes = std::span{detectionMsg.bboxes.data(), detectionMsg.detection_count};
  auto msgTrackIds = std::span{detectionMsg.track_ids.data(), detectionMsg.detection_count};

  // Assign track ids
  rclcpp::Time msgTime(detectionMsg.header.stamp);
  std::chrono::system_clock::time_point sysTime{std::chrono::nanoseconds{msgTime.nanoseconds()}};
  trackMatcher_.update(sysTime, boxes, msgTrackIds);
}

} // namespace zoo
