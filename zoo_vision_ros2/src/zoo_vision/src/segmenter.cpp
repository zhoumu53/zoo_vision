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
  calibratedCameraSize_ = Eigen::Vector2i{config["cameras"][cameraName_]["intrinsics"]["width"].get<int>(),
                                          config["cameras"][cameraName_]["intrinsics"]["height"].get<int>()};
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

struct MaskComponentResult {
  at::Tensor mask;
  int32_t area;
  Eigen::AlignedBox2f bbox;
};

std::optional<MaskComponentResult> getBestMaskComponent(const at::Tensor rawMask) {
  // Segmentation network doesn't always produce a single coherent segmentation
  // Select largest connected component
  const cv::Mat1b cvRawMask = wrapCvFromTensor1b(rawMask);

  cv::Mat labels, stats, centroids;
  cv::connectedComponentsWithStats(cvRawMask, labels, stats, centroids, 8, CV_16U);
  if (stats.empty()) {
    return std::nullopt;
  }

  // Find biggest component
  int bestIdx = 0;
  int32_t bestArea = 0;
  for (int i = 1; i < stats.size[0]; ++i) {
    const int32_t area = stats.at<int32_t>(i, cv::CC_STAT_AREA);
    if (area > bestArea) {
      bestIdx = i;
      bestArea = area;
    }
  }

  // Make mask
  const at::Tensor labelsTensor =
      at::from_blob(labels.data, {labels.rows, labels.cols}, at::TensorOptions().dtype(at::kUInt16));
  at::Tensor bestMask = labelsTensor.eq(bestIdx);

  // Make aligned box
  const Eigen::Vector2f x0{static_cast<float>(stats.at<int32_t>(bestIdx, cv::CC_STAT_LEFT)),
                           static_cast<float>(stats.at<int32_t>(bestIdx, cv::CC_STAT_TOP))};
  const Eigen::Vector2f bboxSize{static_cast<float>(stats.at<int32_t>(bestIdx, cv::CC_STAT_WIDTH)),
                                 static_cast<float>(stats.at<int32_t>(bestIdx, cv::CC_STAT_HEIGHT))};

  auto bbox = Eigen::AlignedBox2f(x0, x0 + bboxSize);

  return {{bestMask, bestArea, bbox}};
}

void Segmenter::onImage(zoo_msgs::msg::Detection &detectionMsg, std::vector<Eigen::AlignedBox2f> &boxes,
                        const at::Tensor &imageTensor) {
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
  const int32_t imageWidth = imageTensor.size(2);
  const int32_t imageHeight = imageTensor.size(1);
  detectionMsg.scalex_image_from_detection = static_cast<float>(imageWidth) / maskWidth;
  detectionMsg.scaley_image_from_detection = static_cast<float>(imageHeight) / maskHeight;

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

  boxes.clear();
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

    // Segmentation network doesn't always produce a single coherent segmentation
    // Select largest connected component
    const auto maybeMask = getBestMaskComponent(masks[inputIndex]);
    if (!maybeMask.has_value()) {
      continue;
    }
    const auto &[mask, area, bbox] = *maybeMask;

    // Threshold area
    const int AREA_THRESHOLD = 5 * 5; // 5x5 box in a map of ~(150x270)
    if (area < AREA_THRESHOLD) {
      continue;
    }

    // Threshold intersection with others
    int maxIntersectionArea = 0;
    for (uint32_t maskIdx = 0; maskIdx < detectionMsg.detection_count; maskIdx++) {
      const int intersection = (masksMap[maskIdx] * mask).sum().item<int>();
      if (intersection > maxIntersectionArea) {
        maxIntersectionArea = intersection;
      }
    }
    const float intersectionRatio = static_cast<float>(maxIntersectionArea) / area;
    const float MAX_INTERSECTION_RATIO = 0.5f;
    if (intersectionRatio > MAX_INTERSECTION_RATIO) {
      // std::cout << "Dropped mask with " << intersectionRatio << " intersection ratio" << std::endl;
      continue;
    }

    // All checks passed, keep this detection
    const auto outputIndex = detectionMsg.detection_count;
    detectionMsg.detection_count += 1;

    // Track ids are decided later

    // Mask
    masksMap[outputIndex].copy_(mask);

    boxes.push_back(bbox);
    copyBboxToRos(detectionMsg.bboxes[outputIndex], bbox);

    // Project to world
    const float32_t scale_calibratedFromImage = static_cast<float32_t>(calibratedCameraSize_[0]) / imageWidth;
    auto scalePoint = [&](Eigen::Vector2f p) {
      return Eigen::Vector2f{p.x() * detectionMsg.scalex_image_from_detection * scale_calibratedFromImage,
                             p.y() * detectionMsg.scaley_image_from_detection * scale_calibratedFromImage};
    };
    const Eigen::AlignedBox2f bboxInCalibrated = {scalePoint(bbox.min()), scalePoint(bbox.max())};
    const Eigen::Vector3f worldPosition = worldFromBbox(bboxInCalibrated);
    worldPositionsMap.col(outputIndex) = worldPosition;
  }
  detectionMsg.masks.sizes[0] = detectionMsg.detection_count;
}

Eigen::Vector3f Segmenter::worldFromBbox(const Eigen::AlignedBox2f &bbox) const {
  const auto world2FromImage = [&](Eigen::Vector2f p) { return (H_world2FromCamera_ * p.homogeneous()).hnormalized(); };
  const auto worldFromWorld2 = [](const Eigen::Vector2f &x2) { return Eigen::Vector3f{x2[0], x2[1], 0.0f}; };

  // Here we do a small trick. The initial imagePosition is at the middle-bottom of the bounding box.
  // But animals are 3D so we try to guess the center by assuming the animal is a rectangle of X-Y-Z dimensions.
  // Observing the animal from the front gives an aspect ratio of Ax=X/Z, whereas from the side the aspect is Ay=Y/Z.
  // For aspect ratio Ax we want to add an offset of Y/2 in the world plane. For aspect Ay we want to add X/2.
  // So we do a linear interpolation between Ax and Ay to find the offset to apply.
  constexpr float32_t aspectMin = 0.5f;
  constexpr float32_t aspectMax = 1.0f;
  constexpr float32_t offsetMin = 1.5f;
  constexpr float32_t offsetMax = 0.5f;

  const float32_t aspect = bbox.sizes()[0] / bbox.sizes()[1];
  float32_t offset;
  if (aspect <= aspectMin) {
    offset = offsetMin;
  } else if (aspect >= aspectMax) {
    offset = offsetMax;
  } else {
    offset = (aspect - aspectMin) / (aspectMax - aspectMin) * (offsetMax - offsetMin) + offsetMin;
  }

  const Eigen::Vector2f bboxMiddleInImage = Eigen::Vector2f{bbox.center()[0], bbox.max()[1]};
  const Eigen::Vector2f bboxMiddleInWorld2 = world2FromImage(bboxMiddleInImage);

  const Eigen::Vector2f deltaInImage = bboxMiddleInImage + Eigen::Vector2f{0.f, -1.f};
  const Eigen::Vector2f deltaInWorld2 = world2FromImage(deltaInImage);
  const Eigen::Vector2f offsetDirection = (deltaInWorld2 - bboxMiddleInWorld2).normalized();

  const Eigen::Vector2f centerInWorld2 = bboxMiddleInWorld2 + offsetDirection * offset;

  return worldFromWorld2(centerInWorld2);
}
} // namespace zoo
