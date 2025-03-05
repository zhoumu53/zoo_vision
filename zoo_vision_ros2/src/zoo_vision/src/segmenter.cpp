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

Segmenter::Segmenter(const rclcpp::NodeOptions &options, int nameIndex)
    : Node(std::format("segmenter_{}", nameIndex), options), cudaStream_{at::cuda::getStreamFromPool()} {
  at::InferenceMode inferenceGuard;

  cameraName_ = declare_parameter<std::string>("camera_name");
  RCLCPP_INFO(get_logger(), "Starting segmenter for %s", cameraName_.c_str());

  readConfig(getConfig());

  // Subscribe to receive images from camera
  const auto imageTopic = cameraName_ + "/image";
  const auto detectionsTopic = cameraName_ + "/detections";
  const auto detectionsImageTopic = cameraName_ + "/detections/image";
  imageSubscriber_ = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
      *this, imageTopic, 10, [this](const zoo_msgs::msg::Image12m &msg) { this->onImage(msg); });

  // Publish detections
  detectionImagePublisher_ = rclcpp::create_publisher<zoo_msgs::msg::Image12m>(*this, detectionsImageTopic, 10);
  detectionPublisher_ = rclcpp::create_publisher<zoo_msgs::msg::Detection>(*this, detectionsTopic, 10);

  identifier_ = std::make_shared<Identifier>(options, nameIndex);
}

void Segmenter::readConfig(const nlohmann::json &config) {
  // Camera calibration
  H_mapFromWorld2_ = config["map"]["T_map_from_world2"];
  H_world2FromCamera_ = config["cameras"][cameraName_]["H_world2_from_camera"];

  // Load model
  const std::filesystem::path modelPath = std::filesystem::canonical(getDataPath() / config["models"]["segmentation"]);
  loadModel(modelPath);
  elephant_label_id_ = config["models"]["elephant_label_id"].get<int>();
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

void Segmenter::onImage(const zoo_msgs::msg::Image12m &imageMsg) {
  rateSampler_.tick();

  at::cuda::CUDAStreamGuard streamGuard{cudaStream_};
  // at::InferenceMode inferenceGuard; // Runtime error: Global alloc not supported yet
  at::NoGradGuard nograd;
  std::optional<nvtx3::scoped_range> nvtxLabel{"seg_before (" + cameraName_ + ")"};

  at::cuda::CUDAEvent eventBeforeNetwork{cudaEventDefault}, eventAfterNetwork{cudaEventDefault};

  // Allocate detection message so we can already start putting things here
  auto detectionMsg = std::make_unique<zoo_msgs::msg::Detection>();
  detectionMsg->header = imageMsg.header;

  const cv::Mat3b img = wrapMat3bFromMsg(imageMsg);
  const float inputAspect = static_cast<float>(imageMsg.width) / imageMsg.height;
  const int DETECTION_HEIGHT = 600;
  at::IValue detectionResult;

  auto detectionImageMsg = std::make_unique<zoo_msgs::msg::Image12m>();
  detectionImageMsg->header = imageMsg.header;
  setMsgString(detectionImageMsg->encoding, sensor_msgs::image_encodings::BGR8);
  detectionImageMsg->height = DETECTION_HEIGHT;
  detectionImageMsg->width = DETECTION_HEIGHT * inputAspect;
  detectionImageMsg->step = detectionImageMsg->width * 3 * sizeof(char);
  cv::Mat3b detectionImage = wrapMat3bFromMsg(*detectionImageMsg);

  cv::resize(img, detectionImage, detectionImage.size());

  // TODO: accept input as uint8
  cv::Mat3f detectionImagef;
  detectionImage.convertTo(detectionImagef, CV_32FC3, 1.0f / 255);

  at::Tensor imageTensor =
      at::from_blob(detectionImagef.data, {detectionImagef.rows, detectionImagef.cols, detectionImagef.channels()},
                    at::TensorOptions().dtype(at::kFloat));

  imageTensor = imageTensor.permute({2, 0, 1}).to(at::kCUDA);

  at::List<at::Tensor> imageList({imageTensor});
  {
    torch::jit::GraphOptimizerEnabledGuard optGuard(false);

    nvtxLabel.emplace("seg_network (" + cameraName_ + ")");

    eventBeforeNetwork.record();
    // TorchScript models require a List[IValue] as input
    detectionResult = model_.forward({imageList});
    eventAfterNetwork.record();
  }

  // Publish image to have it in sync with masks
  nvtxLabel.emplace("seg_after (" + cameraName_ + ")");
  detectionImagePublisher_->publish(std::move(detectionImageMsg));

  const auto detections = detectionResult.toTuple()->elements()[1].toList()[0].get().toGenericDict();

  // Results in gpu
  const at::Tensor masksfGpu = detections.at("masks").toTensor().squeeze(1);
  const at::Tensor boxesGpu = detections.at("boxes").toTensor();
  const at::Tensor labelsGpu = detections.at("labels").toTensor();

  // const torch::Tensor scores = detections.at("scores").toTensor().to(torch::kCPU);

  // Masks to u8
  const at::Tensor masksGpu = masksfGpu.mul(255).clamp(0, 255).to(at::kByte);

  // Check dimensions
  assert(boxesGpu.dim() == 2);

  const int64_t MAX_DETECTION_COUNT = zoo_msgs::msg::Detection::MAX_DETECTION_COUNT;
  const int64_t modelDetectionCount = std::min(MAX_DETECTION_COUNT, boxesGpu.sizes()[0]);

  assert(masksGpu.dim() == 3);
  assert(boxesGpu.sizes()[0] == masksGpu.sizes()[0]);
  const int64_t maskHeight = masksGpu.sizes()[1];
  const int64_t maskWidth = masksGpu.sizes()[2];
  const float32_t resizeFactor = static_cast<float32_t>(img.rows) / maskHeight;

  detectionMsg->detection_count = MAX_DETECTION_COUNT;
  detectionMsg->masks.sizes[0] = MAX_DETECTION_COUNT;
  detectionMsg->masks.sizes[1] = maskHeight;
  detectionMsg->masks.sizes[2] = maskWidth;
  at::Tensor masksMap = mapRosTensor(detectionMsg->masks);

  // Move all to cpu
  const at::Tensor boxesNet = boxesGpu.index({Slice(0, modelDetectionCount)}).to(at::kCPU, true);
  const at::Tensor labels = labelsGpu.index({Slice(0, modelDetectionCount)}).to(at::kCPU, true);
  const at::Tensor masks = masksGpu.index({Slice(0, modelDetectionCount)}).to(at::kCPU, true);
  cudaStreamSynchronize(cudaStream_);

  constexpr auto MS_TO_NS = 1e6f;
  addRosKeyValue(detectionMsg->timings.items_ns, std::format("{}_seg", cameraName_),
                 eventBeforeNetwork.elapsed_time(eventAfterNetwork) * MS_TO_NS);

  Eigen::Map<Eigen::Matrix3Xf> worldPositionsMap{detectionMsg->world_positions.data(), 3, modelDetectionCount};

  size_t outIndex = 0;
  const auto world_from_world2 = [](const Eigen::Vector2f &x2) { return Eigen::Vector3f{x2[0], x2[1], 0.0f}; };
  std::vector<Eigen::AlignedBox2f> boxes;
  for (int i = 0; i < modelDetectionCount; ++i) {
    const int label = labels[i].item<int>();
    if (label != elephant_label_id_) {
      continue;
    }

    // Track ids are decided later

    // Mask
    const at::Tensor mask = masks[i];
    masksMap[outIndex].copy_(mask);

    // Bbox
    const at::Tensor bbox = boxesNet[i];
    const Eigen::Vector2f x0{bbox[0].item<float32_t>(), bbox[1].item<float32_t>()};
    const Eigen::Vector2f x1{bbox[2].item<float32_t>(), bbox[3].item<float32_t>()};
    boxes.push_back(Eigen::AlignedBox2f(x0, x1));

    const Eigen::Vector2f center = (x0 + x1) / 2;
    const Eigen::Vector2f halfSize = x1 - center;
    detectionMsg->bboxes[outIndex].center[0] = center[0];
    detectionMsg->bboxes[outIndex].center[1] = center[1];
    detectionMsg->bboxes[outIndex].half_size[0] = halfSize[0];
    detectionMsg->bboxes[outIndex].half_size[1] = halfSize[1];

    // Project to world
    const Eigen::Vector2f imagePosition = (center + Eigen::Vector2f{0, halfSize[1]}) * resizeFactor;

    const Eigen::Vector3f worldPosition =
        world_from_world2((H_world2FromCamera_ * imagePosition.homogeneous()).hnormalized());
    worldPositionsMap.col(outIndex) = worldPosition;

    outIndex += 1;
  }
  detectionMsg->detection_count = outIndex;
  detectionMsg->masks.sizes[0] = outIndex;

  // Assign track ids
  trackMatcher_.update(boxes, std::span{detectionMsg->track_ids.data(), detectionMsg->detection_count});

  // Identify
  identifier_->onDetection(cudaStream_, imageTensor, std::span{detectionMsg->bboxes.data(), outIndex},
                           std::span{detectionMsg->identity_ids.data(), outIndex});

  // Add timing information
  addRosKeyValue(detectionMsg->timings.items_hz, std::format("{}_seg", cameraName_), rateSampler_.rateHz());

  detectionPublisher_->publish(std::move(detectionMsg));
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::Segmenter)