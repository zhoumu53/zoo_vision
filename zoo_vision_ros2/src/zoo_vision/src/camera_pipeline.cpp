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

#include "zoo_vision/camera_pipeline.hpp"

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

CameraPipeline::CameraPipeline(const rclcpp::NodeOptions &options, int nameIndex)
    : rclcpp::Node(std::format("pipeline_{}", nameIndex), options),
      cameraName_{declare_parameter<std::string>("camera_name")}, cudaStream_{at::cuda::getStreamFromPool()},
      trackMatcher_{}, segmenter_{nameIndex, cameraName_, trackMatcher_, cudaStream_},
      identifier_{nameIndex, cameraName_, trackMatcher_, cudaStream_},
      behaviourer_{nameIndex, cameraName_, cudaStream_} {

  readConfig(getConfig());

  // Subscribe to receive images from camera
  const auto imageTopic = cameraName_ + "/image";
  imageSubscriber_ = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
      *this, imageTopic, 10,
      [this](std::shared_ptr<const zoo_msgs::msg::Image12m> msg) { this->onImage(std::move(msg)); });

  // Publish detections
  const auto detectionsTopic = cameraName_ + "/detections";
  detectionPublisher_ = rclcpp::create_publisher<zoo_msgs::msg::Detection>(*this, detectionsTopic, 10);
  RCLCPP_INFO(get_logger(), "Publishing detections at %s", detectionsTopic.c_str());
}

void CameraPipeline::readConfig(const nlohmann::json & /*config*/) {}

void CameraPipeline::onImage(std::shared_ptr<const zoo_msgs::msg::Image12m> imageMsgPtr) {
  const auto &imageMsg = *imageMsgPtr;

  rateSampler_.tick();
  at::cuda::CUDAStreamGuard streamGuard{cudaStream_};

  // Allocate detection message so we can already start putting things here
  auto detectionMsgPtr = std::make_unique<zoo_msgs::msg::Detection>();
  auto &detectionMsg = *detectionMsgPtr;
  detectionMsg.header = imageMsg.header;
  addRosKeyValue(detectionMsg.timings.items_hz, std::format("seg", cameraName_), rateSampler_.rateHz());

  ////////////////////////////////////////////////////////////
  // Prepare image for segmentation network
  const cv::Mat3b img = wrapMat3bFromMsg(imageMsg);
  assert(imageMsg.step == imageMsg.width * 3 * sizeof(char));
  at::Tensor imageTensorCPU =
      at::from_blob(img.data, {img.rows, img.cols, img.channels()}, at::TensorOptions().dtype(at::kByte))
          .permute({2, 0, 1});

  at::Tensor imageTensor;
  {
    // Convert to float
    imageTensor = imageTensorCPU.to(at::kCUDA).to(at::kFloat) / 255.0f;
  }

  // Segmentation
  segmenter_.onImage(detectionMsg, imageTensor);

  // Identification
  if (detectionMsg.detection_count > 0) {
    auto msgBboxes = std::span{detectionMsg.bboxes.data(), detectionMsg.detection_count};
    auto msgTrackIds = std::span{detectionMsg.track_ids.data(), detectionMsg.detection_count};

    at::Tensor patches;
    cropper_.extractCrops(patches, imageTensor, detectionMsg.scale_image_from_detection, msgBboxes);

    identifier_.onDetection(detectionMsg, patches, msgTrackIds);
    behaviourer_.onDetection(detectionMsg, patches);
  }

  // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Publishing detection");
  detectionPublisher_->publish(std::move(detectionMsgPtr));
}

} // namespace zoo
