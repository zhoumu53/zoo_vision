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

  trackMatcher_.onTrackCloseEvent = [this](TrackId id) { this->onTrackClosed(id); };

  // Subscribe to receive images from camera
  const auto imageTopic = cameraName_ + "/image";
  imageSubscriber_ = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
      *this, imageTopic, 10, [this](std::shared_ptr<const zoo_msgs::msg::Image12m> msg) {
        try {
          this->onImage(std::move(msg));
        } catch (const std::exception &e) {
          RCLCPP_ERROR(this->get_logger(), "Exception:\n%s\nTerminating\n", e.what());
          std::terminate();
        }
      });

  // Publish detections
  const auto detectionsTopic = cameraName_ + "/detections";
  detectionPublisher_ = rclcpp::create_publisher<zoo_msgs::msg::Detection>(*this, detectionsTopic, 10);
  RCLCPP_INFO(get_logger(), "Publishing detections at %s", detectionsTopic.c_str());
}

void CameraPipeline::readConfig(const nlohmann::json &config) {
  // Settings
  recordTracks_ = config["record_tracks"].get<bool>();
}

// std::mutex globalMutex;

void CameraPipeline::onImage(std::shared_ptr<const zoo_msgs::msg::Image12m> imageMsgPtr) {
  const auto &imageMsg = *imageMsgPtr;
  // std::optional<std::lock_guard<std::mutex>> lock;
  // if (imageMsg.header.frame_id.data[0] == '7' && imageMsg.header.frame_id.data[1] == '3') {
  //   lock.emplace(globalMutex);
  // }
  // RCLCPP_INFO(get_logger(), "Pipeline, id=%s", imageMsg.header.frame_id.data.data());
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

  const at::Tensor rawImageTensor = imageTensorCPU.to(at::kCUDA).to(at::kFloat);
  const at::Tensor imageTensor = normalizer_.normalize(rawImageTensor);

  // Segmentation
  segmenter_.onImage(detectionMsg, imageTensor);

  // Identification
  if (detectionMsg.detection_count > 0) {
    auto bboxes = std::span{detectionMsg.bboxes.data(), detectionMsg.detection_count};
    auto trackIds = std::span{detectionMsg.track_ids.data(), detectionMsg.detection_count};

    at::Tensor patches_u8;
    cropper_.extractCrops(patches_u8, rawImageTensor,
                          {detectionMsg.scalex_image_from_detection, detectionMsg.scaley_image_from_detection}, bboxes);
    at::Tensor patches_f32 = normalizer_.normalize(patches_u8);

    // Save track images
    if (recordTracks_) {
      recordTracks(imageMsg, trackIds, patches_u8);
    }

    const at::Tensor embeddings = embedder_.embed(patches_u8);
    for (const auto [i, trackId] : std::views::enumerate(trackIds)) {
      const at::Tensor patch_u8 = patches_u8[i];
      TrackData &track = trackMatcher_.getTrackData(trackId);
      const auto newKeyframeIdx = track.keyframeStore.maybeAddKeyframe(patch_u8, embeddings[i]);
      if (newKeyframeIdx.has_value()) {
        // New keyframe has been added
        identifier_.onKeyframe(*newKeyframeIdx, patches_f32[i], track);
      }
      identifier_.addDetectionInfo(detectionMsg, i, track);
    }
    behaviourer_.onDetection(detectionMsg, patches_f32);
  }

  // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Publishing detection");
  detectionPublisher_->publish(std::move(detectionMsgPtr));
}

void CameraPipeline::recordTracks(const zoo_msgs::msg::Image12m &imageMsg, const std::span<const uint32_t> trackIds,
                                  const at::Tensor &patches) {
  constexpr int MIN_TRACK_LENGTH = 20;
  constexpr int SKIP_COUNT = 20;
  static std::filesystem::path rootPath = "/media/dherrera/ElephantExternal/elephants/tracks/new";

  using Clock = std::chrono::system_clock;
  using nanoseconds = std::chrono::nanoseconds;
  using seconds = std::chrono::seconds;

  const Clock::time_point frameTimeNs{nanoseconds{rclcpp::Time(imageMsg.header.stamp).nanoseconds()}};
  const auto frameTime =
      std::chrono::time_point<Clock, seconds>{std::chrono::duration_cast<seconds>(frameTimeNs.time_since_epoch())};

  for (auto &&[idx, trackId] : std::views::enumerate(trackIds)) {
    TrackData &track = trackMatcher_.getTrackData(trackId);
    const std::string imgName =
        std::format("{}_{:%Y%m%d_%H%M%S}_t{}_{}.png", cameraName_, frameTime, trackId, track.trackLength);
    const std::filesystem::path trackDir = rootPath / cameraName_ / std::format("{:06d}", trackId);

    if (track.trackLength <= MIN_TRACK_LENGTH) {
      continue;
    }
    if ((track.trackLength - MIN_TRACK_LENGTH - 1) % SKIP_COUNT != 0) {
      continue;
    }

    if (trackId > 3000) {
      std::terminate();
    }

    // The first image makes sure we create the directory
    // and it is also stored at the root for quick preview
    if (track.trackLength == MIN_TRACK_LENGTH + 1) {
      std::filesystem::create_directories(trackDir);
      saveTensorImage(patches[idx], rootPath / imgName);
    }
    saveTensorImage(patches[idx], trackDir / imgName);
  }
}

void CameraPipeline::onTrackClosed(TrackId trackId) {
  try {
    const TrackData &data = trackMatcher_.getTrackData(trackId);
    static std::filesystem::path rootPath = "/media/dherrera/ElephantExternal/elephants/tracks/new";
    const std::filesystem::path trackDir = rootPath / cameraName_ / std::format("{:06d}", trackId);
    if (!std::filesystem::exists(trackDir)) {
      return;
    }

    const auto [identityId, voteCount] = data.identityHistogram.getHighest();
    const std::vector<std::string> identityNames = {"00_Invalid", "01_Chandra", "02_Indi",
                                                    "03_Fahra",   "04_Panang",  "05_Thai"};

    const std::filesystem::path idRootDir = rootPath / "identity" / identityNames[identityId + 1];
    const std::filesystem::path newTrackDir = idRootDir / std::format("{}_{:06d}", cameraName_, trackId);
    std::filesystem::create_directories(newTrackDir);

    for (const auto &file : std::filesystem::directory_iterator(trackDir)) {
      const std::filesystem::path newImg = newTrackDir / file.path().filename();
      std::filesystem::rename(file.path(), newImg);
    }
    std::filesystem::remove(trackDir);
  } catch (const std::exception &ex) {
    RCLCPP_ERROR(get_logger(), "Error moving track: %s", ex.what());
  } catch (...) {
    RCLCPP_ERROR(get_logger(), "Error moving track: ???");
  }
}
} // namespace zoo
