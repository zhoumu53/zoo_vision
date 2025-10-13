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

#include "zoo_vision/compute_device.hpp"
#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/timings.hpp"
#include "zoo_vision/utils.hpp"

#include <ATen/core/List.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <nlohmann/json.hpp>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <sensor_msgs/image_encodings.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <stacktrace>
#include <string.h>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

CameraPipeline::CameraPipeline(const rclcpp::NodeOptions &options, int nameIndex)
    : rclcpp::Node(std::format("pipeline_{}", nameIndex), options),
      cameraName_{declare_parameter<std::string>("camera_name")}, calibration_{cameraName_}, cudaStream_{},
      trackMatcher_{}, segmenter_{makeSegmenter(nameIndex, cameraName_, cudaStream_)}, locator_{calibration_},
      identifier_{makeIdentifier(nameIndex, cameraName_, trackMatcher_, cudaStream_)},
      behaviourer_{makeBehaviourer(nameIndex, cameraName_, cudaStream_)} {
  readConfig(getConfig());

  rateLimiter_ = gCameraLimiters.empty() ? nullptr : gCameraLimiters[cameraName_].get();

  // Set up paths to store improvement images
  if (recordDetectionLoss_ || recordTracks_ || recordKeyframes_ || recordBehaviourChange_ || recordMasks_) {
    std::filesystem::create_directories(rootPathImprove_);
  }

  // Subscribe to receive images from camera
  const auto imageTopic = cameraName_ + "/image";
  imageSubscriber_ = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
      *this, imageTopic, 10, [this](std::shared_ptr<zoo_msgs::msg::Image12m> msg) {
        try {
          this->onImage(std::move(msg));
        } catch (const ZooVisionError &e) {
          std::cerr << e.trace << std::endl;
          RCLCPP_ERROR(this->get_logger(), "Exception:\n%s\nTerminating\n", e.what());
          std::terminate();
        } catch (const std::exception &e) {
          auto trace = std::stacktrace::current();
          std::cerr << trace << std::endl;
          RCLCPP_ERROR(this->get_logger(), "Exception:\n%s\nTerminating\n", e.what());
          std::terminate();
        }
      });

  // Publish detections
  {
    const auto topic = cameraName_ + "/detections";
    detectionPublisher_ = rclcpp::create_publisher<zoo_msgs::msg::Detection>(*this, topic, 10);
    RCLCPP_INFO(get_logger(), "Publishing detections at %s", topic.c_str());
  }
  {
    const auto topic = cameraName_ + "/track_state";
    trackStatePublisher_ = rclcpp::create_publisher<zoo_msgs::msg::TrackState>(*this, topic, 10);
    RCLCPP_INFO(get_logger(), "Publishing track state at %s", topic.c_str());
  }
  {
    const auto topic = cameraName_ + "/track_closed";
    trackClosedPublisher_ = rclcpp::create_publisher<zoo_msgs::msg::TrackClosed>(*this, topic, 10);
    RCLCPP_INFO(get_logger(), "Publishing track closed at %s", topic.c_str());
  }
}

void CameraPipeline::readConfig(const nlohmann::json &config) {
  // Settings
  recordDetectionLoss_ = config["record_detection_loss"].get<bool>();
  recordKeyframes_ = config["record_keyframes"].get<bool>();
  recordTracks_ = config["record_tracks"].get<bool>();
  recordBehaviourChange_ = config["record_behaviour_change"].get<bool>();
  recordMasks_ = config["record_masks"].get<bool>();

  const auto detectionImageJson = config["detection"]["image"];
  detectionImageSize_ = Vector2i{detectionImageJson["width"].get<int>(), detectionImageJson["height"].get<int>()};

  rootPathImprove_ = config["record_root"].get<std::string>();
}

void CameraPipeline::dynamicConfig(Vector2i imageSize) {
  dynamicConfigDone_ = true;
  calibration_.setImageSize(imageSize);
  segmenter_->setImageSize(detectionImageSize_);
  locator_.setDetectionImageSize(segmenter_->getDetectionImageSize());
}

void CameraPipeline::onImage(std::shared_ptr<zoo_msgs::msg::Image12m> imageMsgPtr) {
  const auto &imageMsg = *imageMsgPtr;

  // RCLCPP_INFO(get_logger(), "Pipeline, id=%s", imageMsg.header.frame_id.data.data());

  rateSampler_.tick();
  const SysTime sysTime = sysTimeFromRos(imageMsg.header.stamp);
  std::optional<at::cuda::CUDAStreamGuard> streamGuard;
  if (cudaStream_.has_value()) {
    streamGuard.emplace(*cudaStream_);
  }

  if (!dynamicConfigDone_) {
    // First image received, initialized things that need to know the image size
    dynamicConfig(Vector2i(imageMsg.width, imageMsg.height));
  }
  // Allocate detection message so we can already start putting things here
  auto detectionMsgPtr = std::make_unique<zoo_msgs::msg::Detection>();
  auto &detectionMsg = *detectionMsgPtr;
  detectionMsg.header = imageMsg.header;
  addRosKeyValue(detectionMsg.timings.items_hz, std::format("seg", cameraName_), rateSampler_.rateHz());

  ////////////////////////////////////////////////////////////
  // Map ros message to opencv
  cv::Mat3b img = wrapMat3bFromMsg(imageMsg);
  CHECK_EQ(imageMsg.step, imageMsg.width * 3 * sizeof(char));

  ////////////////////////////////////////////////////////////
  // Black out masked areas
  for (const auto &poly : calibration_.maskPolygons_) {
    cv::fillConvexPoly(img, poly, cv::Scalar(0, 0, 0));
  }

  ////////////////////////////////////////////////////////////
  // Prepare image for segmentation network

  at::Tensor imageTensorCPU =
      at::from_blob(img.data, {img.rows, img.cols, img.channels()}, at::TensorOptions().dtype(at::kByte))
          .permute({2, 0, 1});

  const at::Tensor imageTensor_f32 = imageTensorCPU.to(g_computeDevice, /*non_blocking*/ true).to(at::kFloat);
  // const at::Tensor imageNorm = normalizer_.normalize(imageTensor_f32);

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Segmentation
  SegmenterResult segmenterResult;
  {
    cv::Mat3b detectionImage;
    cv::resize(img, detectionImage, {detectionImageSize_.x(), detectionImageSize_.y()});

    const Vector2i detectionImageSize = segmenter_->getDetectionImageSize();
    const int64_t MAX_DETECTION_COUNT = zoo_msgs::msg::Detection::MAX_DETECTION_COUNT;
    detectionMsg.masks.sizes[0] = MAX_DETECTION_COUNT;
    detectionMsg.masks.sizes[1] = detectionImageSize.y();
    detectionMsg.masks.sizes[2] = detectionImageSize.x();
    segmenterResult.masks = mapRosTensor(detectionMsg.masks);

    segmenter_->onImage(segmenterResult, /*imageNorm*/ {}, detectionImage);

    // Copy results to detectionMsg
    detectionMsg.detection_count = segmenterResult.bboxesInDetection.size();
    detectionMsg.masks.sizes[0] = detectionMsg.detection_count;
    detectionMsg.scalex_image_from_detection = static_cast<float>(imageMsg.width) / detectionImageSize[0];
    detectionMsg.scaley_image_from_detection = static_cast<float>(imageMsg.height) / detectionImageSize[1];
    for (auto [i, bbox] : std::views::enumerate(segmenterResult.bboxesInDetection)) {
      copyBboxToRos(detectionMsg.bboxes[i], bbox);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Localize in world
  {
    Eigen::Map<Eigen::Matrix3Xf> worldPositionsMap(detectionMsg.world_positions.data(), 3,
                                                   detectionMsg.world_positions.size() / 3);

    locator_.worldFromBboxes(worldPositionsMap, segmenterResult.bboxesInDetection);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Assign track ids
  auto trackIds = std::span{detectionMsg.track_ids.data(), detectionMsg.detection_count};
  auto trackUpdateStats = trackMatcher_.update(sysTime, segmenterResult.bboxesInDetection, trackIds);
  if (recordDetectionLoss_) {
    if (!trackUpdateStats.justMissedTracks.empty()) {
      saveImageToImproveDetection(sysTime, img);
    }
  }
  if (recordMasks_) {
    const auto frameId = getMsgString(imageMsg.header.frame_id);
    recordMasks(frameId, trackIds, segmenterResult.masks);
  }
  for (const auto &ptrack : trackUpdateStats.closedTracks) {
    const auto &track = *ptrack;
    publishTrackClosed(imageMsg.header, track);
    if (recordKeyframes_) {
      saveKeyframes(track);
    }
    if (recordTracks_) {
      // moveTrackImagesToIdentityPath(track);
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Identification
  if (detectionMsg.detection_count > 0) {
    auto bboxes = std::span{detectionMsg.bboxes.data(), detectionMsg.detection_count};

    at::Tensor patches_f32;
    cropper_.extractCrops(patches_f32, imageTensor_f32,
                          {detectionMsg.scalex_image_from_detection, detectionMsg.scaley_image_from_detection}, bboxes);
    at::Tensor patchesNorm = normalizer_.normalize(patches_f32);

    // Save track images
    if (recordTracks_) {
      recordTracks(sysTime, trackIds, patches_f32.to(at::kByte));
    }

    std::vector<bool> patchQualities = quality_.check(patchesNorm);

    const at::Tensor embeddings = embedder_.embed(patchesNorm);
    for (const auto [i, trackId] : std::views::enumerate(trackIds)) {
      TrackData &track = trackMatcher_.getTrackData(trackId);

      // Is the patch good enough to check for id?
      if (patchQualities[i]) {
        // Yes, try to add a new keyframe
        const at::Tensor patch_f32 = patches_f32[i];
        const auto newKeyframeIdx = track.keyframeStore.maybeAddKeyframe(patch_f32, embeddings[i]);
        if (newKeyframeIdx.has_value()) {
          // New keyframe has been added, execute id net
          identifier_->onKeyframe(*newKeyframeIdx, patchesNorm[i], track);

          publishTrackState(imageMsg.header, *newKeyframeIdx, track);
        }
      }

      // Show selected identity on the detection
      detectionMsg.identity_ids[i] = track.selectedIdentity;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Behaviour
    behaviourer_->onDetection(detectionMsg, patchesNorm);
    for (const auto [i, trackId, behaviourId] :
         std::views::zip(std::views::iota(0), trackIds, detectionMsg.behaviour_ids)) {
      TrackData &track = trackMatcher_.getTrackData(trackId);
      if (track.selectedBehaviour != INVALID_BEHAVIOUR && track.selectedBehaviour != behaviourId) {
        // Behaviour changed, save image for network improvement
        if (recordBehaviourChange_) {
          const at::Tensor patch = patches_f32[i].to(at::kByte);
          saveImageToImproveBehaviour(sysTime, behaviourId, patch);
        }
      }
      // Remember for the next frame
      track.selectedBehaviour = behaviourId;
    }
  }

  // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Publishing detection");
  detectionPublisher_->publish(std::move(detectionMsgPtr));

  if (rateLimiter_) {
    rateLimiter_->signalProcessingComplete();
  }
}

void CameraPipeline::publishTrackState(const zoo_msgs::msg::Header &imageHeader, TKeyframeIndex newKeyframeIndex,
                                       const TrackData &track) {
  auto msgPtr = std::make_unique<zoo_msgs::msg::TrackState>();
  auto &msg = *msgPtr;
  msg.header = imageHeader;
  msg.track_id = track.id;
  msg.new_keyframe_index = newKeyframeIndex;

  // Copy image
  const at::Tensor keyframeImage = track.keyframeStore.getKeyframeImage(newKeyframeIndex).permute({1, 2, 0});
  const uint32_t height = keyframeImage.size(0);
  const uint32_t width = keyframeImage.size(1);
  const uint32_t channels = 3;
  msg.new_keyframe.height = height;
  msg.new_keyframe.width = width;
  msg.new_keyframe.step = width * channels;
  // msg.keyframe_mosaic.encoding = sensor_msgs::image_encodings::RGB8;
  // msg.keyframe_mosaic.data.resize(height * msg.keyframe_mosaic.step);

  at::Tensor msgImage =
      at::from_blob(msg.new_keyframe.data.data(), {height, width, channels}, at::TensorOptions().dtype(at::kByte));
  msgImage.copy_(keyframeImage);

  msg.selected_identity = track.selectedIdentity;
  for (const auto [idx, votes] : std::views::enumerate(track.identityHistogram.getVotes())) {
    msg.identity_probs[idx] = votes;
  }

  trackStatePublisher_->publish(std::move(msgPtr));
}

void CameraPipeline::publishTrackClosed(const zoo_msgs::msg::Header &imageHeader, const TrackData &track) {
  auto msgPtr = std::make_unique<zoo_msgs::msg::TrackClosed>();
  auto &msg = *msgPtr;
  msg.header = imageHeader;
  msg.track_id = track.id;
  msg.track_length = track.trackLength;
  msg.selected_identity = track.selectedIdentity;
  for (const auto [idx, votes] : std::views::enumerate(track.identityHistogram.getVotes())) {
    msg.identity_probs[idx] = votes;
  }

  trackClosedPublisher_->publish(std::move(msgPtr));
}

void CameraPipeline::saveImageToImproveDetection(SysTime time, const cv::Mat3b &cvImg) {
  const std::filesystem::path rootPath = rootPathImprove_ / "detection" / std::format("{:%Y-%m-%d}", time);
  std::filesystem::create_directories(rootPath);

  // Make name based on time
  const auto timeSeconds = secondsTimePointFromTimePoint(time);
  const std::filesystem::path imgName = rootPath / std::format("{}_{:%H%M%S}.jpg", cameraName_, timeSeconds);

  // Save
  cv::Mat cvImgBgr;
  cv::cvtColor(cvImg, cvImgBgr, cv::COLOR_RGB2BGR);
  const auto res = cv::imwrite(imgName.c_str(), cvImgBgr);
  if (res) {
    RCLCPP_INFO(get_logger(), "Saved image from just missed track: %s", imgName.c_str());
  } else {
    RCLCPP_ERROR(get_logger(), "Error saving image from just missed track: %s", imgName.c_str());
  }
}

void CameraPipeline::saveImageToImproveBehaviour(SysTime time, TBehaviour behaviourId, const at::Tensor &img) {
  static std::array<std::string, 4> BEHAVIOUR_NAMES = {"00_Invalid", "01_Standing", "02_SleepL", "03_SleepR"};
  std::filesystem::path rootPath =
      rootPathImprove_ / "behaviour" / BEHAVIOUR_NAMES[behaviourId] / std::format("{:%Y-%m-%d}", time);
  std::filesystem::create_directories(rootPath);

  // Make name based on time
  const auto timeSeconds = secondsTimePointFromTimePoint(time);
  const std::filesystem::path imgName = rootPath / std::format("{}_{:%H%M%S}.jpg", cameraName_, timeSeconds);

  // Save
  const auto res = saveTensorImage(img, imgName);
  if (res) {
    RCLCPP_INFO(get_logger(), "Saved image from behaviour change: %s", imgName.c_str());
  } else {
    RCLCPP_ERROR(get_logger(), "Error saving image from behaviour change: %s", imgName.c_str());
  }
}

void CameraPipeline::recordTracks(const SysTime time, const std::span<const uint32_t> trackIds,
                                  const at::Tensor &patches) {
  const std::filesystem::path rootPath = rootPathImprove_ / "tracks";
  constexpr auto MIN_TRACK_LENGTH = std::chrono::seconds(2);
  constexpr auto TIME_BETWEEN_RECORDS = std::chrono::seconds(2);

  const SecondsTimePoint secondsTime = secondsTimePointFromTimePoint(time);

  for (auto &&[idx, trackId] : std::views::enumerate(trackIds)) {
    TrackData &track = trackMatcher_.getTrackData(trackId);

    if (!track.lastImageSaved.has_value()) {
      // Check that the track has been alive long enough
      const auto ellapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(time - track.startTime);
      if (ellapsedSeconds < MIN_TRACK_LENGTH) {
        continue;
      }
    } else {
      // Check that the last record is not too recent
      const auto ellapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(time - *track.lastImageSaved);
      if (ellapsedSeconds < TIME_BETWEEN_RECORDS) {
        continue;
      }
    }

    const std::filesystem::path trackDir =
        rootPath / std::format("{:%Y-%m-%d}", track.startTime) / cameraName_ / std::format("{:06d}", track.id);

    const std::string imgName =
        std::format("{}_{:%Y%m%d_%H%M%S}_t{}_{}.png", cameraName_, secondsTime, trackId, track.trackLength);

    // The first image makes sure we create the directory
    if (!track.lastImageSaved.has_value()) {
      std::filesystem::create_directories(trackDir);
    }
    saveTensorImage(patches[idx], trackDir / imgName);

    track.lastImageSaved = time;
  }
}

void CameraPipeline::saveKeyframes(const TrackData &track) {
  const auto count = track.keyframeStore.getCount();
  if (count == 0) {
    // No need to create directories for dummy tracks
    return;
  }

  const std::vector<std::string> identityNames = {"00_Invalid", "01_Chandra", "02_Indi",
                                                  "03_Fahra",   "04_Panang",  "05_Thai"};

  const std::filesystem::path rootPath = rootPathImprove_ / "keyframes" / identityNames[track.selectedIdentity] /
                                         std::format("{:%Y-%m-%d}", track.startTime);
  const std::filesystem::path trackDir = rootPath / cameraName_ / std::format("{:06d}", track.id);
  std::filesystem::create_directories(trackDir);
  for (const auto i : std::views::iota(0u, count)) {
    const auto name = trackDir / std::format("k{}.jpg", i);
    saveTensorImage(track.keyframeStore.getKeyframeImage(i), name.c_str());
  }
}

void CameraPipeline::moveTrackImagesToIdentityPath(const TrackData &track) {

  try {
    const std::filesystem::path rootPath = rootPathImprove_ / "tracks" / std::format("{:%Y-%m-%d}", track.startTime);
    const std::filesystem::path trackDir = rootPath / cameraName_ / std::format("{:06d}", track.id);
    if (!std::filesystem::exists(trackDir)) {
      return;
    }

    const std::vector<std::string> identityNames = {"00_Invalid", "01_Chandra", "02_Indi",
                                                    "03_Fahra",   "04_Panang",  "05_Thai"};

    const std::filesystem::path idRootDir = rootPath / "identity" / identityNames[track.selectedIdentity];
    const std::filesystem::path newTrackDir = idRootDir / std::format("{}_{:06d}", cameraName_, track.id);
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

void CameraPipeline::recordMasks(std::string_view frameId, std::span<TrackId> trackIds, const at::Tensor &masks) {
  int frameId2 = parseInt(frameId);

  for (const auto [index, trackId] : std::views::enumerate(trackIds)) {
    const std::filesystem::path dir = rootPathImprove_ / "masks" / cameraName_ / std::format("track_{:04}", trackId);
    const std::filesystem::path filename = dir / (std::format("frame_{:06}", frameId2) + ".png");

    const cv::Mat1b mask = wrapCvFromTensor1b(masks.index({index}));

    std::filesystem::create_directories(dir);
    cv::imwrite(filename.string(), mask);
  }
}

} // namespace zoo
