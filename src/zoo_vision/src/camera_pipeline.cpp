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
namespace {

CameraPipelineConfig readConfig() {
  const nlohmann::json &config = getConfig();
  // Settings
  CameraPipelineConfig res;
  res.recordDetectionLoss = config["record_detection_loss"].get<bool>();
  res.recordKeyframes = config["record_keyframes"].get<bool>();
  res.recordTracks = config["record_tracks"].get<bool>();
  res.recordBehaviourChange = config["record_behaviour_change"].get<bool>();
  res.recordMasks = config["record_masks"].get<bool>();

  const auto detectionImageJson = config["detection"]["image"];
  res.detectionImageSize = Vector2i{detectionImageJson["width"].get<int>(), detectionImageJson["height"].get<int>()};

  res.rootPathImprove = config["record_root"].get<std::string>();
  return res;
}
} // namespace

CameraPipeline::CameraPipeline(const rclcpp::NodeOptions &options, int nameIndex)
    : CameraPipeline(options, nameIndex, readConfig()) {}

CameraPipeline::CameraPipeline(const rclcpp::NodeOptions &options, int nameIndex, CameraPipelineConfig config)
    : rclcpp::Node(std::format("pipeline_{}", nameIndex), options),
      cameraName_{declare_parameter<std::string>("camera_name")}, config_{config}, calibration_{cameraName_},
      cudaStream_{}, trackMatcher_{config_.rootPathImprove / "tracks" / cameraName_},
      segmenter_{makeSegmenter(nameIndex, cameraName_, cudaStream_)}, locator_{calibration_},
      trackCountRecorder_(cameraName_) {

  rateLimiter_ = gCameraLimiters.empty() ? nullptr : gCameraLimiters[cameraName_].get();

  // Set up paths to store improvement images
  std::filesystem::create_directories(config_.rootPathImprove);

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

void CameraPipeline::dynamicConfig(Vector2i imageSize) {
  dynamicConfigDone_ = true;
  calibration_.setImageSize(imageSize);
  segmenter_->setImageSize(config_.detectionImageSize);
  locator_.setDetectionImageSize(segmenter_->getDetectionImageSize());
}

void CameraPipeline::onImage(std::shared_ptr<zoo_msgs::msg::Image12m> imageMsgPtr) {
  const auto &imageMsg = *imageMsgPtr;
  const auto frameId = getMsgString(imageMsg.header.frame_id);

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

  // const at::Tensor imageTensor_f32 = imageTensorCPU.to(g_computeDevice, /*non_blocking*/ true).to(at::kFloat);
  // const at::Tensor imageNorm = normalizer_.normalize(imageTensor_f32);

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Segmentation
  SegmenterResult segmenterResult;
  {
    cv::Mat3b detectionImage;
    cv::resize(img, detectionImage, {config_.detectionImageSize.x(), config_.detectionImageSize.y()});

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

  // Record number of tracks
  trackCountRecorder_.recordCount(sysTime, detectionMsg.detection_count);

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
  auto trackUpdateStats =
      trackMatcher_.update(sysTime, segmenterResult.bboxesInDetection, segmenterResult.scores, trackIds);
  if (config_.recordDetectionLoss) {
    if (!trackUpdateStats.justMissedTracks.empty()) {
      saveImageToImproveDetection(sysTime, img);
    }
  }
  if (config_.recordMasks) {
    const auto videoFile = getMsgString(imageMsg.header.video_filename);
    recordMasks(videoFile, frameId, trackIds, segmenterResult.masks);
  }
  if (!trackUpdateStats.closedTracks.empty()) {
    for (auto &ptrack : trackUpdateStats.closedTracks) {
      auto &track = *ptrack;
      publishTrackClosed(imageMsg.header, track);
      if (config_.recordKeyframes) {
        saveKeyframes(track);
      }
      if (config_.recordTracks) {
        // moveTrackImagesToIdentityPath(track);
        track.writer.close(sysTime);
      }
      trackUpdateStats.closedTracks.clear();
    }
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Identification
  if (detectionMsg.detection_count > 0) {
    auto bboxes = std::span{detectionMsg.bboxes.data(), detectionMsg.detection_count};

    at::Tensor patches_u8;
    cropper_.extractCrops(patches_u8, imageTensorCPU,
                          {detectionMsg.scalex_image_from_detection, detectionMsg.scaley_image_from_detection}, bboxes);
    // at::Tensor patchesNorm = normalizer_.normalize(patches_f32);

    // Save track images
    if (config_.recordTracks) {
      recordTracks(sysTime, frameId, trackIds, patches_u8);
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
  const std::filesystem::path rootPath = config_.rootPathImprove / "detection" / std::format("{:%Y-%m-%d}", time);
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
      config_.rootPathImprove / "behaviour" / BEHAVIOUR_NAMES[behaviourId] / std::format("{:%Y-%m-%d}", time);
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

void CameraPipeline::recordTracks(const SysTime /*time*/, std::string_view frameId,
                                  const std::span<const uint32_t> trackIds, const at::Tensor &patches) {
  at::Tensor patchesRgb = patches.permute({0, 2, 3, 1}).flip(3).to(at::kCPU).contiguous();

  static std::mutex g_mutex;
  std::lock_guard guard{g_mutex};

  for (auto &&[idx, trackId] : std::views::enumerate(trackIds)) {
    if (trackId == TrackMatcher::INVALID_TRACK_ID) {
      continue;
    }
    TrackData &track = trackMatcher_.getTrackData(trackId);

    track.writer.writeFrame(frameId, patchesRgb[idx]);
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

  const std::filesystem::path rootPath = config_.rootPathImprove / "keyframes" / identityNames[track.selectedIdentity] /
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
    const std::filesystem::path rootPath =
        config_.rootPathImprove / "tracks" / std::format("{:%Y-%m-%d}", track.startTime);
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

void CameraPipeline::recordMasks(std::string_view videoFile, std::string_view frameId, std::span<TrackId> trackIds,
                                 const at::Tensor &masks) {
  int frameId2 = parseInt(frameId);

  for (const auto [index, trackId] : std::views::enumerate(trackIds)) {
    const std::filesystem::path dir =
        config_.rootPathImprove / "masks" / videoFile / std::format("track_{:04}", trackId);
    const std::filesystem::path filename = dir / (std::format("frame_{:06}", frameId2) + ".png");

    const cv::Mat1b mask = wrapCvFromTensor1b(masks.index({index}));

    std::filesystem::create_directories(dir);
    cv::imwrite(filename.string(), mask);
  }
}

} // namespace zoo
