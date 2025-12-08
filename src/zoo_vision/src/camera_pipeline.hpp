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
#pragma once

#include "zoo_msgs/msg/detection.hpp"
#include "zoo_msgs/msg/image12m.hpp"
#include "zoo_msgs/msg/image4m.hpp"
#include "zoo_msgs/msg/track_closed.hpp"
#include "zoo_msgs/msg/track_state.hpp"
#include "zoo_vision/camera_calibration.hpp"
#include "zoo_vision/image_embedder.hpp"
#include "zoo_vision/image_normalizer.hpp"
#include "zoo_vision/image_rate_limiter.hpp"
#include "zoo_vision/patch_cropper.hpp"
#include "zoo_vision/segmenter_interface.hpp"
#include "zoo_vision/timings.hpp"
#include "zoo_vision/track_count_recorder.hpp"
#include "zoo_vision/track_matcher.hpp"
#include "zoo_vision/track_writer.hpp"
#include "zoo_vision/world_locator.hpp"

#include <Eigen/Dense>
#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/rclcpp.hpp>

#include <filesystem>
#include <string_view>

namespace zoo {

struct CameraPipelineConfig {
  Vector2i detectionImageSize;
  std::filesystem::path rootPathImprove;

  bool recordDetectionLoss;
  bool recordTracks;
  bool recordKeyframes;
  bool recordBehaviourChange;
  bool recordMasks;
};

class CameraPipeline : public rclcpp::Node {
public:
  explicit CameraPipeline(const rclcpp::NodeOptions &options = rclcpp::NodeOptions(), int nameIndex = 999);

  void onImage(std::shared_ptr<zoo_msgs::msg::Image12m> msg);

  void saveImageToImproveDetection(const SysTime time, const cv::Mat3b &cvImg);
  void saveImageToImproveBehaviour(const SysTime time, TBehaviour behaviourId, const at::Tensor &img);
  void moveTrackImagesToIdentityPath(const TrackData &track);
  void saveKeyframes(const TrackData &track);

private:
  CameraPipeline(const rclcpp::NodeOptions &options, int nameIndex, CameraPipelineConfig config);
  void dynamicConfig(Vector2i imageSize);

  void recordTracks(const SysTime time, std::string_view frameId, const std::span<const uint32_t> trackIds,
                    const at::Tensor &patches);
  void publishTrackState(const zoo_msgs::msg::Header &imageHeader, const TKeyframeIndex newKeyframeIndex,
                         const TrackData &track);
  void publishTrackClosed(const zoo_msgs::msg::Header &imageHeader, const TrackData &track);
  void recordMasks(std::string_view videoFile, std::string_view frameId, std::span<TrackId> trackIds,
                   const at::Tensor &masks);

  std::string cameraName_;
  CameraPipelineConfig config_;

  ImageRateLimiter *rateLimiter_;
  RateSampler rateSampler_;

  bool dynamicConfigDone_ = false;
  ImageNormalizer normalizer_;

  CameraCalibration calibration_;

  std::optional<at::cuda::CUDAStream> cudaStream_;
  TrackMatcher trackMatcher_;
  PatchCropper cropper_;
  std::unique_ptr<ISegmenter> segmenter_;
  WorldLocator locator_;
  ImageEmbedder embedder_;
  TrackWriter trackWriter_;
  TrackCountRecorder trackCountRecorder_;

  std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Image12m>> imageSubscriber_;
  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::Detection>> detectionPublisher_;
  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::TrackState>> trackStatePublisher_;
  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::TrackClosed>> trackClosedPublisher_;
};
} // namespace zoo