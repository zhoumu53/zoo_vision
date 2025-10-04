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
#include "zoo_vision/behaviourer.hpp"
#include "zoo_vision/camera_calibration.hpp"
#include "zoo_vision/identifier_interface.hpp"
#include "zoo_vision/image_embedder.hpp"
#include "zoo_vision/image_normalizer.hpp"
#include "zoo_vision/image_quality.hpp"
#include "zoo_vision/patch_cropper.hpp"
#include "zoo_vision/segmenter_interface.hpp"
#include "zoo_vision/timings.hpp"
#include "zoo_vision/track_matcher.hpp"
#include "zoo_vision/world_locator.hpp"

#include <Eigen/Dense>
#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/mat.hpp>
#include <rclcpp/rclcpp.hpp>

#include <filesystem>

namespace zoo {

class CameraPipeline : public rclcpp::Node {
public:
  explicit CameraPipeline(const rclcpp::NodeOptions &options = rclcpp::NodeOptions(), int nameIndex = 999);

  void readConfig(const nlohmann::json &config);

  void onImage(std::shared_ptr<zoo_msgs::msg::Image12m> msg);

  void saveImageToImproveDetection(const SysTime time, const cv::Mat3b &cvImg);
  void saveImageToImproveBehaviour(const SysTime time, TBehaviour behaviourId, const at::Tensor &img);
  void moveTrackImagesToIdentityPath(const TrackData &track);
  void saveKeyframes(const TrackData &track);

private:
  void dynamicConfig(Vector2i imageSize);

  void recordTracks(const SysTime time, const std::span<const uint32_t> trackIds, const at::Tensor &patches);
  void publishTrackState(const zoo_msgs::msg::Header &imageHeader, const TKeyframeIndex newKeyframeIndex,
                         const TrackData &track);
  void publishTrackClosed(const zoo_msgs::msg::Header &imageHeader, const TrackData &track);
  std::string cameraName_;

  RateSampler rateSampler_;

  bool recordDetectionLoss_;
  bool recordTracks_;
  bool recordKeyframes_;
  bool recordBehaviourChange_;

  bool dynamicConfigDone_ = false;
  Vector2i detectionImageSize_{0, 0};
  ImageNormalizer normalizer_;

  CameraCalibration calibration_;

  at::DeviceType device_ = at::kCPU;
  std::optional<at::cuda::CUDAStream> cudaStream_;
  TrackMatcher trackMatcher_;
  PatchCropper cropper_;
  std::unique_ptr<ISegmenter> segmenter_;
  WorldLocator locator_;
  ImageEmbedder embedder_;
  ImageQualityNet quality_;
  std::unique_ptr<IIdentifier> identifier_;
  Behaviourer behaviourer_;

  std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Image12m>> imageSubscriber_;
  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::Detection>> detectionPublisher_;
  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::TrackState>> trackStatePublisher_;
  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::TrackClosed>> trackClosedPublisher_;

  std::filesystem::path rootPathImprove_;
};
} // namespace zoo