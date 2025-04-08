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
#include "zoo_vision/timings.hpp"
#include "zoo_vision/track_matcher.hpp"

#include <Eigen/Dense>
#include <c10/cuda/CUDAStream.h>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>
#include <torch/script.h>

#include <filesystem>

namespace zoo {

using float32_t = float;

class Identifier {
public:
  explicit Identifier(int nameIndex, std::string cameraName, TrackMatcher &trackMatcher,
                      at::cuda::CUDAStream cudaStream);

  void readConfig(const nlohmann::json &config);
  void loadModel(const std::filesystem::path &modelPath);

  void onKeyframe(TKeyframeIndex keyframeIndex, const torch::Tensor &patch_f32, TrackData &track);
  void addDetectionInfo(zoo_msgs::msg::Detection &msg, int detectionIndex, const TrackData &track) const;

private:
  const rclcpp::Logger &get_logger() const { return logger_; }

  void callStatefulModel(at::Tensor &logitsGpu, const torch::Tensor &patches, const std::span<const TrackId> trackIds);
  void callStatelessModel(at::Tensor &logitsGpu, const torch::Tensor &patches);

  std::string name_;
  rclcpp::Logger logger_;
  at::cuda::CUDAStream cudaStream_;
  std::string cameraName_;

  TrackMatcher &trackMatcher_;

  bool isStatefulModel_;
  torch::jit::script::Module identityNetwork_;
};
} // namespace zoo