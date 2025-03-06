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
#include "zoo_vision/identifier.hpp"
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

class Segmenter : public rclcpp::Node {
public:
  explicit Segmenter(const rclcpp::NodeOptions &options = rclcpp::NodeOptions(), int nameIndex = 999);

  void readConfig(const nlohmann::json &config);
  void loadModel(const std::filesystem::path &modelPath);
  void onImage(std::shared_ptr<const zoo_msgs::msg::Image12m> msg);

private:
  std::string cameraName_;
  at::cuda::CUDAStream cudaStream_;
  RateSampler rateSampler_;
  Eigen::Matrix3f H_world2FromCamera_;
  Eigen::Matrix3f H_mapFromWorld2_;

  int elephant_label_id_;
  torch::jit::script::Module model_;

  TrackMatcher trackMatcher_;

  // TODO: Move this to an independent node
  std::shared_ptr<Identifier> identifier_;

  std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Image12m>> imageSubscriber_;

  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::Detection>> detectionPublisher_;
};
} // namespace zoo