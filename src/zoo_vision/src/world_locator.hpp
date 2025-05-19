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

#include "zoo_vision/types.hpp"

#include <Eigen/Dense>
#include <nlohmann/json_fwd.hpp>
#include <rclcpp/rclcpp.hpp>

#include <filesystem>

namespace zoo {

class WorldLocator {
public:
  explicit WorldLocator(int nameIndex, std::string cameraName);

  void readConfig(const nlohmann::json &config);
  void setDetectionImageSize(Eigen::Vector2i size) { detectionImageSize_ = size; }

  void worldFromBboxes(Eigen::Ref<Matrix3Xf> positionsInWorld, std::span<const AlignedBox2f> bboxesInDetection) const;
  Eigen::Vector3f worldFromBbox(const AlignedBox2f &bboxInDetection) const;

private:
  const rclcpp::Logger &get_logger() const { return logger_; }

  std::string name_;
  rclcpp::Logger logger_;

  std::string cameraName_;

  Vector2i detectionImageSize_;

  Vector2i calibratedCameraSize_;
  Matrix3f H_world2FromCamera_;
  Matrix3f H_mapFromWorld2_;
};
} // namespace zoo
