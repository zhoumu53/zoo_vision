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

#include "zoo_vision/world_locator.hpp"

#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/utils.hpp"

#include <ATen/core/List.h>
#include <opencv2/core.hpp>
#include <rclcpp/time.hpp>

namespace zoo {

WorldLocator::WorldLocator(int nameIndex, std::string cameraName)
    : name_{std::format("world_locator_{}", nameIndex)}, logger_{rclcpp::get_logger(name_)}, cameraName_{cameraName} {
  at::InferenceMode inferenceGuard;
  readConfig(getConfig());
}

void WorldLocator::readConfig(const nlohmann::json &config) {
  // Camera calibration
  calibratedCameraSize_ = Eigen::Vector2i{config["cameras"][cameraName_]["intrinsics"]["width"].get<int>(),
                                          config["cameras"][cameraName_]["intrinsics"]["height"].get<int>()};
  H_mapFromWorld2_ = config["map"]["T_map_from_world2"];
  H_world2FromCamera_ = config["cameras"][cameraName_]["H_world2_from_camera"];
}

void WorldLocator::worldFromBboxes(Eigen::Ref<MatrixX3f> positionsInWorld,
                                   std::span<const AlignedBox2f> bboxesInDetection) const {
  for (auto i : std::views::iota(0uz, bboxesInDetection.size())) {
    // Copy results to detectionMsg
    positionsInWorld.col(i) = worldFromBbox(bboxesInDetection[i]);
  }
}

Eigen::Vector3f WorldLocator::worldFromBbox(const Eigen::AlignedBox2f &bboxInDetection) const {
  const float32_t scaleX_calibratedFromDetection =
      static_cast<float32_t>(calibratedCameraSize_.x()) / detectionImageSize_.x();
  const float32_t scaleY_calibratedFromDetection =
      static_cast<float32_t>(calibratedCameraSize_.y()) / detectionImageSize_.y();
  auto scalePoint = [&](Eigen::Vector2f p) {
    return Eigen::Vector2f{p.x() * scaleX_calibratedFromDetection, p.y() * scaleY_calibratedFromDetection};
  };
  const Eigen::AlignedBox2f bboxInCalibrated = {scalePoint(bboxInDetection.min()), scalePoint(bboxInDetection.max())};

  const auto world2FromImage = [&](Eigen::Vector2f p) { return (H_world2FromCamera_ * p.homogeneous()).hnormalized(); };
  const auto worldFromWorld2 = [](const Eigen::Vector2f &x2) { return Eigen::Vector3f{x2[0], x2[1], 0.0f}; };

  // Here we do a small trick. The initial imagePosition is at the middle-bottom of the bounding box.
  // But animals are 3D so we try to guess the center by assuming the animal is a rectangle of X-Y-Z dimensions.
  // Observing the animal from the front gives an aspect ratio of Ax=X/Z, whereas from the side the aspect is Ay=Y/Z.
  // For aspect ratio Ax we want to add an offset of Y/2 in the world plane. For aspect Ay we want to add X/2.
  // So we do a linear interpolation between Ax and Ay to find the offset to apply.
  constexpr float32_t aspectMin = 0.5f;
  constexpr float32_t aspectMax = 1.0f;
  constexpr float32_t offsetMin = 1.5f;
  constexpr float32_t offsetMax = 0.5f;

  const float32_t aspect = bboxInCalibrated.sizes()[0] / bboxInCalibrated.sizes()[1];
  float32_t offset;
  if (aspect <= aspectMin) {
    offset = offsetMin;
  } else if (aspect >= aspectMax) {
    offset = offsetMax;
  } else {
    offset = (aspect - aspectMin) / (aspectMax - aspectMin) * (offsetMax - offsetMin) + offsetMin;
  }

  const Eigen::Vector2f bboxMiddleInImage = Eigen::Vector2f{bboxInCalibrated.center()[0], bboxInCalibrated.max()[1]};
  const Eigen::Vector2f bboxMiddleInWorld2 = world2FromImage(bboxMiddleInImage);

  const Eigen::Vector2f deltaInImage = bboxMiddleInImage + Eigen::Vector2f{0.f, -1.f};
  const Eigen::Vector2f deltaInWorld2 = world2FromImage(deltaInImage);
  const Eigen::Vector2f offsetDirection = (deltaInWorld2 - bboxMiddleInWorld2).normalized();

  const Eigen::Vector2f centerInWorld2 = bboxMiddleInWorld2 + offsetDirection * offset;

  return worldFromWorld2(centerInWorld2);
}
} // namespace zoo
