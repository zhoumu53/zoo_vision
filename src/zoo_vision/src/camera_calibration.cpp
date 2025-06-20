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
#include "zoo_vision/camera_calibration.hpp"

#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/utils.hpp"

#include <nlohmann/json.hpp>

namespace zoo {

CameraCalibration::CameraCalibration(std::string cameraName) : cameraName_{std::move(cameraName)} {
  readConfig(getConfig());
}

void CameraCalibration::readConfig(const nlohmann::json &config) {
  calibratedCameraSize_ = Eigen::Vector2i{config["cameras"][cameraName_]["intrinsics"]["width"].get<int>(),
                                          config["cameras"][cameraName_]["intrinsics"]["height"].get<int>()};
  H_mapFromWorld2_ = config["map"]["T_map_from_world2"];
  H_world2FromCamera_ = config["cameras"][cameraName_]["H_world2_from_camera"];

  // Mask polyogons
  for (const auto &polyJson : config["cameras"][cameraName_]["mask_polygons"]) {
    Polygon poly;
    for (const auto &pointJson : polyJson) {
      const auto p = Vector2i(pointJson[0].get<int>(), pointJson[1].get<int>());
      poly.push_back(p);
    }
    calibratedMaskPolygons_.push_back(std::move(poly));
  }
}

void CameraCalibration::setImageSize(Vector2i size) {
  imageSize_ = size;

  scaleX_calibratedFromImage_ = static_cast<float32_t>(calibratedCameraSize_.x()) / imageSize_.x();
  scaleY_calibratedFromImage_ = static_cast<float32_t>(calibratedCameraSize_.y()) / imageSize_.y();

  // Scale mask
  maskPolygons_.clear();
  for (const auto &poly : calibratedMaskPolygons_) {
    Polygon scaledPoly;
    for (const auto &point : poly) {
      scaledPoly.push_back(Vector2i(point.x() / scaleX_calibratedFromImage_, point.y() / scaleY_calibratedFromImage_));
    }
    maskPolygons_.push_back(std::move(scaledPoly));
  }
}

} // namespace zoo
