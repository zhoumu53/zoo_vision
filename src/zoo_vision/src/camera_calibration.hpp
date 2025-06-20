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

#include <nlohmann/json_fwd.hpp>

namespace zoo {

class CameraCalibration {
public:
  explicit CameraCalibration(std::string cameraName);

  void readConfig(const nlohmann::json &config);

  void setImageSize(Vector2i size);

  // Members are public
  std::string cameraName_;

  // The original calibrated values
  Vector2i calibratedCameraSize_;
  Matrix3f H_world2FromCamera_;
  Matrix3f H_mapFromWorld2_;

  std::vector<Polygon> calibratedMaskPolygons_;

  // Scalings to match the calibration to the real image size received
  Vector2i imageSize_;
  float32_t scaleX_calibratedFromImage_;
  float32_t scaleY_calibratedFromImage_;

  std::vector<Polygon> maskPolygons_;
};
} // namespace zoo
