
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

#include "zoo_vision/timings.hpp"
#include "zoo_vision/track_matcher.hpp"
#include "zoo_vision/types.hpp"

#include <ATen/Tensor.h>
#include <Eigen/Dense>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace zoo {

class TrackWriter {
public:
  TrackWriter(std::filesystem::path rootPath);

  void startVideo(TrackData &track, const at::Tensor &cropImage);
  void addFrame(TrackData &track, const at::Tensor &cropImage);
  void close(const TrackData &track, SysTime time);

private:
  std::filesystem::path rootPath_;
};

} // namespace zoo