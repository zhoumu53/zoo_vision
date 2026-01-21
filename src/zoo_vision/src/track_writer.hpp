
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
#include "zoo_vision/types.hpp"
#include "zoo_vision/video_writer.hpp"

#include <ATen/Tensor.h>
#include <Eigen/Dense>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace zoo {

struct TrackData;

class TrackWriter {
public:
  TrackWriter(const std::filesystem::path &rootTracksPath, TrackData &track, float32_t fps);

  void writeFrame(uint64_t frameId, const at::Tensor &cropImage, const Vector3f &worldPosition,
                  const Vector2i imageSize);
  void close(SysTime time);

private:
  TrackData &track_;
  std::ofstream infoFd_;
  VideoWriter trackVideo_;
};

} // namespace zoo