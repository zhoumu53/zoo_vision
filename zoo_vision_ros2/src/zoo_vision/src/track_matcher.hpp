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

#include <ATen/Tensor.h>
#include <Eigen/Dense>
#include <span>
#include <unordered_map>

namespace zoo {

using float32_t = float;

using TrackId = uint32_t;

struct TrackData {
  TrackId id;
  Eigen::AlignedBox2f box;
  std::optional<at::Tensor> identityState;
};

class TrackMatcher {
public:
  static constexpr TrackId INVALID_TRACK_ID = 0;
  static constexpr size_t MAX_TRACK_COUNT = 15;

  TrackMatcher();

  void update(std::span<const Eigen::AlignedBox2f> boxes, std::span<TrackId> outputTrackIds);

  TrackData *getTrackData(TrackId id);

private:
  TrackId nextTrackId_ = 1;
  std::unordered_map<TrackId, TrackData> tracks_;
};
} // namespace zoo