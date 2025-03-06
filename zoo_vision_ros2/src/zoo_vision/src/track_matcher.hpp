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

#include <Eigen/Dense>
#include <span>
#include <unordered_map>

namespace zoo {

using float32_t = float;

class TrackMatcher {
public:
  using TrackId = uint32_t;
  static constexpr TrackId INVALID_TRACK_ID = 0;
  static constexpr size_t MAX_TRACK_COUNT = 15;

  TrackMatcher();

  void update(std::span<const Eigen::AlignedBox2f> boxes, std::span<TrackId> outputTrackIds);

private:
  TrackId nextTrackId_ = 1;
  size_t validTrackCount_ = 0;
  std::array<std::pair<TrackId, Eigen::AlignedBox2f>, MAX_TRACK_COUNT> tracks_;
};
} // namespace zoo