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

#include "zoo_vision/keyframe_store.hpp"
#include "zoo_vision/types.hpp"
#include "zoo_vision/video_writer.hpp"
#include "zoo_vision/vote_histogram.hpp"

#include <ATen/Tensor.h>
#include <Eigen/Dense>

#include <chrono>
#include <fstream>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace zoo {

struct TrackData {
  using time_point = std::chrono::system_clock::time_point;

  TrackId id;
  time_point startTime;

  time_point lastObservation;
  int skippedObservationCount = 0;

  size_t trackLength = 1;
  AlignedBox2f box;
  std::optional<at::Tensor> identityState = std::nullopt;

  KeyframeStore keyframeStore;
  std::vector<at::Tensor> identityProbsByKeyframe;
  VoteHistogram identityHistogram;
  TIdentity selectedIdentity = INVALID_IDENTITY;
  TBehaviour selectedBehaviour = INVALID_BEHAVIOUR;

  std::ofstream infoFd;
  VideoWriter trackVideo;

  std::vector<time_point> timestampHistory;
  std::vector<AlignedBox2f> boxHistory;
  std::vector<float32_t> confidenceHistory;

  TrackData(TrackId id_, time_point startTime_, AlignedBox2f box_)
      : id{id_}, startTime{startTime_}, lastObservation{startTime_}, box{box_} {
    timestampHistory.push_back(startTime_);
    boxHistory.push_back(box_);
  }

  void update(time_point now, AlignedBox2f newBox) {
    timestampHistory.push_back(now);
    boxHistory.push_back(box);

    trackLength += 1;
    lastObservation = now;
    skippedObservationCount = 0;
    box = newBox;
  }
};

struct TrackUpdateStats {
  std::vector<std::unique_ptr<TrackData>> closedTracks;
  std::vector<TrackData *> newTracks;
  std::vector<TrackData *> justMissedTracks;
};

class TrackMatcher {
public:
  using Clock = std::chrono::system_clock;

  static constexpr TrackId INVALID_TRACK_ID = 0;
  static constexpr size_t MAX_TRACK_COUNT = 25;
  static constexpr auto MAX_INACTIVE_DURATION = std::chrono::milliseconds{3000};

  TrackMatcher();

  TrackUpdateStats update(Clock::time_point now, std::span<const Eigen::AlignedBox2f> boxes,
                          std::span<TrackId> outputTrackIds);

  TrackData &getTrackData(TrackId id);

  std::unordered_map<TrackId, std::unique_ptr<TrackData>>::const_iterator begin() const { return tracks_.begin(); }
  std::unordered_map<TrackId, std::unique_ptr<TrackData>>::const_iterator end() const { return tracks_.end(); }

private:
  TrackId nextTrackId_ = 1;
  std::unordered_map<TrackId, std::unique_ptr<TrackData>> tracks_;
};
} // namespace zoo