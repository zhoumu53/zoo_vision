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
#include "zoo_vision/track_writer.hpp"
#include "zoo_vision/types.hpp"
#include "zoo_vision/vote_histogram.hpp"

#include <ATen/Tensor.h>
#include <ByteTrack/BYTETracker.h>
#include <Eigen/Dense>

#include <chrono>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace zoo {
namespace {
template <class T> AlignedBox<T, 2> eigenFromByteTrack(const byte_track::Rect<T> &r) {
  return AlignedBox<T, 2>(Vector<T, 2>{r.tl_x(), r.tl_y()}, Vector<T, 2>{r.br_x(), r.br_y()});
}
template <class T> byte_track::Rect<T> byteTrackFromEigen(const AlignedBox<T, 2> &r) {
  return byte_track::Rect<T>(r.min().x(), r.min().y(), r.sizes()[0], r.sizes()[1]);
}

} // namespace
struct TrackData {
  using time_point = std::chrono::system_clock::time_point;

  std::shared_ptr<byte_track::STrack> byteTrack;

  TrackId id;
  time_point startTime;

  time_point lastObservation;
  int skippedObservationCount = 0;

  size_t trackLength = 1;
  AlignedBox2f box;
  float32_t score;
  std::optional<at::Tensor> identityState = std::nullopt;

  KeyframeStore keyframeStore;
  std::vector<at::Tensor> identityProbsByKeyframe;
  VoteHistogram identityHistogram;
  TIdentity selectedIdentity = INVALID_IDENTITY;
  TBehaviour selectedBehaviour = INVALID_BEHAVIOUR;

  TrackWriter writer;

  std::vector<time_point> timestampHistory;
  std::vector<AlignedBox2f> boxHistory;
  std::vector<float32_t> scoreHistory;

  TrackData(std::shared_ptr<byte_track::STrack> byteTrack_, time_point startTime_,
            const std::filesystem::path &rootTracksPath)
      : byteTrack{std::move(byteTrack_)}, id{static_cast<TrackId>(byteTrack->getTrackId())}, startTime{startTime_},
        lastObservation{startTime_}, box{eigenFromByteTrack(byteTrack->getRect())}, score{byteTrack->getScore()},
        writer{rootTracksPath, *this} {
    timestampHistory.push_back(startTime_);
    boxHistory.push_back(box);
    scoreHistory.push_back(score);
  }

  void update(time_point now) {
    timestampHistory.push_back(now);
    boxHistory.push_back(box);
    scoreHistory.push_back(score);

    trackLength += 1;
    lastObservation = now;
    skippedObservationCount = 0;
    box = eigenFromByteTrack(byteTrack->getRect());
    score = byteTrack->getScore();
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

  TrackMatcher(const std::filesystem::path &rootTracksPath);

  TrackUpdateStats update(Clock::time_point now, std::span<const AlignedBox2f> boxes, std::span<const float32_t> scores,
                          std::span<TrackId> outputTrackIds);

  TrackData &getTrackData(TrackId id);

  std::unordered_map<TrackId, std::unique_ptr<TrackData>>::const_iterator begin() const { return tracks_.begin(); }
  std::unordered_map<TrackId, std::unique_ptr<TrackData>>::const_iterator end() const { return tracks_.end(); }

private:
  std::filesystem::path rootCameraPath_;
  byte_track::BYTETracker byteTracker_;
  std::unordered_map<TrackId, std::unique_ptr<TrackData>> tracks_;
};
} // namespace zoo