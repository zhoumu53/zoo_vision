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

#include "zoo_vision/track_matcher.hpp"
#include "zoo_vision/utils.hpp"

#include <nlohmann/json.hpp>

#include <ranges>

namespace zoo {

TrackMatcher::TrackMatcher(const std::filesystem::path &rootCameraPath) : rootCameraPath_{rootCameraPath} {
  const auto config = getConfig();
  recordTracks_ = config["record_tracks"].get<bool>();
}

TrackData &TrackMatcher::getTrackData(TrackId id) {
  auto it = tracks_.find(id);
  if (it == tracks_.end()) {
    throw ZooVisionError("Track id not found in track matcher");
  }
  return *it->second;
}

TrackUpdateStats TrackMatcher::update(Clock::time_point now, float32_t fps, std::span<const AlignedBox2f> boxes,
                                      std::span<const float32_t> scores, std::span<TrackId> outputTrackIds) {
  TrackUpdateStats result;

  // Convert to ByteTrack input
  std::vector<byte_track::Object> inputs;
  for (const auto &[box, score] : std::views::zip(boxes, scores)) {
    inputs.emplace_back(byteTrackFromEigen(box), 1, score);
  }

  // Run bytetrack
  auto outputs = byteTracker_.update(inputs);

  // Delete tracks that were dropped
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    if (it->second->byteTrack->getSTrackState() == byte_track::STrackState::Removed) {
      result.closedTracks.push_back(std::move(it->second));
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }
  // Add new tracks
  for (std::shared_ptr<byte_track::STrack> &output : outputs) {
    if (output->getSTrackState() == byte_track::STrackState::Removed) {
      // Ignore removed tracks (in case they are returned here)
      continue;
    }

    auto it = tracks_.find(output->getTrackId());
    if (it == tracks_.end()) {
      // Not present in our table, add
      auto newTrack = std::make_unique<TrackData>(std::move(output), now);
      if (recordTracks_) {
        newTrack->writer.emplace(rootCameraPath_, *newTrack, fps);
      }
      result.newTracks.push_back(newTrack.get());
      tracks_.insert({newTrack->id, std::move(newTrack)});
    } else {
      TrackData &track = *it->second;
      track.update(now);
    }
  }

  // ByteTrack does not do the matching between detection index and track index.
  // We need to do it ourselves.
  for (auto &&[box, outputTrackId] : std::views::zip(boxes, outputTrackIds)) {
    float32_t bestDistance = 1e10;
    TrackId bestId = INVALID_TRACK_ID;

    for (const auto &[trackId, data] : tracks_) {
      const float32_t distance = (data->box.center() - box.center()).squaredNorm();
      if (distance <= bestDistance) {
        bestDistance = distance;
        bestId = trackId;
      }
    }
    outputTrackId = bestId;
  }
  return result;
}
} // namespace zoo
