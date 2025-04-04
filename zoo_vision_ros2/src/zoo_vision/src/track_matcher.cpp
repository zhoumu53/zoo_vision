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

#include <ranges>

namespace {
float iou(const Eigen::AlignedBox2f a, const Eigen::AlignedBox2f b) {
  const auto anb = a.intersection(b);
  const auto aub = a.merged(b);
  const auto area = [](const Eigen::AlignedBox2f x) {
    const auto sizes = x.sizes();
    return sizes[0] * sizes[1];
  };
  return area(anb) / area(aub);
}

std::pair<std::pair<Eigen::Index, Eigen::Index>, float> eigen_argmax(Eigen::MatrixXf &m) {
  float maxValue = m(0, 0);
  std::pair<Eigen::Index, Eigen::Index> maxIndex{0, 0};

  for (const Eigen::Index r : std::views::iota(Eigen::Index{0}, m.rows())) {
    for (const Eigen::Index c : std::views::iota(Eigen::Index{0}, m.cols())) {
      const float value = m(r, c);
      if (value > maxValue) {
        maxValue = value;
        maxIndex = {r, c};
      }
    }
  }
  return {maxIndex, maxValue};
}

} // namespace

namespace zoo {

TrackMatcher::TrackMatcher() = default;

TrackData &TrackMatcher::getTrackData(TrackId id) {
  auto it = tracks_.find(id);
  if (it == tracks_.end()) {
    throw std::runtime_error("Track id not found in track matcher");
  }
  return it->second;
}

void TrackMatcher::update(Clock::time_point now, std::span<const Eigen::AlignedBox2f> boxes,
                          std::span<TrackId> outputTrackIds) {
  const size_t inputBoxCount = boxes.size();
  assert(inputBoxCount <= MAX_TRACK_COUNT);

  // Build a mapping index->track iterator
  std::vector<decltype(tracks_)::iterator> trackIts;
  for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
    trackIts.push_back(it);
  }

  // Build score matrix: scores(track_idx, box_idx)
  Eigen::MatrixXf score;
  score.resize(tracks_.size(), inputBoxCount);

  for (const auto &[r, trackIt] : std::views::enumerate(trackIts)) {
    for (const int c : std::views::iota(0uz, inputBoxCount)) {
      score(r, c) = iou(trackIt->second.box, boxes[c]);
    }
  }

  // std::cout << "ScoreMat:\n" << score << std::endl;

  // Init output to invalids
  std::array<bool, MAX_TRACK_COUNT> inputUsed{false};
  std::array<bool, MAX_TRACK_COUNT> trackUsed{false};

  // Greedy matching
  [[maybe_unused]] int matchCount = 0;
  if (tracks_.size() > 0 && inputBoxCount > 0) {
    auto argmax = eigen_argmax(score);
    while (argmax.second > 0) {
      const auto [r, c] = argmax.first;

      TrackData &track = trackIts[r]->second;
      outputTrackIds[c] = track.id;
      track.trackLength += 1;
      track.lastObservation = now;
      track.box = boxes[c];

      inputUsed[c] = true;
      trackUsed[r] = true;
      score.row(r).setConstant(0.0f);
      score.col(c).setConstant(0.0f);
      matchCount += 1;

      argmax = eigen_argmax(score);
    }
  }
  // std::cout << "matchCount=" << matchCount << std::endl;

  // Drop missed tracks
  for (const auto &[r, it] : std::views::enumerate(trackIts)) {
    if (!trackUsed[r]) {
      TrackData &data = it->second;
      auto ellapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - data.lastObservation);
      data.skippedObservationCount += 1;
      if (ellapsedTime > MAX_INACTIVE_DURATION) {
        // Raise event that a track has been closed
        onTrackCloseEvent(it->second.id);

        // Mark as invalid to delete in one go after this loop
        it->second.id = INVALID_TRACK_ID;
      }
    }
  }
  std::erase_if(tracks_, [](const std::pair<TrackId, TrackData> &item) { return item.second.id == INVALID_TRACK_ID; });

  // Create new tracks
  for (const int c : std::views::iota(0uz, inputBoxCount)) {
    if (inputUsed[c]) {
      continue;
    }
    const auto newTrackId = nextTrackId_;
    nextTrackId_ += 1;

    outputTrackIds[c] = newTrackId;
    tracks_.insert({newTrackId, TrackData{
                                    newTrackId,
                                    now,
                                    boxes[c],
                                }});
  }

  // std::cout << "Box count: " << boxes.size() << " centers=";
  // for (auto [box, id] : std::views::zip(boxes, outputTrackIds)) {
  //   std::cout << "T" << id << "=(" << box.center().transpose() << "), ";
  // }
  // std::cout << std::endl;
}
} // namespace zoo
