
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

#include "zoo_vision/vote_histogram.hpp"
#include <ranges>

namespace zoo {

VoteHistogram::VoteHistogram() : dampeningFactor_{0.99f} {}

void VoteHistogram::clear() {
  for (auto &count : votes_) {
    count = 0;
  }
}
void VoteHistogram::addVote(int classId) {
  // if (classId >= votes_.size()) {
  //   throw std::runtime_error("Class outside of histogram range");
  // }
  for (auto &count : votes_) {
    count *= dampeningFactor_;
  }
  votes_[classId] += 1;
}

std::span<const float32_t> VoteHistogram::getVotes() const { return votes_; }

auto VoteHistogram::getHighest() const -> std::pair<TClassId, float32_t> {
  float32_t bestCount = 0;
  TClassId bestClass = 0;
  for (auto [classId, count] : std::views::enumerate(votes_)) {
    if (count > bestCount) {
      bestCount = count;
      bestClass = classId;
    }
  }
  return {bestClass, bestCount};
}

} // namespace zoo
