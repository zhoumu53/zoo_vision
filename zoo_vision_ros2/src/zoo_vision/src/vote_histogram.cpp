
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
#include <algorithm>
#include <numeric> // std::iota
#include <ranges>

namespace zoo {
namespace {

template <typename T> std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

} // namespace
VoteHistogram::VoteHistogram() : dampeningFactor_{0.99f} {
  resize(5); // TODO: get this from somewhere meaningful
}

void VoteHistogram::clear() {
  for (auto &count : votes_) {
    count = 0;
  }
}
void VoteHistogram::addVote(TClassId classId) {
  // if (classId >= votes_.size()) {
  //   throw std::runtime_error("Class outside of histogram range");
  // }
  // for (auto &count : votes_) {
  //   count *= dampeningFactor_;
  // }
  votes_[classId] += 1;
}
void VoteHistogram::removeVote(TClassId classId) { votes_[classId] -= 1; }

std::span<const float32_t> VoteHistogram::getVotes() const { return votes_; }

VoteHistogramBest VoteHistogram::getHighest() const {
  if (votes_.size() < 2) {
    return {0, 0.0f, 0.0f};
  }
  const auto sortedIndices = sort_indexes(votes_);
  const TClassId bestClass = sortedIndices[0];
  const TClassId secondClass = sortedIndices[1];
  const float32_t bestScore = votes_[bestClass];
  const float32_t secondScore = votes_[secondClass];

  return {bestClass, bestScore, bestScore / (bestScore + secondScore)};
}

} // namespace zoo
