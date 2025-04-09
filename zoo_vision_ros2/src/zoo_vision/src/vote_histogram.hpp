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

#include "zoo_vision/types.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace zoo {

struct VoteHistogramBest {
  TClassId best;
  float32_t count;
  float32_t firstToSecondRatio;
};

class VoteHistogram {
public:
  VoteHistogram();

  void resize(size_t classCount) { votes_.resize(classCount, 0); }

  void clear();
  void addVote(TClassId classId, float32_t weight);
  void removeVote(TClassId classId, float32_t weight);
  std::span<const float32_t> getVotes() const;

  VoteHistogramBest getHighest() const;

private:
  float32_t dampeningFactor_;
  std::vector<float32_t> votes_;
};
} // namespace zoo
