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

#include "zoo_vision/utils.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace zoo {

class VoteHistogram {
public:
  using TClassId = int;

  VoteHistogram();

  void resize(size_t classCount) { votes_.resize(classCount, 0); }

  void clear();
  void addVote(TClassId classId);
  std::span<const float32_t> getVotes() const;

  std::pair<TClassId, float32_t> getHighest() const;

private:
  float32_t dampeningFactor_;
  std::vector<float32_t> votes_;
};
} // namespace zoo
