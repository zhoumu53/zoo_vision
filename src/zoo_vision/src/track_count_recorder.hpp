
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

#include "zoo_vision/timings.hpp"
#include "zoo_vision/types.hpp"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>

namespace zoo {
class TrackCountRecorder {
public:
  TrackCountRecorder(std::string_view cameraName);
  ~TrackCountRecorder();
  void recordCount(SysTime time, size_t count);

private:
  std::string cameraName_;
  std::ofstream fd_;

  std::optional<SysTime> lastSeenTime_;
  std::optional<SysTime> startTime_;
  size_t count_;

  void writeCount(SysTime endTime);
};
} // namespace zoo