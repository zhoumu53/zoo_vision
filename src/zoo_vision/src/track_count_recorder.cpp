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

#include "zoo_vision/track_count_recorder.hpp"

#include "zoo_vision/utils.hpp"

#include <nlohmann/json.hpp>

#include <fstream>

namespace zoo {
namespace {
template <typename T> std::string formatTime(const T time) {
  const auto timeS = secondsTimePointFromTimePoint(time);
  return std::format("{0:%Y-%m-%d} {0:%T}", timeS);
}
} // namespace

TrackCountRecorder::TrackCountRecorder(std::string_view cameraName) : cameraName_{cameraName} {

  const auto &config = getConfig();
  const std::filesystem::path rootPath = config["record_root"].get<std::string>();
  const std::filesystem::path path = rootPath / "tracks" / cameraName / "empty.csv";
  std::filesystem::create_directories(path.parent_path());

  if (std::filesystem::exists(path)) {
    fd_.open(path, std::ios_base::app);
  } else {
    // Write header
    fd_.open(path);
    fd_ << "start,end,track_count" << std::endl;
  }
  CHECK_TRUE(!fd_.fail());
}

TrackCountRecorder::~TrackCountRecorder() {
  if (startTime_.has_value() && lastSeenTime_.has_value()) {
    writeCount(*lastSeenTime_);
  }
}

void TrackCountRecorder::recordCount(SysTime time, size_t count) {
  lastSeenTime_ = time;

  if (!startTime_.has_value()) {
    startTime_ = time;
    count_ = count;
    return;
  }

  if (count_ == count) {
    // Same count as before, nothing to do
    return;
  }

  // Count has changed, record
  writeCount(time);
  startTime_ = time;
  count_ = count;
}

void TrackCountRecorder::writeCount(SysTime endTime) {
  fd_ << formatTime(*startTime_) << "," << formatTime(endTime) << "," << count_ << std::endl;
}

} // namespace zoo
