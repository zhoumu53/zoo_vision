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

#include "zoo_vision/track_writer.hpp"
#include "zoo_vision/utils.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <mutex>

namespace zoo {
namespace {
template <typename T> std::string formatTime(const T time) {
  const auto timeS = secondsTimePointFromTimePoint(time);
  return std::format("{0:%Y-%m-%d} {0:%T}", timeS);
}

std::filesystem::path getTrackPath(const std::filesystem::path &rootPath, TrackData::time_point startTime, TrackId id) {
  return rootPath / std::format("{:%Y-%m-%d}", startTime) / std::format("{:06d}.", id);
}

nlohmann::json createTrackInfo(const TrackData &track, std::optional<SysTime> endTime) {
  nlohmann::json j;
  j["id"] = track.id;
  j["start_time"] = formatTime(track.startTime);
  j["length"] = track.trackLength;
  if (endTime.has_value()) {
    j["end_time"] = formatTime(endTime.value());
  } else {
    j["end_time"] = nullptr;
  }

  nlohmann::json boxes;
  for (const auto &box : track.boxHistory) {
    boxes.push_back({box.min()[0], box.min()[1], box.max()[0], box.max()[1]});
  }
  j["boxes_tlbr"] = boxes;

  return j;
}

void writeInfo(const TrackData &track, std::optional<SysTime> endTime, std::filesystem::path trackPath) {
  const std::filesystem::path infoPath = std::filesystem::path(trackPath).replace_extension(".json");
  const auto info = createTrackInfo(track, endTime);
  std::ofstream infoFd(infoPath);
  infoFd << info.dump(/*indent*/ 2);
}
} // namespace

TrackWriter::TrackWriter(std::filesystem::path rootPath) : rootPath_{rootPath} {}

void TrackWriter::startVideo(TrackData &track, const at::Tensor &cropImage) {
  const std::filesystem::path trackPath = getTrackPath(rootPath_, track.startTime, track.id);
  std::filesystem::create_directories(trackPath.parent_path());

  // Write json at start
  writeInfo(track, std::nullopt, trackPath);

  // Write video
  const std::filesystem::path videoPath = std::filesystem::path(trackPath).replace_extension(".mkv");
  const Vector2i cropSize = {static_cast<int>(cropImage.size(1)), static_cast<int>(cropImage.size(0))};

  if (!track.trackVideo.open(videoPath.string(), cropSize)) {
    throw std::runtime_error(std::format("Could not create track video ({})", videoPath.string()));
  }

  addFrame(track, cropImage);
}

void TrackWriter::addFrame(TrackData &track, const at::Tensor &cropImage) {

  cv::Mat3b cropCv = wrapCvFromTensor3b(cropImage);
  // cv::imwrite((rootPath_ / "test.png").string(), cropCv);
  track.trackVideo.write(cropCv);

  if (track.trackLength % 100 == 0) {
    // Write info every 100 frames
    const std::filesystem::path trackPath = getTrackPath(rootPath_, track.startTime, track.id);
    writeInfo(track, std::nullopt, trackPath);
  }
}

void TrackWriter::close(const TrackData &track, SysTime time) {
  // Write json at end
  const std::filesystem::path trackPath = getTrackPath(rootPath_, track.startTime, track.id);
  writeInfo(track, time, trackPath);
}
} // namespace zoo