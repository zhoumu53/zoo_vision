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
  const auto timeS = secondsTimePointFromTimePoint(startTime);
  return rootPath / std::format("{:%Y-%m-%d}", startTime) / (std::format("T{:%H%M%S}_ID{:06d}.", timeS, id));
}

void writeHeader(std::ofstream &fd) {
  fd << "frame_id,timestamp,bbox_top,bbox_left,bbox_bottom,bbox_right,score" << std::endl;
}

void writeRow(std::ofstream &fd, const TrackData &track, std::string_view frameId) {
  CHECK_TRUE(!track.timestampHistory.empty());
  CHECK_EQ(track.timestampHistory.size(), track.boxHistory.size());
  CHECK_EQ(track.timestampHistory.size(), track.scoreHistory.size());

  const auto time = track.timestampHistory.back();
  const auto bbox = track.boxHistory.back();
  const float32_t score = track.scoreHistory.back();

  fd << frameId << "," << std::format("{0:%Y-%m-%d} {0:%T}", time) << "," << bbox.min()[0] << "," << bbox.min()[1]
     << "," << bbox.max()[0] << "," << bbox.max()[1] << "," << score << std::endl;
}

} // namespace

TrackWriter::TrackWriter(std::filesystem::path rootPath) : rootPath_{rootPath} {}

void TrackWriter::writeFrame(TrackData &track, std::string_view frameId, const at::Tensor &cropImage) {
  if (track.trackLength == 1) {
    const std::filesystem::path trackPath = getTrackPath(rootPath_, track.startTime, track.id);
    std::filesystem::create_directories(trackPath.parent_path());

    // Write csv
    const std::filesystem::path infoPath = std::filesystem::path(trackPath).replace_extension(".csv");
    if (std::filesystem::exists(infoPath)) {
      std::cout << "Warning: track info file already exists for new track (" << infoPath << ")" << std::endl;
    }
    track.infoFd.open(infoPath);
    if (track.infoFd.fail()) {
      throw ZooVisionError(
          std::format("Could not write the track data file to {}, errno: {}", infoPath.string(), strerror(errno)));
    }
    writeHeader(track.infoFd);

    // Start video
    const std::filesystem::path videoPath = std::filesystem::path(trackPath).replace_extension(".mkv");
    const Vector2i cropSize = {static_cast<int>(cropImage.size(1)), static_cast<int>(cropImage.size(0))};

    if (!track.trackVideo.open(videoPath.string(), cropSize)) {
      throw std::runtime_error(std::format("Could not create track video ({})", videoPath.string()));
    }
  }

  writeRow(track.infoFd, track, frameId);

  cv::Mat3b cropCv = wrapCvFromTensor3b(cropImage);
  // cv::imwrite((rootPath_ / "test.png").string(), cropCv);
  track.trackVideo.write(cropCv);
}

void TrackWriter::close(const TrackData &track, SysTime time) {
  (void)track;
  (void)time;
}
} // namespace zoo