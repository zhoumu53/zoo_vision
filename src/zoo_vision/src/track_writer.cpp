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
#include "zoo_vision/patch_cropper.hpp"
#include "zoo_vision/track_matcher.hpp"
#include "zoo_vision/utils.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <mutex>
#include <thread>

namespace zoo {
namespace {
template <typename T> std::string formatTime(const T time) {
  const auto timeS = secondsTimePointFromTimePoint(time);
  return std::format("{0:%Y-%m-%d} {0:%T}", timeS);
}

std::filesystem::path getTrackPath(const std::filesystem::path &rootTracksPath, TrackData::time_point startTime,
                                   TrackId id, bool temp) {
  const auto timeS = secondsTimePointFromTimePoint(startTime);

  std::string name;
  if (temp) {
    constexpr auto formatStr = "part_T{:%H%M%S}_ID{:06d}.";
    name = std::format(formatStr, timeS, id);
  } else {
    constexpr auto formatStr = "T{:%H%M%S}_ID{:06d}.";
    name = std::format(formatStr, timeS, id);
  }
  return rootTracksPath / std::format("{:%Y-%m-%d}", startTime) / name;
}

void writeHeader(std::ofstream &fd) {
  fd << "frame_id,timestamp,bbox_top2,bbox_left2,bbox_bottom2,bbox_right2,score,world_x,world_y" << std::endl;
}

void writeRow(std::ofstream &fd, const TrackData &track, uint64_t frameId, const Vector3f &worldPosition,
              const Vector2i imageSize) {
  CHECK_TRUE(!track.timestampHistory.empty());
  CHECK_EQ(track.timestampHistory.size(), track.boxHistory.size());
  CHECK_EQ(track.timestampHistory.size(), track.scoreHistory.size());

  const auto time = track.timestampHistory.back();
  const auto bbox = track.boxHistory.back();
  const float32_t score = track.scoreHistory.back();

  const float32_t top = bbox.min()[1] / imageSize[1];
  const float32_t left = bbox.min()[0] / imageSize[0];
  const float32_t bottom = bbox.max()[1] / imageSize[1];
  const float32_t right = bbox.max()[0] / imageSize[0];

  // TODO: top and left dimensions normalized with the wrong values !!!
  fd << frameId << "," << std::format("{0:%Y-%m-%d} {0:%T}", time) << "," << top << "," << left << "," << bottom << ","
     << right << "," << score << "," << worldPosition[0] << "," << worldPosition[1] << std::endl;
}

} // namespace

TrackWriter::TrackWriter(const std::filesystem::path &rootTracksPath, TrackData &track, float32_t fps)
    : track_{track}, rootTracksPath_{rootTracksPath} {
  const std::filesystem::path trackPath = getTrackPath(rootTracksPath, track.startTime, track.id, true /*temp*/);
  std::filesystem::create_directories(trackPath.parent_path());

  // Write csv
  infoPath_ = std::filesystem::path(trackPath).replace_extension(".csv");
  if (std::filesystem::exists(infoPath_)) {
    std::cout << "Warning: track info file already exists for new track (" << infoPath_ << ")" << std::endl;
  }

  int attemptCount = 0;
  const int MAX_ATTEMPT_COUNT = 5;
  while (attemptCount < MAX_ATTEMPT_COUNT) {
    infoFd_.clear();
    CHECK_TRUE(!infoFd_.is_open());
    infoFd_.open(infoPath_);
    if (!infoFd_.fail()) {
      break;
    } else {
      std::cout << std::format("Error opening track data file {}, errno: {}. Retrying with delay.", infoPath_.string(),
                               strerror(errno))
                << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  if (infoFd_.fail()) {
    throw ZooVisionError(
        std::format("Could not open the track data file {} after {} attempts.", infoPath_.string(), attemptCount));
  }
  writeHeader(infoFd_);

  // Start video
  videoPath_ = std::filesystem::path(trackPath).replace_extension(".mp4");
  const Vector2i cropSize = {PatchCropper::CROP_SIZE, PatchCropper::CROP_SIZE};

  if (!trackVideo_.open(videoPath_.string(), cropSize, fps)) {
    throw std::runtime_error(std::format("Could not create track video ({})", videoPath_.string()));
  }
}

void TrackWriter::writeFrame(uint64_t frameId, const at::Tensor &cropImage, const Vector3f &worldPosition,
                             const Vector2i imageSize) {
  CHECK_EQ(static_cast<int>(cropImage.size(1)), PatchCropper::CROP_SIZE);
  CHECK_EQ(static_cast<int>(cropImage.size(0)), PatchCropper::CROP_SIZE);

  CHECK_TRUE(infoFd_.is_open());
  writeRow(infoFd_, track_, frameId, worldPosition, imageSize);

  cv::Mat3b cropCv = wrapCvFromTensor3b(cropImage);
  // cv::imwrite((rootPath_ / "test.png").string(), cropCv);
  trackVideo_.write(cropCv.data, cropCv.step[0]);
}

void TrackWriter::close() {
  infoFd_.close();
  trackVideo_.close();

  // Rename
  const std::filesystem::path trackPath = getTrackPath(rootTracksPath_, track_.startTime, track_.id, false /*temp*/);
  try {
    const std::filesystem::path finalInfoPath = std::filesystem::path(trackPath).replace_extension(".csv");
    const std::filesystem::path finalVideoPath_ = std::filesystem::path(trackPath).replace_extension(".mp4");
    std::filesystem::rename(infoPath_, finalInfoPath);
    std::filesystem::rename(videoPath_, finalVideoPath_);
  } catch (const std::exception &ex) {
    std::cerr << "Error moving track: " << trackPath << std::endl << "Exception: " << ex.what() << std::endl;
  }
}

TrackWriter::~TrackWriter() {
  if (trackVideo_.isOpen()) {
    close();
  }
}

} // namespace zoo