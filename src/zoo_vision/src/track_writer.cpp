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

#include <mutex>
#include <opencv2/highgui.hpp>

namespace zoo {

TrackWriter::TrackWriter(std::filesystem::path rootPath) : rootPath_{rootPath} {}

void TrackWriter::startVideo(TrackData &track, const at::Tensor &cropImage) {
  const std::filesystem::path dir = rootPath_ / std::format("{:%Y-%m-%d}", track.startTime);
  std::filesystem::create_directories(dir);

  const std::filesystem::path videoPath = dir / std::format("{:06d}.mkv", track.id);

  const Vector2i cropSize = {static_cast<int>(cropImage.size(1)), static_cast<int>(cropImage.size(0))};

  if (!track.trackVideo.open(videoPath.string(),  cropSize)) {
    throw std::runtime_error(std::format("Could not create track video ({})", videoPath.string()));
  }

  addFrame(track, cropImage);
}

void TrackWriter::addFrame(TrackData &track, const at::Tensor &cropImage) {

  cv::Mat3b cropCv = wrapCvFromTensor3b(cropImage);
  // cv::imwrite((rootPath_ / "test.png").string(), cropCv);
  track.trackVideo.write(cropCv);
}

} // namespace zoo