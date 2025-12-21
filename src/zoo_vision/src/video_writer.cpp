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

#include "zoo_vision/video_writer.hpp"
#include <iostream>

namespace zoo {
VideoWriter::VideoWriter() {}

VideoWriter::~VideoWriter() {}

bool VideoWriter::open(std::string_view filename, Vector2i frameSize, float32_t fps) {
  const std::vector<int> params{{cv::VIDEOWRITER_PROP_IS_COLOR, 1 /*, cv::VIDEOWRITER_PROP_QUALITY, 0*/}};
  const auto fourcc = "h264";
  return writer_.open(std::string(filename), cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]), fps,
                      cv::Size{frameSize[0], frameSize[1]}, params);
}
void VideoWriter::write(const cv::Mat3b &img) { writer_ << img; }
} // namespace zoo