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

#include <opencv2/videoio.hpp>

#include <filesystem>
#include <memory>

namespace zoo {
namespace detail {
class VideoWriterImpl;
}

/* Class dedicated to write videos.
We want to abstract cv::VideoWriter away because OpenCV does not allow
configuring the bitrate and the track videos have terrible quality.
We'll use directly libav soon to have better control.*/
class VideoWriter {
public:
  VideoWriter();
  ~VideoWriter();

  bool isOpen() const { return impl_ != nullptr; }
  bool open(const std::string &filename, Vector2i frameSize, float32_t fps);
  void write(const cv::Mat3b &img);

private:
  std::unique_ptr<detail::VideoWriterImpl> impl_;
};
} // namespace zoo