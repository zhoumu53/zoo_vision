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
#include "zoo_vision/segmenter_interface.hpp"
#include "zoo_vision/segmenter_fake.hpp"
#include "zoo_vision/segmenter_yolo.hpp"
#include "zoo_vision/utils.hpp"

#include <nlohmann/json.hpp>

namespace zoo {

std::unique_ptr<ISegmenter> makeSegmenter(int nameIndex, std::string cameraName_,
                                          std::optional<at::cuda::CUDAStream> cudaStream_) {
  const auto config = getConfig();
  if (config["detection"]["model"].get<std::string>().empty()) {
    return std::make_unique<SegmenterFake>(nameIndex);
  } else {
    return std::make_unique<SegmenterYolo>(nameIndex, cameraName_, cudaStream_);
  }
}

} // namespace zoo