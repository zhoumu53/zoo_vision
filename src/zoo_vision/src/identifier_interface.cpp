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
#include "zoo_vision/identifier_interface.hpp"
#include "zoo_vision/identifier.hpp"
#include "zoo_vision/identifier_fake.hpp"
#include "zoo_vision/utils.hpp"

namespace zoo {

std::unique_ptr<IIdentifier> makeIdentifier(int nameIndex, std::string cameraName, TrackMatcher &trackMatcher,
                                            std::optional<at::cuda::CUDAStream> cudaStream) {
  const auto config = getConfig();
  if (config["detection"]["model"].get<std::string>().empty()) {
    return std::make_unique<IdentifierFake>(nameIndex);
  } else {
    return std::make_unique<Identifier>(nameIndex, cameraName, trackMatcher, cudaStream);
  }
}
} // namespace zoo