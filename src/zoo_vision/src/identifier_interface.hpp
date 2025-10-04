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

#include "zoo_vision/keyframe_store.hpp"
#include "zoo_vision/track_matcher.hpp"

#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>

#include <filesystem>

namespace zoo {

class IIdentifier {
public:
  virtual ~IIdentifier() = default;

  virtual void onKeyframe(TKeyframeIndex keyframeIndex, const at::Tensor &patch_f32, TrackData &track) = 0;
};

std::unique_ptr<IIdentifier> makeIdentifier(int nameIndex, std::string cameraName, TrackMatcher &trackMatcher,
                                            std::optional<at::cuda::CUDAStream> cudaStream);
} // namespace zoo