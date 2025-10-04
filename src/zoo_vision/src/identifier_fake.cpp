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

#include "zoo_vision/identifier_fake.hpp"
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

IdentifierFake::IdentifierFake(int nameIndex) : name_{std::format("IdentifierFake_{}", nameIndex)} {
  RCLCPP_INFO(rclcpp::get_logger(name_), "Starting %s", name_.c_str());
}

void IdentifierFake::onKeyframe(TKeyframeIndex /*keyframeIndex*/, const at::Tensor &/*patch_f32*/, TrackData &track) {
  track.selectedIdentity = INVALID_IDENTITY;
}

} // namespace zoo
