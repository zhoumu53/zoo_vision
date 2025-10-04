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

#include "zoo_vision/behaviourer_fake.hpp"
#include <rclcpp/rclcpp.hpp>

#include <ranges>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

BehaviourerFake::BehaviourerFake(int nameIndex) : name_{std::format("behaviourer_fake_{}", nameIndex)} {
  RCLCPP_INFO(rclcpp::get_logger(name_), "Starting %s", name_.c_str());
}

void BehaviourerFake::onDetection(zoo_msgs::msg::Detection &msg, const at::Tensor &patches) {
  const size_t patchCount = patches.size(0);
  for (const auto i : std::views::iota(0u, patchCount)) {
    msg.behaviour_ids[i] = 0;
    for (auto &logit : msg.behaviour_logits) {
      logit = 0;
    }
    msg.behaviour_logits[0] = 1;
  }
}

} // namespace zoo
