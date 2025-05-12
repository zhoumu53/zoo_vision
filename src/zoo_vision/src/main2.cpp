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

#include "rclcpp/rclcpp.hpp"
#include "zoo_vision/rerun_forwarder.hpp"
#include "zoo_vision/zoo_camera.hpp"
#include <memory>

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor exec;
  rclcpp::NodeOptions options;
  // auto camera_node = std::make_shared<zoo::ZooCamera>(options);
  auto rerun_node = std::make_shared<zoo::RerunForwarder>(options);

  // exec.add_node(camera_node);
  exec.add_node(rerun_node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}