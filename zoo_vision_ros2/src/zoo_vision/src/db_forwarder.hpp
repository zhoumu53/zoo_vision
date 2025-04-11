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

#include "zoo_msgs/msg/detection.hpp"

#include "zoo_vision/timings.hpp"
#include "zoo_vision/types.hpp"

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#define OTL_ODBC_UNIX
#define OTL_ODBC_POSTGRESQL
#define OTL_CPP_23_ON
#define OTL_STL
#include <otlv4.h>

namespace zoo {

class DbForwarder : public rclcpp::Node {
public:
  explicit DbForwarder(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  virtual ~DbForwarder() noexcept = default;

  void onDetection(std::shared_ptr<const zoo_msgs::msg::Detection> msg);

private:
  void createTrack(TrackId id, SysTime startTime);
  void insertObservation(TrackId id, SysTime time, Eigen::Vector2f location, TBehaviour behaviourId);
  void closeTrack(TrackId, SysTime endTime, int frameCount, TIdentity identityId, Eigen::VectorXf identityProbs);

  otl_connect db_;
  std::unique_ptr<otl_stream> cmdCreateTrack_;
  std::unique_ptr<otl_stream> cmdInsertObservation_;
};
} // namespace zoo