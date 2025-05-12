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
#include "zoo_msgs/msg/track_closed.hpp"

#include "zoo_vision/timings.hpp"
#include "zoo_vision/types.hpp"

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#define OTL_ODBC_UNIX
#define OTL_ODBC_POSTGRESQL
#define OTL_STREAM_WITH_STD_SPAN_ON
#define OTL_CPP_23_ON
#define OTL_STL
#include <otlv4.h>

namespace std {

// Definition of std::hash() for std::pair<A,B>
// Copied from: https://www.reddit.com/r/cpp_questions/comments/us3nyb/why_doesnt_c_have_a_default_pairint_int_hash/
template <class A, class B> struct hash<pair<A, B>> {
  size_t operator()(const pair<A, B> &p) const { return std::rotl(hash<A>{}(p.first), 1) ^ hash<B>{}(p.second); }
};
} // namespace std

namespace zoo {

class DbForwarder : public rclcpp::Node {
public:
  explicit DbForwarder(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  virtual ~DbForwarder() noexcept = default;

  void onDetection(int cameraIndex, const zoo_msgs::msg::Detection &msg);
  void onTrackClosed(int cameraIndex, const zoo_msgs::msg::TrackClosed &msg);

private:
  int createTrack(int cameraIndex, TrackId id, SysTime startTime);
  int getOrCreateTrack(int cameraIndex, TrackId id, SysTime startTime);
  void insertObservation(int cameraIndex, TrackId id, SysTime time, Eigen::Vector2f location, TBehaviour behaviourId);
  void closeTrack(int cameraIndex, TrackId, SysTime endTime, int frameCount, TIdentity identityId,
                  Eigen::VectorXf identityProbs);

  otl_connect db_;
  std::unique_ptr<otl_stream> cmdCreateTrack_;
  std::unique_ptr<otl_stream> cmdInsertObservation_;
  std::unique_ptr<otl_stream> cmdCloseTrack_;
  std::unique_ptr<otl_stream> cmdInsertIdentityProb_;

  std::unordered_map<std::pair<int, TrackId>, int> dbTrackIdFromCameraAndTrackId_;

  std::vector<std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Detection>>> detectionSubscribers_;
  std::vector<std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::TrackClosed>>> trackClosedSubscribers_;
};
} // namespace zoo