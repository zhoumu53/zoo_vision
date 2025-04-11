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

#include "zoo_vision/db_forwarder.hpp"

#include "zoo_vision/utils.hpp"

#include <ctime>
#include <iostream>

namespace zoo {
namespace {

void logOtlException(const rclcpp::Logger &logger, const otl_exception &ex) {
  RCLCPP_ERROR(logger, "Error executing db command:\nMsg: %s\nSql: %s\nSql state: %s\nVariable: %s\n", ex.msg,
               ex.stm_text, ex.sqlstate, ex.var_info);
}

otl_datetime otlTimestampFromSys(const SysTime time) {

  otl_datetime otime{};
  // otime.year = utc_tm.tm_year;
  // otime.month = utc_tm.tm_mon;
  // otime.day = utc_tm.tm_mday;
  // otime.hour = utc_tm.tm_hour;
  // otime.minute = utc_tm.tm_min;
  // otime.second = utc_tm.tm_sec;
  // otime.fraction = 10;
  // otime.frac_precision = 3;
  return otime;
}
} // namespace

DbForwarder::DbForwarder(const rclcpp::NodeOptions &options) : rclcpp::Node("DbForwarder", options) {
  try {
    // initialize ODBC environment
    otl_connect::otl_initialize();

    const auto &config = getConfig();
    const std::string logonString = config["db"]["logon"];

    db_.rlogon(logonString.c_str(), /*auto_commit*/ 1);

    cmdCreateTrack_ = std::make_unique<otl_stream>(
        /*buffer_size*/ 10, "INSERT INTO tracks(start_time) VALUES(:start_time<timestamp,in>) RETURNING id;", db_,
        otl_implicit_select);
    cmdInsertObservation_ = std::make_unique<otl_stream>(
        /*buffer_size*/ 10,
        "INSERT INTO observations(track_id, time, location, behaviour_id) VALUES(:track_id<int>, :time<timestamp>, "
        ":location<text>, :behaviour_id<int>);",
        db_);

    insertObservation(4, SysClock::now(), {1.0f, 66.f}, 2);

    std::terminate();
  } catch (const otl_exception &ex) {
    logOtlException(get_logger(), ex);
    throw;
  }
}

void DbForwarder::onDetection(std::shared_ptr<const zoo_msgs::msg::Detection> msg);

void DbForwarder::createTrack(TrackId id, SysTime startTime) {
  otl_stream &cmd = *cmdCreateTrack_;

  otl_datetime time{};
  time.year = 2025;
  time.month = 01;
  time.day = 01;
  time.hour = 10;
  time.minute = 22;
  time.second = 03;
  time.fraction = 10;
  time.frac_precision = 3;
  cmd << time;

  int id = 666;
  cmd >> id;
  std::cout << "New id: " << id << std::endl;
}

void DbForwarder::insertObservation(TrackId id, SysTime time, Eigen::Vector2f location, TBehaviour behaviourId) {
  otl_stream &cmd = *cmdInsertObservation_;
  cmd << id;
}
void DbForwarder::closeTrack(TrackId, SysTime endTime, int frameCount, TIdentity identityId,
                             Eigen::VectorXf identityProbs) {}

} // namespace zoo