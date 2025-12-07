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
#include <nlohmann/json.hpp>

namespace zoo {
namespace {

void logOtlException(const rclcpp::Logger &logger, const otl_exception &ex) {
  RCLCPP_ERROR(logger, "Error executing db command:\nMsg: %s\nSql: %s\nSql state: %s\nVariable: %s\n", ex.msg,
               ex.stm_text, ex.sqlstate, ex.var_info);
}

otl_datetime otlTimestampFromSys(const SysTime sysTime) {
  otl_datetime otime{};
  auto dp = std::chrono::floor<std::chrono::days>(sysTime); // dp is a sys_days, which is a
  // type alias for a C::time_point
  auto ymd = std::chrono::year_month_day{dp};
  std::chrono::hh_mm_ss time{std::chrono::floor<std::chrono::milliseconds>(sysTime - dp)};
  otime.year = static_cast<int>(ymd.year());
  otime.month = static_cast<unsigned int>(ymd.month());
  otime.day = static_cast<unsigned int>(ymd.day());
  otime.hour = time.hours().count();
  otime.minute = time.minutes().count();
  otime.second = time.seconds().count();
  otime.fraction = time.subseconds().count();
  otime.frac_precision = 3;
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
        /*buffer_size*/ 1,
        "INSERT INTO tracks(camera_id, start_time) VALUES(:camera_id<int,in>, :start_time<timestamp,in>) RETURNING id;",
        db_, otl_implicit_select);
    cmdInsertObservation_ = std::make_unique<otl_stream>(
        /*buffer_size*/ 10,
        "INSERT INTO observations(track_id, time, location, behaviour_id) VALUES(:track_id<int>, :time<timestamp>, "
        "point(:locationx<float>,:locationy<float>), :behaviour_id<int>);",
        db_);
    cmdCloseTrack_ = std::make_unique<otl_stream>(
        /*buffer_size*/ 1,
        "UPDATE tracks "
        "  SET end_time=:end_time<timestamp>,"
        "      frame_count=:frame_count<int>,"
        "      identity_id=:identity_id<int> "
        "  WHERE id=:id<int>;",
        db_);
    cmdInsertIdentityProb_ = std::make_unique<otl_stream>(
        /*buffer_size*/ 10,
        "INSERT INTO identity_probs(track_id, identity_id, prob) "
        "VALUES (:track_id<int>, :identity_id<int>, :prob<float>)",
        db_);

  } catch (const otl_exception &ex) {
    logOtlException(get_logger(), ex);
    throw;
  }

  // Subscribe to all cameras
  const rclcpp::SubscriptionOptions subOptions{};
  const rclcpp::QoS qos{/*history_depth*/ 2};
  auto subscribeDetection = [&](int cameraIndex, std::string cameraName) {
    const std::string channel = cameraName + "/detections";
    auto subscription = rclcpp::create_subscription<zoo_msgs::msg::Detection>(
        *this, channel, qos,
        [this, cameraIndex](std::shared_ptr<const zoo_msgs::msg::Detection> msg) {
          this->onDetection(cameraIndex, *msg);
        },
        subOptions);
    RCLCPP_INFO(get_logger(), "Subscribed to detection results %s (loans=%d)", channel.c_str(),
                subscription->can_loan_messages());
    detectionSubscribers_.push_back(std::move(subscription));
  };
  auto subscribeClosedTrack = [&](int cameraIndex, std::string cameraName) {
    const std::string channel = cameraName + "/track_closed";
    auto subscription = rclcpp::create_subscription<zoo_msgs::msg::TrackClosed>(
        *this, channel, qos,
        [this, cameraIndex](std::shared_ptr<const zoo_msgs::msg::TrackClosed> msg) {
          this->onTrackClosed(cameraIndex, *msg);
        },
        subOptions);
    trackClosedSubscribers_.push_back(std::move(subscription));
  };

  // Camera names are hardcoded to match their indices with the db cameras table
  const std::vector<std::string> cameraNames = {"zag_elp_cam_016", "zag_elp_cam_017", "zag_elp_cam_018",
                                                "zag_elp_cam_019"};
  for (const auto [idx, name] : std::views::enumerate(cameraNames)) {
    subscribeDetection(idx, name);
    subscribeClosedTrack(idx, name);
  }
}

void DbForwarder::onDetection(int cameraIndex, const zoo_msgs::msg::Detection &msg) {
  const SysTime msgTime = sysTimeFromRos(msg.header.stamp);
  const auto locationSpan =
      std::span(reinterpret_cast<const Eigen::Vector3f *>(msg.world_positions.data()), msg.detection_count);
  for (const auto idx : std::views::iota(0u, msg.detection_count)) {
    insertObservation(cameraIndex, msg.track_ids[idx], msgTime, {locationSpan[idx][0], locationSpan[idx][1]},
                      msg.behaviour_ids[idx]);
  }
  cmdInsertObservation_->flush();
}

int DbForwarder::createTrack(int cameraIndex, TrackId id, SysTime startTime) {
  try {
    otl_stream &cmd = *cmdCreateTrack_;

    cmd << cameraIndex;
    cmd << otlTimestampFromSys(startTime);

    int dbId = -1;
    cmd >> dbId;
    dbTrackIdFromCameraAndTrackId_.insert(std::make_pair(std::make_pair(cameraIndex, id), dbId));

    return dbId;
  } catch (const otl_exception &ex) {
    logOtlException(get_logger(), ex);
    throw;
  }
}

int DbForwarder::getOrCreateTrack(int cameraIndex, TrackId id, SysTime startTime) {
  const auto key = std::make_pair(cameraIndex, id);
  auto it = dbTrackIdFromCameraAndTrackId_.find(key);
  if (it == dbTrackIdFromCameraAndTrackId_.end()) {
    return createTrack(cameraIndex, id, startTime);
  } else {
    return it->second;
  }
}
void DbForwarder::insertObservation(int cameraIndex, TrackId id, SysTime time, Eigen::Vector2f location,
                                    TBehaviour behaviourId) {
  try {
    const int dbTrackId = getOrCreateTrack(cameraIndex, id, time);

    otl_stream &cmd = *cmdInsertObservation_;
    cmd << dbTrackId;
    cmd << otlTimestampFromSys(time);
    cmd << location[0];
    cmd << location[1];
    cmd << static_cast<int>(behaviourId);
  } catch (const otl_exception &ex) {
    logOtlException(get_logger(), ex);
    throw;
  }
}

void DbForwarder::onTrackClosed(int cameraIndex, const zoo_msgs::msg::TrackClosed &msg) {
  const SysTime msgTime = sysTimeFromRos(msg.header.stamp);
  Eigen::Map<const Eigen::VectorXf> identityProbs(msg.identity_probs.data(), msg.identity_probs.size());
  closeTrack(cameraIndex, msg.track_id, msgTime, msg.track_length, msg.selected_identity, identityProbs);

  // Forget the track
  auto it = dbTrackIdFromCameraAndTrackId_.find(std::make_pair(cameraIndex, msg.track_id));
  assert(it != dbTrackIdFromCameraAndTrackId_.end());
  dbTrackIdFromCameraAndTrackId_.erase(it);
}

void DbForwarder::closeTrack(int cameraIndex, TrackId id, SysTime endTime, int frameCount, TIdentity identityId,
                             Eigen::VectorXf identityProbs) {
  try {
    const int dbTrackId = getOrCreateTrack(cameraIndex, id, endTime);

    {
      otl_stream &cmd = *cmdCloseTrack_;
      cmd << otlTimestampFromSys(endTime);
      cmd << frameCount;
      cmd << static_cast<int>(identityId);
      cmd << dbTrackId;
    }

    {
      otl_stream &cmd = *cmdInsertIdentityProb_;
      for (const auto identityId : std::views::iota(0l, identityProbs.size())) {
        cmd << dbTrackId;
        cmd << static_cast<int>(identityId);
        cmd << identityProbs[identityId];
      }
      cmd.flush();
    }
  } catch (const otl_exception &ex) {
    logOtlException(get_logger(), ex);
    throw;
  }
}

} // namespace zoo