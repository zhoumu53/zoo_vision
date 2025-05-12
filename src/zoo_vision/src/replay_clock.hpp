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

#include <chrono>

namespace zoo {

struct ReplayClock {
  typedef std::chrono::milliseconds duration;
  typedef duration::rep rep;
  typedef duration::period period;
  typedef std::chrono::time_point<ReplayClock> time_point;
  static const bool is_steady = false;

  static constexpr int epochYear_ = 2025;
  static constexpr int epochMonth_ = 1;
  static constexpr int epochDay_ = 1;

  time_point now_;

  time_point now() noexcept { return now_; }
  static std::chrono::system_clock::time_point toSystem(time_point);
};

} // namespace zoo

////////////////////////////////////////
// Implementation
#include <spanstream>

namespace zoo {

inline ReplayClock::time_point parseTime(std::string_view timeStr) {
  std::tm t = {};
  std::ispanstream ss(timeStr);

  constexpr auto DATE_FORMAT = "%Y-%m-%dT%H:%M:%S";
  ss >> std::get_time(&t, DATE_FORMAT);

  ReplayClock::duration dur = std::chrono::years(t.tm_year - ReplayClock::epochYear_) +
                              std::chrono::months(t.tm_mon - ReplayClock::epochMonth_) +
                              std::chrono::days(t.tm_mday - ReplayClock::epochDay_) + std::chrono::hours(t.tm_hour) +
                              std::chrono::hours(t.tm_min) + std::chrono::hours(t.tm_sec);
  return ReplayClock::time_point(dur);
}

inline ReplayClock::duration parseDuration(std::string_view timeStr) {
  std::tm t = {};
  std::ispanstream ss(timeStr);

  constexpr auto DATE_FORMAT = "%H:%M";
  ss >> std::get_time(&t, DATE_FORMAT);

  ReplayClock::duration dur = std::chrono::hours(t.tm_hour) + std::chrono::hours(t.tm_min);
  return dur;
}

inline std::chrono::system_clock::time_point ReplayClock::toSystem(time_point replayTime) {
  std::chrono::sys_days replayEpoch = std::chrono::year_month_day(std::chrono::year(ReplayClock::epochYear_),
                                                                  std::chrono::month(ReplayClock::epochMonth_),
                                                                  std::chrono::day(ReplayClock::epochDay_));

  return std::chrono::system_clock::time_point{replayEpoch}; // + replayTime.time_since_epoch();
}
} // namespace zoo
