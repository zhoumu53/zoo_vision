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

#include "zoo_vision/stats.hpp"

#include <builtin_interfaces/msg/detail/time__struct.hpp>
#include <rclcpp/time.hpp>

#include <chrono>
#include <numeric>

namespace zoo {

using SysClock = std::chrono::system_clock;
using SysTime = SysClock::time_point;
using nanoseconds = std::chrono::nanoseconds;
using seconds = std::chrono::seconds;
using SecondsTimePoint = std::chrono::time_point<SysClock, seconds>;

inline SysTime sysTimeFromRos(const builtin_interfaces::msg::Time rosTime) {
  const SysTime frameTimeNs{nanoseconds{rclcpp::Time(rosTime).nanoseconds()}};
  return frameTimeNs;
}

inline SecondsTimePoint secondsTimePointFromTimePoint(const SysTime time) {
  return std::chrono::time_point<SysClock, seconds>{std::chrono::duration_cast<seconds>(time.time_since_epoch())};
}

class RateSampler {
public:
  using clock_t = std::chrono::high_resolution_clock;
  using time_point_t = clock_t::time_point;

  RateSampler(size_t maxSamples = 50) : stats_{maxSamples} {}

  void tick() {
    const time_point_t now = clock_t::now();
    if (lastTick_) {
      const auto period = now - *lastTick_;
      const int64_t period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(period).count();
      constexpr auto NS_TO_S = 1e-9f;
      const float period_s = static_cast<float>(period_ns) * NS_TO_S;
      stats_.push(period_s);
    }
    lastTick_ = now;
  }

  float rateHz() const { return 1.0f / stats_.mean(); }

private:
  BufferedStats stats_;
  std::optional<clock_t::time_point> lastTick_;
};

class TimedSection {
public:
  using clock_t = std::chrono::high_resolution_clock;

  TimedSection() : startTime_{clock_t::now()} {}
  std::chrono::nanoseconds time() const { return clock_t::now() - startTime_; }

  clock_t::time_point startTime_;
};
} // namespace zoo