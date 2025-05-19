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

// GCC raises a false-positive with eigen array assignments
// See https://gitlab.com/libeigen/eigen/-/issues/2506
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overread"

#include <Eigen/Dense>
#include <cstdint>
#include <format>
#include <stacktrace>
#include <stdexcept>

using float32_t = float;

#define CHECK_EQ(a, b)                                                                                                 \
  if (!(a == b)) {                                                                                                     \
    throw ZooVisionError(std::format("Check failed: " #a "==" #b ", with \n" #a "={}\n" #b "={}", a, b));              \
  }

#define CHECK_TRUE(a) CHECK_EQ((a), true)

#define CHECK_LE(a, b)                                                                                                 \
  if (!(a <= b)) {                                                                                                     \
    throw ZooVisionError(std::format("Check failed: " #a "<=" #b ", with \n" #a "={}\n" #b "={}", a, b));              \
  }

namespace zoo {

using TrackId = uint32_t;

using TClassId = uint32_t;
using TIdentity = uint32_t;
constexpr TIdentity INVALID_IDENTITY = TIdentity(0);

using TBehaviour = uint32_t;
constexpr TBehaviour INVALID_BEHAVIOUR = TBehaviour(0);

using Matrix3f = Eigen::Matrix3f;
using MatrixX2f = Eigen::MatrixX2f;
using MatrixX3f = Eigen::MatrixX3f;
using Vector2i = Eigen::Vector2i;
using Vector2f = Eigen::Vector2f;

using AlignedBox2f = Eigen::AlignedBox2f;

////////////////////////////////////////
// Exceptions
class ZooVisionError : public std::runtime_error {
public:
  ZooVisionError(const std::string &__arg) : std::runtime_error(__arg) { trace = std::stacktrace::current(1); }

  std::stacktrace trace;
};

} // namespace zoo