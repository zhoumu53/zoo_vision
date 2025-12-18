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

#include <cstdint>
#include <format>
#include <stacktrace>
#include <stdexcept>

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

using float32_t = float;

#define CHECK_EQ(a, b)                                                                                                 \
  if (!(a == b)) {                                                                                                     \
    throw ZooVisionError(std::format("Check failed: " #a "==" #b ", with \n  " #a "={}\n  " #b "={}", a, b));          \
  }
#define CHECK_NE(a, b)                                                                                                 \
  if (!(a != b)) {                                                                                                     \
    throw ZooVisionError(std::format("Check failed: " #a "!=" #b ", with \n  " #a "={}\n  " #b "={}", a, b));          \
  }

#define CHECK_NOT_NULL(a)                                                                                              \
  if (!(a != nullptr)) {                                                                                               \
    throw ZooVisionError(std::format("Check failed: " #a "!=nullptr"));                                                \
  }

#define CHECK_TRUE(a) CHECK_EQ((a), true)
#define CHECK_FALSE(a) CHECK_EQ((a), false)

#define CHECK_LE(a, b)                                                                                                 \
  if (!(a <= b)) {                                                                                                     \
    throw ZooVisionError(std::format("Check failed: " #a "<=" #b ", with \n  " #a "={}\n  " #b "={}", a, b));          \
  }
#define CHECK_GE(a, b)                                                                                                 \
  if (!(a >= b)) {                                                                                                     \
    throw ZooVisionError(std::format("Check failed: " #a ">=" #b ", with \n  " #a "={}\n  " #b "={}", a, b));          \
  }

namespace cv {
// Define opencv DataType for Eigen Vectors so that we can use it inside the fillConvexPoly() function.
template <typename _Tp> class DataType<Eigen::Vector2<_Tp>> {
public:
  typedef Eigen::Vector2<_Tp> value_type;
  typedef Eigen::Vector2<typename DataType<_Tp>::work_type> work_type;
  typedef _Tp channel_type;

  enum {
    generic_type = 0,
    channels = 2,
    fmt = traits::SafeFmt<channel_type>::fmt + ((channels - 1) << 8),
    depth = DataType<channel_type>::depth,
    type = CV_MAKETYPE(depth, channels)
  };

  typedef Vec<channel_type, channels> vec_type;
};

} // namespace cv

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
using Matrix3Xf = Eigen::Matrix3Xf;

template <class T, int dims> using Vector = Eigen::Matrix<T, dims, 1>;
using Vector2i = Vector<int, 2>;
using Vector2f = Vector<float32_t, 2>;

using Polygon = std::vector<Vector2i>;

template <class T, int dims> using AlignedBox = Eigen::AlignedBox<T, dims>;
using AlignedBox2f = AlignedBox<float32_t, 2>;

////////////////////////////////////////
// Exceptions
class ZooVisionError : public std::runtime_error {
public:
  ZooVisionError(const std::string &__arg) : std::runtime_error(__arg) { trace = std::stacktrace::current(1); }

  std::stacktrace trace;
};

} // namespace zoo