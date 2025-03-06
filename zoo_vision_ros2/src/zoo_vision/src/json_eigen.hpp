#pragma once

#include <iostream>
#include <type_traits>
#include <typeinfo>

#include <Eigen/Core>

#include <nlohmann/json_fwd.hpp>

/**
 * Provide to_json() and from_json() overloads for nlohmann::json,
 * which allows simple syntax like:
 *
 * @code
 * Eigen::Matrix3f in, out;
 *
 * json j;
 * j = in;
 * out = j;
 * @endcode
 *
 * @see https://github.com/nlohmann/json#arbitrary-types-conversions
 */
namespace Eigen {

// MatrixBase
template <typename Derived> void to_json(nlohmann::json &j, const MatrixBase<Derived> &matrix);
template <typename Derived> void from_json(const nlohmann::json &j, MatrixBase<Derived> &matrix);

// Specialization for Vector3f
template <> void to_json<Vector3f>(nlohmann::json &j, const MatrixBase<Vector3f> &vector);
template <> void from_json<Vector3f>(const nlohmann::json &j, MatrixBase<Vector3f> &vector);

// Quaternion
template <typename Derived> void to_json(nlohmann::json &j, const QuaternionBase<Derived> &quat);
template <typename Derived> void from_json(const nlohmann::json &j, QuaternionBase<Derived> &quat);
} // namespace Eigen

#include <nlohmann/json.hpp>

namespace Eigen {
// IMPLEMENTATION
template <typename Derived> void to_json(nlohmann::json &j, const MatrixBase<Derived> &matrix) {
  for (int row = 0; row < matrix.rows(); ++row) {
    nlohmann::json column = nlohmann::json::array();
    for (int col = 0; col < matrix.cols(); ++col) {
      column.push_back(matrix(row, col));
    }
    j.push_back(column);
  }
}

template <typename Derived> void from_json(const nlohmann::json &j, MatrixBase<Derived> &matrix) {
  using Scalar = typename MatrixBase<Derived>::Scalar;

  for (std::size_t row = 0; row < j.size(); ++row) {
    const auto &jrow = j.at(row);
    for (std::size_t col = 0; col < jrow.size(); ++col) {
      const auto &value = jrow.at(col);
      matrix(row, col) = value.get<Scalar>();
    }
  }
}

} // namespace Eigen