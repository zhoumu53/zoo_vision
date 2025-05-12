#include "zoo_vision/json_eigen.hpp"

#include <nlohmann/json.hpp>

namespace Eigen {

template <> void to_json<Vector3f>(nlohmann::json &j, const MatrixBase<Vector3f> &vector) {
  j["x"] = vector.x();
  j["y"] = vector.y();
  j["z"] = vector.z();
}

template <> void from_json<Vector3f>(const nlohmann::json &j, MatrixBase<Vector3f> &vector) {
  vector.x() = j.at("x").get<float>();
  vector.y() = j.at("y").get<float>();
  vector.z() = j.at("z").get<float>();
}

template <typename Derived> void to_json(nlohmann::json &j, const QuaternionBase<Derived> &quat) {
  j["qw"] = quat.w();
  j["qx"] = quat.x();
  j["qy"] = quat.y();
  j["qz"] = quat.z();
}

template <typename Derived> void from_json(const nlohmann::json &j, QuaternionBase<Derived> &quat) {
  using Scalar = typename QuaternionBase<Derived>::Scalar;
  quat.w() = j.at("qw").get<Scalar>();
  quat.x() = j.at("qx").get<Scalar>();
  quat.y() = j.at("qy").get<Scalar>();
  quat.z() = j.at("qz").get<Scalar>();
}

} // namespace Eigen