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

#include "zoo_vision/types.hpp"

#include "zoo_msgs/msg/bounding_box2_d.hpp"
#include "zoo_msgs/msg/image12m.hpp"
#include "zoo_msgs/msg/image4m.hpp"
#include "zoo_msgs/msg/key_value_arrayf.hpp"
#include "zoo_msgs/msg/key_value_arrayi64.hpp"
#include "zoo_msgs/msg/string.hpp"
#include "zoo_msgs/msg/tensor3b32m.hpp"

#include <ATen/Tensor.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>

#include <filesystem>

namespace zoo {

std::filesystem::path getDataPath();

void setMsgString(zoo_msgs::msg::String &dest, const std::string_view &src);
std::string_view getMsgString(const zoo_msgs::msg::String &dest);

void addRosKeyValue(zoo_msgs::msg::KeyValueArrayi64 &array, const std::string_view &key, int64_t value);
void addRosKeyValue(zoo_msgs::msg::KeyValueArrayf &array, const std::string_view &key, float value);

void copyBboxToRos(zoo_msgs::msg::BoundingBox2D &outBbox, const AlignedBox2f &in);

cv::Mat3b wrapMat3bFromMsg(zoo_msgs::msg::Image12m &);

// DANGER: Message is const but the returned mat will cast away the const
// Do not modify it!
cv::Mat3b wrapMat3bFromMsg(const zoo_msgs::msg::Image12m &);

cv::Mat1b wrapCvFromTensor1b(const at::Tensor img);

void copyMat1bToMsg(const cv::Mat1b &, zoo_msgs::msg::Image4m &);
at::Tensor mapRosTensor(zoo_msgs::msg::Tensor3b32m &rosTensor);

void loadConfig();
nlohmann::json &getConfig();

bool saveTensorImage(const at::Tensor &imgTensor, const std::string &name);

} // namespace zoo

/////////////////////////////////////////
// Implementations
namespace zoo {
inline std::string_view getMsgString(const zoo_msgs::msg::String &dest) {
  return std::string_view(reinterpret_cast<const char *>(dest.data.data()), dest.size);
}
} // namespace zoo