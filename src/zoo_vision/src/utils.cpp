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

#include "zoo_vision/utils.hpp"

#include <ATen/ops/from_blob.h>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <string.h>

namespace zoo {
namespace {
/// Can only be accessed through loadConfig() and getConfig()
std::unique_ptr<nlohmann::json> global_config;
} // namespace

std::filesystem::path getDataPath() {
  static std::filesystem::path dataPath = {};
  if (dataPath.empty()) {
    const int MAX_DEPTH = 10;

    std::filesystem::path root = std::filesystem::path(".");
    int depth = 0;
    while (depth < MAX_DEPTH) {
      dataPath = root / "data";
      if (std::filesystem::is_directory(dataPath)) {
        return dataPath;
      }

      depth++;
      root = root / "..";
    }
    throw ZooVisionError("Could not find data path");
  } else {
    return dataPath;
  }
}

void setMsgString(zoo_msgs::msg::String &dest, const std::string_view &src) {
  constexpr auto MAX_SIZE = zoo_msgs::msg::String::MAX_SIZE;

  size_t len = src.size();
  if (len > MAX_SIZE - 1) {
    len = MAX_SIZE - 1;
  }

  memcpy(dest.data.data(), src.data(), len);
  dest.data[len] = 0;
  dest.size = len;
}

void addRosKeyValue(zoo_msgs::msg::KeyValueArrayi64 &array, const std::string_view &key, int64_t value) {
  const auto idx = array.item_count;
  array.item_count += 1;
  setMsgString(array.keys[idx], key);
  array.values[idx] = value;
}
void addRosKeyValue(zoo_msgs::msg::KeyValueArrayf &array, const std::string_view &key, float value) {
  const auto idx = array.item_count;
  array.item_count += 1;
  setMsgString(array.keys[idx], key);
  array.values[idx] = value;
}

void copyBboxToRos(zoo_msgs::msg::BoundingBox2D &outBbox, const Eigen::AlignedBox2f &in) {
  const Eigen::Vector2f center = in.center();
  const Eigen::Vector2f halfSize = in.sizes() / 2;
  outBbox.center[0] = center[0];
  outBbox.center[1] = center[1];
  outBbox.half_size[0] = halfSize[0];
  outBbox.half_size[1] = halfSize[1];
}

namespace detail {
template <class TMsg> cv::Mat3b wrapMat3bFromMsg(TMsg &msg) {
  // DANGER: casting away const. Data may be modified by opencv if we're not careful
  auto *dataPtr = reinterpret_cast<cv::Vec3b *>(const_cast<unsigned char *>(msg.data.data()));
  CHECK_LE(msg.step * msg.height, TMsg::DATA_MAX_SIZE);
  return cv::Mat3b(msg.height, msg.width, dataPtr, msg.step);
}
} // namespace detail

cv::Mat3b wrapMat3bFromMsg(zoo_msgs::msg::Image12m &msg) { return detail::wrapMat3bFromMsg(msg); }
cv::Mat3b wrapMat3bFromMsg(const zoo_msgs::msg::Image12m &msg) { return detail::wrapMat3bFromMsg(msg); }

void copyMat1bToMsg(const cv::Mat1b &img, zoo_msgs::msg::Image4m &msg) {
  setMsgString(msg.encoding, "mono8");
  msg.width = img.cols;
  msg.height = img.rows;
  msg.is_bigendian = false;
  msg.step = img.step;
  size_t byteCount = msg.step * msg.height;
  CHECK_LE(byteCount, zoo_msgs::msg::Image4m::DATA_MAX_SIZE);
  memcpy(msg.data.data(), img.data, byteCount);
}

at::Tensor mapRosTensor(zoo_msgs::msg::Tensor3b32m &rosTensor) {
  [[maybe_unused]] const size_t totalSize = rosTensor.sizes[0] * rosTensor.sizes[1] * rosTensor.sizes[2];
  CHECK_LE(totalSize, zoo_msgs::msg::Tensor3b32m::DATA_MAX_SIZE);
  return at::from_blob(rosTensor.data.data(), {rosTensor.sizes[0], rosTensor.sizes[1], rosTensor.sizes[2]},
                       at::TensorOptions().dtype(at::kByte));
}

void loadConfig() {
  std::ifstream f(getDataPath() / "config.json");

  nlohmann::json config = nlohmann::json::parse(f);
  global_config = std::make_unique<nlohmann::json>(std::move(config));
}
nlohmann::json &getConfig() {
  if (global_config == nullptr) {
    throw ZooVisionError("Config not loaded, call loadConfig() first");
  }
  return *global_config;
}

bool saveTensorImage(const at::Tensor &imgTensor, const std::string &name) {
  // std::cout << "Recording at " << name << std::endl;
  CHECK_TRUE(imgTensor.dtype() == at::kByte);
  CHECK_EQ(imgTensor.size(0), 3);
  const auto img = imgTensor.permute({1, 2, 0}).cpu().contiguous();
  CHECK_EQ(img.stride(1), 3);
  CHECK_EQ(img.stride(2), 1);
  auto cvImg = cv::Mat(img.size(0), img.size(1), CV_8UC3, img.data_ptr(), img.stride(0));
  cv::Mat cvImgBgr;
  cv::cvtColor(cvImg, cvImgBgr, cv::COLOR_RGB2BGR);
  return cv::imwrite(name.c_str(), cvImgBgr);
}

cv::Mat1b wrapCvFromTensor1b(const at::Tensor img) {
  CHECK_EQ(img.dim(), 2);
  CHECK_TRUE(img.dtype().isScalarType(torch::kByte));
  CHECK_EQ(img.stride(1), 1);
  return cv::Mat1b(img.size(0), img.size(1), reinterpret_cast<uchar *>(img.data_ptr()), img.stride(0));
}

cv::Mat3b wrapCvFromTensor3b(const at::Tensor img) {
  CHECK_EQ(img.dim(), 3);
  CHECK_TRUE(img.dtype().isScalarType(torch::kByte));
  CHECK_EQ(img.stride(1), 3);
  return cv::Mat3b(img.size(0), img.size(1), reinterpret_cast<cv::Vec3b *>(img.data_ptr()), img.stride(0));
}

int parseInt(std::string_view data) {
  int out;
  const auto result = std::from_chars(data.data(), data.data() + data.size(), out);
  if (result.ec == std::errc::invalid_argument || result.ec == std::errc::result_out_of_range) {
    throw std::invalid_argument(std::format("parseInt cannot parse: {}", data));
  }
  return out;
}

} // namespace zoo