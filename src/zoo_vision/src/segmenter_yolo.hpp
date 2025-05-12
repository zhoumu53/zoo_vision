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

#include "zoo_vision/segmenter_interface.hpp"
#include "zoo_vision/timings.hpp"

#include <c10/cuda/CUDAStream.h>
#include <nlohmann/json_fwd.hpp>
#include <rclcpp/rclcpp.hpp>

#include <filesystem>

class YOLOv11SegDetector;

namespace zoo {

class SegmenterYolo : public ISegmenter {
public:
  explicit SegmenterYolo(int nameIndex, std::string cameraName, at::cuda::CUDAStream cudaStream);
  ~SegmenterYolo();

  void readConfig(const nlohmann::json &config);
  void loadModel(const std::filesystem::path &modelPath);

  void setImageSize(Vector2i size) override { detectionImageSize_ = size; }
  Vector2i getDetectionImageSize() const override { return detectionImageSize_; }

  void onImage(SegmenterResult &result, const at::Tensor &imageGpu,
               const cv::Mat &imageCpu /*TODO: remove imageCpu*/) override;

private:
  const rclcpp::Logger &get_logger() const { return logger_; }

  struct SegmentationResult {
    at::Tensor masks_u8;
    at::Tensor boxes;
    at::Tensor scores;
    at::Tensor labels;
  };

  SegmentationResult callYolo(const cv::Mat &image);

  std::string name_;
  rclcpp::Logger logger_;
  at::cuda::CUDAStream cudaStream_;

  std::string cameraName_;

  float32_t scoreThreshold_;

  std::unique_ptr<YOLOv11SegDetector> model_;
  Vector2i detectionImageSize_;
};
} // namespace zoo