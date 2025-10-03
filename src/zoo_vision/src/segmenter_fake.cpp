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

#include "zoo_vision/segmenter_fake.hpp"

#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/utils.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>

#include <algorithm>
#include <chrono>
#include <string.h>

using namespace std::chrono_literals;
using namespace at::indexing;

namespace zoo {

SegmenterFake::SegmenterFake(int nameIndex)
    : name_{std::format("segmenter_fake_{}", nameIndex)}, logger_{rclcpp::get_logger(name_)} {
  RCLCPP_INFO(get_logger(), "Starting %s", name_.c_str());
}

SegmenterFake::~SegmenterFake() = default;

void SegmenterFake::onImage(SegmenterResult &result, const at::Tensor & /*imageGpu*/, const cv::Mat & /*imageCpu*/) {
  const float size_factor = 0.1f;
  const Vector2f size = size_factor * detectionImageSize_.cast<float>();
  const Vector2f min = {0.4f * detectionImageSize_.x(), 0.4f * detectionImageSize_.y()};
  const Vector2f max = min + size;

  const AlignedBox2f bbox{min, max};
  result.bboxesInDetection.push_back(bbox);

  cv::Mat1b maskMap = wrapCvFromTensor1b(result.masks[0]);
  const Vector2i center = ((min + max) / 2).cast<int>();
  cv::circle(maskMap, {center.x(), center.y()}, size.minCoeff(), {1, 1, 1, 1}, cv::FILLED);
}

} // namespace zoo
