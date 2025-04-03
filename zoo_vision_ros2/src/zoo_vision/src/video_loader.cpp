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

#include "zoo_vision/video_loader.hpp"

#include "zoo_vision/utils.hpp"

#include <date/chrono_io.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <chrono>
#include <fstream>
#include <spanstream>

using namespace std::chrono_literals;

namespace zoo {
namespace {

std::chrono::system_clock::time_point parseTime(std::string_view timeStr) {
  std::ispanstream ss(timeStr);

  constexpr auto DATE_FORMAT = "%Y-%m-%dT%H:%M:%S";
  std::chrono::system_clock::time_point time_point;
  ss >> date::parse(DATE_FORMAT, time_point);
  return time_point;
}

// std::chrono::system_clock::duration parseDuration(std::string_view timeStr) {
//   using Clock = std::chrono::system_clock;
//   std::tm t = {};
//   std::ispanstream ss(timeStr);

//   constexpr auto DATE_FORMAT = "%H:%M";
//   ss >> std::get_time(&t, DATE_FORMAT);

//   Clock::duration dur = std::chrono::hours(t.tm_hour) + std::chrono::minutes(t.tm_min);
//   return dur;
// }

} // namespace

VideoLoader::VideoLoader(const rclcpp::NodeOptions &options) : Node("video_loader", options) {
  RCLCPP_INFO(get_logger(), "Starting video_loader");

  const nlohmann::json &config = getConfig();

  // Load database
  std::vector<std::string> enabledCameras = config["enabled_cameras"];
  std::filesystem::path videoDatabase = config["video_db"];
  loadVideoDatabase(videoDatabase, enabledCameras);

  // Advance replay clock
  replayNow_ = parseTime(config["replay_time"].get<std::string>());
  RCLCPP_INFO(get_logger(), "Replay start time: %s", std::format("{:%Y-%m-%d %T}", replayNow_).c_str());

  // Load videos
  for (auto &[cameraName, cameraData] : cameras_) {
    cameraData.publisher_ = rclcpp::create_publisher<zoo_msgs::msg::Image12m>(*this, cameraName + "/image", 10);
    loadVideo(cameraName, cameraData, replayNow_);
  }

  timer_ = create_wall_timer(40ms, [this]() { this->onTimer(); });
}

void VideoLoader::loadVideoDatabase(const std::filesystem::path &database,
                                    const std::span<const std::string> enabledCameras) {
  const auto videoRootPath = database.parent_path();

  std::ifstream f(database);
  nlohmann::json databaseJson = nlohmann::json::parse(f);

  // Set replay clock to database start time
  const std::string databaseStartTime = databaseJson["start_time"];
  replayNow_ = parseTime(databaseStartTime);

  // Read all videos
  for (const std::string_view camera : enabledCameras) {
    nlohmann::json cameraJson = databaseJson["cameras"][camera];

    const auto pair = cameras_.emplace(std::make_pair(camera, CameraData()));
    CameraData &cameraData = pair.first->second;

    for (auto [videoJson, startTimeJson, endTimeJson] :
         std::ranges::views::zip(cameraJson["videos"], cameraJson["start_times"], cameraJson["end_times"])) {
      cameraData.videoList_.emplace_back(videoRootPath / videoJson.get<std::string>(),
                                         parseTime(startTimeJson.get<std::string>()),
                                         parseTime(endTimeJson.get<std::string>()));
    }
  }
}

void VideoLoader::loadVideo(const std::string &cameraName, CameraData &cameraData, const Clock::time_point time) {
  if (cameraData.videoList_.empty()) {
    RCLCPP_WARN(get_logger(), "Video list for %s is empty.", cameraName.c_str());
    return;
  }

  auto videoIt = cameraData.videoList_.end();
  if (cameraData.videoList_.front().startTime > time) {
    // First video starts in the future. Open this one.
    videoIt = cameraData.videoList_.begin();
  } else {
    for (auto it = cameraData.videoList_.begin(); it != cameraData.videoList_.end(); ++it) {
      if (it->startTime <= time && time < it->endTime) {
        videoIt = it;
        break;
      }
    }
  }
  if (videoIt != cameraData.videoList_.end()) {
    const auto &videoFile = videoIt->videoFile;
    const auto &startTime = videoIt->startTime;

    cv::VideoCapture cvVideo;
    const bool ok = cvVideo.open(videoFile);
    if (ok) {
      cameraData.frameSize = cv::Size2i{static_cast<int>(cvVideo.get(cv::CAP_PROP_FRAME_WIDTH)),
                                        static_cast<int>(cvVideo.get(cv::CAP_PROP_FRAME_HEIGHT))};
      cameraData.videoStartTime_ = startTime;
      cameraData.videoStream_ = std::move(cvVideo);
      RCLCPP_INFO(get_logger(), "Loaded video %s", videoFile.c_str());
      RCLCPP_INFO(get_logger(), "Resolution=%dx%d, now=%s, start time=%s", cameraData.frameSize.width,
                  cameraData.frameSize.height, std::format("{:%Y-%m-%d %T}", time).c_str(),
                  std::format("{:%Y-%m-%d %T}", *cameraData.videoStartTime_).c_str());

      // Adjust time
      const int64_t offsetMs = std::chrono::duration_cast<std::chrono::milliseconds>(time - startTime).count();
      if (offsetMs > 0) {
        RCLCPP_INFO(get_logger(), "Advancing video by %ldms", offsetMs);
        cameraData.videoStream_->set(cv::CAP_PROP_POS_MSEC, offsetMs);
      }

    } else {
      RCLCPP_ERROR(get_logger(), "Failed to open video %s", videoFile.c_str());
    }
  } else {
    RCLCPP_ERROR(get_logger(), "No video found for camera %s at time %s", cameraName.c_str(),
                 std::format("{}", time).c_str());
  }
}

auto VideoLoader::findNextValidReplayTime() const -> std::optional<Clock::time_point> {
  std::optional<Clock::time_point> bestTime;

  for (const auto &[_, cameraData] : cameras_) {
    for (const auto &videoInfo : cameraData.videoList_) {
      if (videoInfo.startTime > replayNow_) {
        if (!bestTime.has_value() || *bestTime > videoInfo.startTime) {
          bestTime = videoInfo.startTime;
        }
        break;
      }
    }
  }
  return bestTime;
}

void VideoLoader::loadImage(CameraData &cameraData, cv::Mat3b &image) {
  if (!cameraData.videoStartTime_.has_value() || *cameraData.videoStartTime_ > replayNow_) {
    return;
  }

  assert(cameraData.videoStream_.has_value());
  auto &cvVideo = *cameraData.videoStream_;

  cvVideo >> image;
}

void VideoLoader::onTimer() {
  std::optional<Clock::time_point> newReplayTime;
  bool framePublished = false;

  for (auto &[cameraName, cameraData] : cameras_) {
    if (!cameraData.videoStream_.has_value()) {
      // Load video before we start
      loadVideo(cameraName, cameraData, replayNow_);
      if (!cameraData.videoStream_.has_value()) {
        // No video can be loaded for this camera
        continue;
      }
    }

    auto msg = std::make_unique<zoo_msgs::msg::Image12m>();
    msg->header.stamp =
        rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(replayNow_.time_since_epoch()).count());
    setMsgString(msg->header.frame_id, std::to_string(frameIndex_).c_str());
    setMsgString(msg->encoding, sensor_msgs::image_encodings::RGB8);
    msg->width = cameraData.frameSize.width;
    msg->height = cameraData.frameSize.height;
    msg->is_bigendian = false;
    msg->step = msg->width * 3 * sizeof(char);

    cv::Mat3b image = wrapMat3bFromMsg(*msg);
    loadImage(cameraData, image);
    if (image.empty()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "Video for %s EOF", cameraName.c_str());

      cameraData.videoStartTime_.reset();
      cameraData.videoStream_.reset();
      loadVideo(cameraName, cameraData, replayNow_);
      loadImage(cameraData, image);
      if (image.empty()) {
        RCLCPP_ERROR(get_logger(), "%s: Loading image failed from new video", cameraName.c_str());
      }
    }
    if (image.empty()) {
      continue;
    }

    RCLCPP_INFO(get_logger(), "%s: Loading image success at %s", cameraName.c_str(),
                std::format("{}", replayNow_).c_str());

    if (!newReplayTime.has_value() && cameraData.videoStream_.has_value()) {
      // Calculate replay time based on how much we've advanced in the video
      const auto offsetMs = cameraData.videoStream_->get(cv::CAP_PROP_POS_MSEC);
      newReplayTime = *cameraData.videoStartTime_ + std::chrono::milliseconds(static_cast<int64_t>(offsetMs));
      assert(newReplayTime.value() >= replayNow_);
    }

    // TODO: converting BGR->RGB like this is inefficient!
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cameraData.publisher_->publish(std::move(msg));
    framePublished = true;
  }
  if (framePublished) {
    frameIndex_ += 1;
  }

  if (!newReplayTime.has_value()) {
    assert(framePublished == false);
    RCLCPP_WARN(get_logger(), "No images produced");
    newReplayTime = findNextValidReplayTime();
    if (newReplayTime.has_value()) {
      RCLCPP_INFO(get_logger(), "No images produced, advancing to %s", std::format("{}", *newReplayTime).c_str());
    } else {
      RCLCPP_INFO(get_logger(), "End of all videos");
      std::terminate();
    }
  }

  assert(newReplayTime.has_value());
  RCLCPP_INFO(get_logger(), "newReplayTime=%s", std::format("{}", *newReplayTime).c_str());
  replayNow_ = *newReplayTime;

  static int64_t minutesLastLog = 0;
  const int64_t minutesNow = std::chrono::duration_cast<std::chrono::minutes>(replayNow_.time_since_epoch()).count();
  if (abs(minutesNow - minutesLastLog) > 5) {
    minutesLastLog = minutesNow;
    RCLCPP_INFO(get_logger(), "Replay time: %s", std::format("{}", replayNow_).c_str());
  }
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::VideoLoader)