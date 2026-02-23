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

#include "zoo_vision/video_db_loader.hpp"

#include "zoo_vision/utils.hpp"

#include <date/chrono_io.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>

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

VideoDBLoader::VideoDBLoader(const rclcpp::NodeOptions &options)
    : Node("video_db_loader", options), profileTic_{"VidoeDBLoader::tic"} {
  RCLCPP_INFO(get_logger(), "Starting video_db_loader");

  const nlohmann::json &config = getConfig();

  skipFrameCount_ = config["skip_frame_count"].get<int>();

  // Load database
  std::vector<std::string> enabledCameras = config["enabled_cameras"];
  std::filesystem::path videoDatabase = config["video_db"];
  if (config.contains("video_root") && !config["video_root"].is_null()) {
    const std::string videoRootStr = config["video_root"].get<std::string>();
    if (!videoRootStr.empty()) {
      videoRootPath_ = videoRootStr;
    }
  }
  if (videoRootPath_.empty()) {
    videoRootPath_ = videoDatabase.parent_path();
  }
  RCLCPP_INFO(get_logger(), "Using video root: %s", videoRootPath_.string().c_str());
  loadVideoDatabase(videoDatabase, enabledCameras);

  // Advance replay clock
  replayNow_ = parseTime(config["replay_time"].get<std::string>());
  RCLCPP_INFO(get_logger(), "Replay start time: %s", std::format("{:%Y-%m-%d %T}", replayNow_).c_str());

  // Load videos
  const auto videoQoS =
      rclcpp::QoS(rclcpp::KeepLast{ImageRateLimiter::MAX_QUEUE_SIZE}).durability_volatile().reliable();
  for (auto &[cameraName, cameraData] : cameras_) {
    cameraData.publisher_ = rclcpp::create_publisher<zoo_msgs::msg::Image12m>(*this, cameraName + "/image", videoQoS);
    loadVideo(cameraName, cameraData, replayNow_);
  }

  timerCbGroup_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  timer_ = create_wall_timer(1ms, [this]() { this->onTimer(); }, timerCbGroup_);
}

void VideoDBLoader::loadVideoDatabase(const std::filesystem::path &database,
                                      const std::span<const std::string> enabledCameras) {
  std::ifstream f(database);
  if (f.fail()) {
    using namespace std::literals;
    throw std::runtime_error("Could not open video database file: "s + database.string());
  }
  nlohmann::json databaseJson = nlohmann::json::parse(f);

  // Set replay clock to database start time
  const std::string databaseStartTime = databaseJson["start_time"];
  replayNow_ = parseTime(databaseStartTime);

  // Read all videos
  for (const std::string &camera : enabledCameras) {
    nlohmann::json cameraJson = databaseJson["cameras"][camera];

    const auto pair = cameras_.emplace(std::make_pair(camera, CameraData()));
    CameraData &cameraData = pair.first->second;
    cameraData.rateLimiter = gCameraLimiters[camera].get();

    for (auto [videoJson, startTimeJson, endTimeJson] :
         std::ranges::views::zip(cameraJson["videos"], cameraJson["start_times"], cameraJson["end_times"])) {
      cameraData.videoList_.emplace_back(videoRootPath_ / videoJson.get<std::string>(),
                                         parseTime(startTimeJson.get<std::string>()),
                                         parseTime(endTimeJson.get<std::string>()));
    }
  }
}

void VideoDBLoader::openVideo(const std::string & /*cameraName*/, CameraData &cameraData, const VideoInfo &info) {
  cv::VideoCapture cvVideo;
  const bool ok = cvVideo.open(info.videoFile);
  if (ok) {
    cameraData.frameSize = cv::Size2i{static_cast<int>(cvVideo.get(cv::CAP_PROP_FRAME_WIDTH)),
                                      static_cast<int>(cvVideo.get(cv::CAP_PROP_FRAME_HEIGHT))};
    cameraData.videoStartTime_ = info.startTime;
    cameraData.videoStream_ = std::move(cvVideo);
    RCLCPP_INFO(get_logger(), "Loaded video %s", info.videoFile.c_str());
    RCLCPP_INFO(get_logger(), "Resolution=%dx%d, now=%s, start time=%s", cameraData.frameSize.width,
                cameraData.frameSize.height, std::format("{:%Y-%m-%d %T}", replayNow_).c_str(),
                std::format("{:%Y-%m-%d %T}", *cameraData.videoStartTime_).c_str());

    const float32_t videoFps = cameraData.videoStream_->get(cv::CAP_PROP_FPS);
    const float32_t replayFps = videoFps / (skipFrameCount_ + 1);
    RCLCPP_INFO(get_logger(), "Video FPS=%f, serving frames at FPS=%f", videoFps, replayFps);

    // Adjust time
    const int64_t offsetMs = std::chrono::duration_cast<std::chrono::milliseconds>(replayNow_ - info.startTime).count();
    if (offsetMs > 0) {
      RCLCPP_INFO(get_logger(), "Advancing video by %ldms", offsetMs);
      cameraData.videoStream_->set(cv::CAP_PROP_POS_MSEC, offsetMs);
    }

  } else {
    RCLCPP_ERROR(get_logger(), "Failed to open video %s", info.videoFile.c_str());
  }
}
void VideoDBLoader::loadVideo(const std::string &cameraName, CameraData &cameraData, const Clock::time_point time) {
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
      if (time < it->endTime) {
        videoIt = it;
        break;
      }
    }
  }
  if (videoIt != cameraData.videoList_.end()) {
    cameraData.currentVideo_ = videoIt;
    openVideo(cameraName, cameraData, *videoIt);
  } else {
    RCLCPP_ERROR(get_logger(), "No video found for camera %s at time %s", cameraName.c_str(),
                 std::format("{}", time).c_str());
  }
}
void VideoDBLoader::loadNextVideo(const std::string &cameraName, CameraData &cameraData) {
  if (cameraData.currentVideo_ != cameraData.videoList_.end()) {
    cameraData.currentVideo_++;
  }
  if (cameraData.currentVideo_ == cameraData.videoList_.end()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "No more videos for camera %s.", cameraName.c_str());
    cameraData.videoStartTime_.reset();
    cameraData.videoStream_.reset();
    return;
  }

  openVideo(cameraName, cameraData, *cameraData.currentVideo_);
}

cv::Mat3b VideoDBLoader::loadImage(CameraData &cameraData, zoo_msgs::msg::Image12m &msg) {

  if (!cameraData.videoStartTime_.has_value() || *cameraData.videoStartTime_ > replayNow_) {
    return cv::Mat3b{};
  }

  assert(cameraData.videoStream_.has_value());

  msg.header.stamp =
      rclcpp::Time(std::chrono::duration_cast<std::chrono::nanoseconds>(replayNow_.time_since_epoch()).count());
  setMsgString(msg.encoding, "rgb8");
  msg.width = cameraData.frameSize.width;
  msg.height = cameraData.frameSize.height;
  msg.is_bigendian = false;
  msg.step = msg.width * 3 * sizeof(char);

  cv::Mat3b image = wrapMat3bFromMsg(msg);

  auto &cvVideo = *cameraData.videoStream_;

  // Retry fetch a few times because some videos have encoding errors
  constexpr int MAX_TRIES = 5;
  int tries = 0;
  bool ok = false;
  while (tries < MAX_TRIES && !ok) {
    ok = cvVideo.read(image);
    tries++;
  }

  // If we didn't get any image we are probably at the end of the video
  if (!ok) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "Video for %s EOF",
                         cameraData.currentVideo_->videoFile.c_str());
    cameraData.videoStartTime_.reset();
    cameraData.videoStream_.reset();
    return cv::Mat3b{};
  }

  CHECK_PTR_EQ(&msg.data[0], image.data);
  return image;
}

auto VideoDBLoader::findNextValidReplayTime() const -> std::optional<Clock::time_point> {
  std::optional<Clock::time_point> minTime;
  for (const auto &[_, cameraData] : cameras_) {
    if (cameraData.videoStartTime_.has_value()) {
      if (!minTime.has_value() || *minTime > *cameraData.videoStartTime_) {
        minTime.emplace(*cameraData.videoStartTime_);
      }
    }
  }
  return minTime;
}

void VideoDBLoader::onTimer() {
  ProfileStackGuard stackGuard{profilerStack_};
  profileTic_.tic();
  ProfileSection s{"onTimer"};

  std::optional<Clock::time_point> newReplayTime;

  int imageCount = 0;

  for (auto &[cameraName, cameraData] : cameras_) {
    // Try to load the next video if we are not at the end of the list
    if (!cameraData.videoStream_.has_value() && cameraData.currentVideo_ != cameraData.videoList_.end()) {
      loadNextVideo(cameraName, cameraData);
    }
    if (!cameraData.videoStream_.has_value()) {
      // No video to load, ignore camera from now on
      continue;
    }

    auto msg = std::make_unique<zoo_msgs::msg::Image12m>();

    cv::Mat3b image;
    {
      ProfileSection s{"loadImage"};

      image = loadImage(cameraData, *msg);
      if (!cameraData.videoStream_.has_value()) {
        loadNextVideo(cameraName, cameraData);
        image = loadImage(cameraData, *msg);
        if (image.empty()) {
          RCLCPP_ERROR(get_logger(), "%s: Loading image failed from new video", cameraName.c_str());
        }
      }
    }
    if (image.empty()) {
      continue;
    }
    if (cameraData.videoStream_.has_value()) {
      ProfileSection s{"skipFrames"};
      for (int i = 0; i < skipFrameCount_; ++i) {
        cameraData.videoStream_->grab();
      }
    }
    {
      const auto frameIndex = static_cast<uint64_t>(cameraData.videoStream_->get(cv::CAP_PROP_POS_FRAMES) - 1);
      const std::filesystem::path videoFile = cameraData.currentVideo_->videoFile;
      const std::string videoName = videoFile.stem();
      setMsgString(msg->header.video_filename, videoName);
      msg->header.frame_id = frameIndex;
    }

    if (!newReplayTime.has_value() && cameraData.videoStream_.has_value()) {
      // Calculate replay time based on how much we've advanced in the video
      const auto offsetMs = cameraData.videoStream_->get(cv::CAP_PROP_POS_MSEC);
      newReplayTime = *cameraData.videoStartTime_ + std::chrono::milliseconds(static_cast<int64_t>(offsetMs));
      // assert(newReplayTime.value() >= replayNow_);
    }

    // TODO: converting BGR->RGB like this is inefficient!
    {
      ProfileSection s{"cvtColor"};
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }
    {
      ProfileSection s{"publish"};
      CHECK_PTR_EQ(&msg->data[0], image.data);
      cameraData.publisher_->publish(std::move(msg));
      cameraData.rateLimiter->addToQueue();
    }
    imageCount++;
  }
  imageCountStats_.push(static_cast<float32_t>(imageCount));

  if (!newReplayTime.has_value()) {
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
  replayNow_ = *newReplayTime;

  // Keep track of wall time vs replay time to print how fast we are going.
  static std::optional<Clock::time_point> gLastLogReplayTime{};
  static std::optional<std::chrono::system_clock::time_point> gLastLogSystemTime{};
  {
    if (gLastLogReplayTime.has_value()) {
      constexpr auto LOG_INTERVAL = std::chrono::seconds(60);
      const auto replayDuration =
          std::chrono::duration_cast<std::chrono::milliseconds>(replayNow_ - *gLastLogReplayTime);
      if (replayDuration > LOG_INTERVAL) {
        const auto systemDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now() - *gLastLogSystemTime);
        const float32_t speedFactor =
            static_cast<float32_t>(replayDuration.count()) / static_cast<float32_t>(systemDuration.count());
        RCLCPP_INFO(get_logger(),
                    std::format("Replay time: {}, replay speed: {:.1f}x, mean camera count per frame: {:.1}",
                                replayNow_, speedFactor, imageCountStats_.mean())
                        .c_str());
        imageCountStats_.clear();
        gLastLogReplayTime = replayNow_;
        gLastLogSystemTime = std::chrono::system_clock::now();
      }
    } else {
      gLastLogReplayTime = replayNow_;
      gLastLogSystemTime = std::chrono::system_clock::now();
      RCLCPP_INFO(get_logger(),
                  std::format("Replay time at start: {}, image count: {}", replayNow_, imageCount).c_str());
    }
  }

  // Wait for processing after all images have been published
  {
    ProfileSection s{"rateLimiter"};
    for (auto &[cameraName, cameraData] : cameras_) {
      cameraData.rateLimiter->waitForProcessing();
    }
  }
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::VideoDBLoader)
