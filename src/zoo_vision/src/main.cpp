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

#include "zoo_vision/camera_pipeline.hpp"
#include "zoo_vision/compute_device.hpp"
#include "zoo_vision/db_forwarder.hpp"
#include "zoo_vision/image_rate_limiter.hpp"
#include "zoo_vision/profiler_log_node.hpp"
#include "zoo_vision/rerun_forwarder.hpp"
#include "zoo_vision/utils.hpp"
#include "zoo_vision/video_db_loader.hpp"
#include "zoo_vision/video_loader.hpp"
#include "zoo_vision/zoo_camera.hpp"

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>

std::unique_ptr<argparse::ArgumentParser> parse_args(int argc, char *argv[]) {
  auto args = std::make_unique<argparse::ArgumentParser>("zoo_vision");
  args->add_argument("-c", "--config").help("Modify configs values").append();

  try {
    args->parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << args;
    return nullptr;
  }

  return args;
}

void overrideConfig(nlohmann::json &config, const std::string_view &configOverride) {
  auto splitView = std::ranges::views::split(configOverride, '=');
  const auto split = std::vector<std::string_view>(splitView.begin(), splitView.end());
  if (split.size() != 2) {
    std::cout << "Vec " << split << std::endl;
    throw zoo::ZooVisionError(std::format("Invalid config override: {}", configOverride));
  }
  const auto &path = split[0];
  const auto &newValueStr = split[1];
  using json_pointer = nlohmann::json_pointer<std::string>;
  const auto jsonPath = json_pointer(std::string(path));

  nlohmann::json item = config[jsonPath];
  const nlohmann::json newValueJson = nlohmann::json::parse(newValueStr);
  std::cout << "Overriding " << path << ": old=" << item << ", new=" << newValueJson << std::endl;
  config[jsonPath] = newValueJson;
}

int main(int argc, char *argv[]) {
  using namespace zoo;

  rclcpp::init(argc, argv);

  setComputeDevice();

  // Load config once before initializing all nodes
  loadConfig();
  const auto args = parse_args(argc, argv);
  const auto configOverrides = args->get<std::vector<std::string>>("--config");
  std::cout << "Args: " << configOverrides << std::endl;

  auto &config = getConfig();
  for (const std::string &configOverride : configOverrides) {
    overrideConfig(config, configOverride);
  }

  std::vector<std::string> cameraNames = config["enabled_cameras"];

  rclcpp::executors::MultiThreadedExecutor exec{rclcpp::ExecutorOptions(), cameraNames.size() + 10};

  rclcpp::NodeOptions options;
  options.use_intra_process_comms(false);

  std::vector<std::shared_ptr<rclcpp::Node>> nodes;

  // Start rerun first so we can connect right away
  nodes.push_back(std::make_shared<RerunForwarder>(options));

  // Camera rate limiters
  for (const auto &cameraName : cameraNames) {
    gCameraLimiters.emplace(std::string{cameraName}, std::make_unique<ImageRateLimiter>());
  }

  // Start db node
  if (config["db"]["enabled"].get<bool>()) {
    nodes.push_back(std::make_shared<DbForwarder>(options));
  }

  const bool useLiveStream = config["live_stream"].get<bool>();
  if (!useLiveStream) {
    const std::vector<std::string> videoFiles = config["videos"];
    if (videoFiles.empty()) {
      nodes.push_back(std::make_shared<VideoDBLoader>(options));

    } else {
      nodes.push_back(std::make_shared<VideoLoader>(options));
    }
  }

  int index = 0;
  for (const auto &cameraName : cameraNames) {
    rclcpp::NodeOptions optionsCamera = options;
    optionsCamera.append_parameter_override("camera_name", cameraName);

    if (useLiveStream) {
      nodes.push_back(std::make_shared<ZooCamera>(optionsCamera, index));
    }
    nodes.push_back(std::make_shared<CameraPipeline>(optionsCamera, index));
    index += 1;
  }
  nodes.push_back(std::make_shared<ProfilerLogNode>());

  for (const auto &node : nodes) {
    exec.add_node(node);
  }

  exec.spin();
  rclcpp::shutdown();
  return 0;
}