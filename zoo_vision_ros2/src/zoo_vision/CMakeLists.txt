cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(zoo_vision LANGUAGES CUDA CXX)

include(FetchContent)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(zoo_msgs REQUIRED)
# find_package(zoo_vision_rs REQUIRED)

find_package( OpenCV 4.6.0 EXACT REQUIRED )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Download all deps
FetchContent_Declare(TorchRelease URL https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_Declare(nvtx3 URL https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.1.0.tar.gz)
FetchContent_Declare(argparse GIT_REPOSITORY https://github.com/p-ranav/argparse.git)
FetchContent_Declare(date GIT_REPOSITORY https://github.com/HowardHinnant/date.git)
FetchContent_MakeAvailable(TorchRelease json nvtx3 argparse date)

# torch
list(APPEND CMAKE_PREFIX_PATH "${torchrelease_SOURCE_DIR}")
find_package(Torch REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/home/dherrera/install/vision")
find_package(TorchVision REQUIRED)

# nvtext
include("${nvtx3_SOURCE_DIR}/c/nvtxImportedTargets.cmake")

# zoo_vision_rs
add_library(zoo_vision_rs STATIC IMPORTED)
set_target_properties(zoo_vision_rs PROPERTIES 
  IMPORTED_LOCATION_DEBUG   /home/dherrera/git/zoo_vision/zoo_vision_ros2/build/zoo_vision_rs/debug/libzoo_vision_rs.a
  IMPORTED_LOCATION_RELEASE /home/dherrera/git/zoo_vision/zoo_vision_ros2/build/zoo_vision_rs/release/libzoo_vision_rs.a
  IMPORTED_LOCATION_RELWITHDEBINFO /home/dherrera/git/zoo_vision/zoo_vision_ros2/build/zoo_vision_rs/release/libzoo_vision_rs.a)

####################################
# zoo_vision package
add_executable(zoo_vision 
  src/types.hpp
  src/utils.hpp
  src/utils.cpp
  src/stats.hpp
  src/stats.cpp
  src/replay_clock.hpp
  src/json_eigen.hpp
  src/json_eigen.cpp
  src/zoo_camera.hpp
  src/zoo_camera.cpp
  src/image_normalizer.hpp
  src/image_normalizer.cpp
  src/image_embedder.hpp
  src/image_embedder.cpp
  src/video_loader.hpp
  src/video_loader.cpp
  src/segmenter.hpp
  src/segmenter.cpp
  src/patch_cropper.hpp
  src/patch_cropper.cpp
  src/vote_histogram.hpp
  src/vote_histogram.cpp
  src/identifier.hpp
  src/identifier.cpp
  src/behaviourer.hpp
  src/behaviourer.cpp
  src/keyframe_store.hpp
  src/keyframe_store.cpp
  src/image_queue.hpp
  src/image_queue.cpp
  src/rerun_forwarder.hpp
  src/rerun_forwarder.cpp
  src/track_matcher.hpp
  src/track_matcher.cpp
  src/camera_pipeline.hpp
  src/camera_pipeline.cpp
  src/main.cpp
)

target_include_directories(zoo_vision PUBLIC include)
target_compile_features(zoo_vision PUBLIC cxx_std_23)

ament_target_dependencies(zoo_vision
  rclcpp
  rclcpp_components
  image_transport
  cv_bridge
  OpenCV
  zoo_msgs
)
target_link_libraries(zoo_vision 
  Eigen3::Eigen torch
  -Wl,--no-as-needed TorchVision::TorchVision -Wl,--as-needed 
  nlohmann_json::nlohmann_json
  nvtx3-cpp
  zoo_vision_rs
  argparse
  date
)

install(TARGETS
  zoo_vision
  DESTINATION lib/zoo_vision)

####################################

####################################
# zoo_rerun package
# add_executable(zoo_rerun
#   # Headers
#   include/zoo_vision/rerun_forwarder.hpp
#   # Sources
#   src/rerun_forwarder.cpp
#   src/main2.cpp)

# target_include_directories(zoo_rerun PUBLIC include)
# target_compile_features(zoo_rerun PUBLIC cxx_std_20)

# ament_target_dependencies(zoo_rerun
#   rclcpp
#   rclcpp_components
#   image_transport
#   zoo_msgs
#   cv_bridge
#   OpenCV
# )
# # target_link_libraries(zoo_vision2 rerun_sdk)
# target_link_libraries(zoo_rerun zoo_vision_rs)

# install(TARGETS
#   zoo_rerun
#   DESTINATION lib/zoo_vision)


####################################

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
