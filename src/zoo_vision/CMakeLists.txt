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

find_package( OpenCV 4.6.0 EXACT REQUIRED )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Download all deps
FetchContent_Declare(onnxruntime_release URL https://github.com/microsoft/onnxruntime/releases/download/v1.21.1/onnxruntime-linux-x64-gpu-1.21.1.tgz)
FetchContent_Declare(TorchRelease URL https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_Declare(nvtx3 URL https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.1.0.tar.gz)
FetchContent_Declare(argparse GIT_REPOSITORY https://github.com/p-ranav/argparse.git)
FetchContent_Declare(date GIT_REPOSITORY https://github.com/HowardHinnant/date.git)
FetchContent_Declare(otl URL https://otl.sourceforge.net/otlv4_h2.zip)
FetchContent_Declare(bytetrack_cpp GIT_REPOSITORY git@github.com:Vertical-Beach/ByteTrack-cpp.git GIT_TAG main)

FetchContent_MakeAvailable(onnxruntime_release TorchRelease json nvtx3 argparse date otl bytetrack_cpp)

# yolos-cpp asks for a very old cmake so the ONNXRUNTIME_DIR gets cleared
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW) 
set(ONNXRUNTIME_DIR ${onnxruntime_release_SOURCE_DIR})
FetchContent_Declare(yolos-cpp GIT_REPOSITORY https://github.com/plumonito/YOLOs-CPP GIT_TAG feature/header-cmake-target)
FetchContent_MakeAvailable(yolos-cpp)

# onnx
# Bugfix: there are cmake files that could produce the onnxruntime target with
#   list(APPEND CMAKE_PREFIX_PATH "${onnxruntime_release_SOURCE_DIR}")
#   find_package(onnxruntime REQUIRED)
# But the cmake files in the release package are broken. Ignore and create our own imported targets.
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
target_include_directories(onnxruntime::onnxruntime INTERFACE ${onnxruntime_release_SOURCE_DIR}/include)
set_target_properties(onnxruntime::onnxruntime PROPERTIES 
  IMPORTED_LOCATION ${onnxruntime_release_SOURCE_DIR}/lib/libonnxruntime.so)

# torch
list(APPEND CMAKE_PREFIX_PATH "${torchrelease_SOURCE_DIR}")
find_package(Torch REQUIRED)

# TorchVision was installed by install_deps.sh. Find it from home dir.
file(REAL_PATH "~" HOME EXPAND_TILDE)
list(APPEND CMAKE_PREFIX_PATH "${HOME}/install/vision")
find_package(TorchVision REQUIRED)

# nvtext
include("${nvtx3_SOURCE_DIR}/c/nvtxImportedTargets.cmake")

####################################
# zoo_vision package
add_executable(zoo_vision 
  src/types.hpp
  src/compute_device.hpp
  src/compute_device.cpp
  src/utils.hpp
  src/utils.cpp
  src/timings.hpp
  src/stats.hpp
  src/stats.cpp
  src/replay_clock.hpp
  src/json_eigen.hpp
  src/json_eigen.cpp
  src/zoo_camera.hpp
  src/zoo_camera.cpp
  src/camera_calibration.hpp
  src/camera_calibration.cpp
  src/image_normalizer.hpp
  src/image_normalizer.cpp
  src/image_embedder.hpp
  src/image_embedder.cpp
  src/image_quality.hpp
  src/image_quality.cpp
  src/image_rate_limiter.hpp
  src/image_rate_limiter.cpp
  src/video_db_loader.hpp
  src/video_db_loader.cpp
  src/video_loader.hpp
  src/video_loader.cpp
  src/segmenter_interface.hpp
  src/segmenter_interface.cpp
  src/segmenter.hpp
  src/segmenter.cpp
  src/segmenter_fake.hpp
  src/segmenter_fake.cpp
  src/segmenter_yolo.hpp
  src/segmenter_yolo.cpp
  src/world_locator.hpp
  src/world_locator.cpp
  src/patch_cropper.hpp
  src/patch_cropper.cpp
  src/vote_histogram.hpp
  src/vote_histogram.cpp
  src/identifier_interface.hpp
  src/identifier_interface.cpp
  src/identifier_fake.hpp
  src/identifier_fake.cpp
  src/identifier.hpp
  src/identifier.cpp
  src/behaviourer_interface.hpp
  src/behaviourer_interface.cpp
  src/behaviourer_fake.hpp
  src/behaviourer_fake.cpp
  src/behaviourer.hpp
  src/behaviourer.cpp
  src/keyframe_store.hpp
  src/keyframe_store.cpp
  src/image_queue.hpp
  src/image_queue.cpp
  src/rerun_forwarder.hpp
  src/rerun_forwarder.cpp
  src/db_forwarder.hpp
  src/db_forwarder.cpp
  src/track_count_recorder.hpp
  src/track_count_recorder.cpp
  src/track_matcher.hpp
  src/track_matcher.cpp
  src/track_writer.hpp
  src/track_writer.cpp
  src/camera_pipeline.hpp
  src/camera_pipeline.cpp
  src/video_writer.hpp
  src/video_writer.cpp
  src/main.cpp
)

target_include_directories(zoo_vision PUBLIC include)
target_include_directories(zoo_vision SYSTEM PRIVATE ${otl_SOURCE_DIR})
target_compile_features(zoo_vision PUBLIC cxx_std_23)

ament_target_dependencies(zoo_vision
  rclcpp
  rclcpp_components
  builtin_interfaces
)
# add_dependencies(zoo_vision zoo_msgs)
target_link_libraries(zoo_vision 
  zoo_vision_rs
  zoo_msgs_libs
  stdc++exp
  /usr/lib/x86_64-linux-gnu/odbc/psqlodbcw.so
  Eigen3::Eigen torch
  -Wl,--no-as-needed TorchVision::TorchVision -Wl,--as-needed 
  nlohmann_json::nlohmann_json
  nvtx3-cpp
  argparse
  date
  onnxruntime::onnxruntime
  opencv_core
  opencv_highgui
  yolos_cpp
  bytetrack
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

# Configure launcher
# TODO: this adds an app shortcut whenever the project is configured.
#       Maybe we should change this to build or install time?
configure_file(launcher/run_zoo_vision.sh.in ${CMAKE_BINARY_DIR}/run_zoo_vision.sh)
configure_file(launcher/run_zoo_vision_ui.sh.in ${CMAKE_BINARY_DIR}/run_zoo_vision_ui.sh)
configure_file(launcher/zoo_vision.desktop.in $ENV{HOME}/.local/share/applications/zoo_vision.desktop)
configure_file(launcher/zoo_vision_ui.desktop.in $ENV{HOME}/.local/share/applications/zoo_vision_ui.desktop)

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
