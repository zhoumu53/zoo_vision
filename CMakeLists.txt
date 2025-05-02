cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(zoo_vision LANGUAGES CUDA CXX)

include(FetchContent)

add_subdirectory(zoo_vision_ros2/src/zoo_msgs)


FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
)
FetchContent_MakeAvailable(Corrosion)
corrosion_import_crate(MANIFEST_PATH zoo_vision_ros2/src/zoo_vision_rs//Cargo.toml)

add_subdirectory(zoo_vision_ros2/src/zoo_vision)