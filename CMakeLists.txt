cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(zoo_vision LANGUAGES CUDA CXX)

set(CMAKE_COLOR_DIAGNOSTICS ON)

# Download dependencies
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
)
FetchContent_MakeAvailable(Corrosion)

# Components
add_subdirectory(src/zoo_msgs)

corrosion_import_crate(
    MANIFEST_PATH src/zoo_vision_rs/Cargo.toml
    FLAGS --config patch.crates-io.zoo_msgs.path="${CMAKE_BINARY_DIR}/src/zoo_msgs/rosidl_generator_rs/zoo_msgs/rust"
          --config patch.crates-io.builtin_interfaces.path="${builtin_interfaces_DIR}/../rust"
          --config patch.crates-io.std_msgs.path="${std_msgs_DIR}/../rust"
)
add_dependencies(_cargo-build_zoo_vision_rs  zoo_msgs)
# target_link_libraries(zoo_vision_rs INTERFACE zoo_msgs)

add_subdirectory(src/zoo_vision)