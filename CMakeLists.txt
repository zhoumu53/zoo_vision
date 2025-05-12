cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(zoo_vision LANGUAGES CUDA CXX)

# Download dependencies
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
)
FetchContent_MakeAvailable(Corrosion)

# Components
add_subdirectory(src/zoo_msgs)

configure_file("${CMAKE_BINARY_DIR}/.cargo/config.toml" "${CMAKE_BINARY_DIR}/src/zoo_vision_rs/.cargo/config.toml" COPYONLY)

corrosion_import_crate(MANIFEST_PATH src/zoo_vision_rs//Cargo.toml)
add_dependencies(_cargo-build_zoo_vision_rs  zoo_msgs)
# target_link_libraries(zoo_vision_rs INTERFACE zoo_msgs)

add_subdirectory(src/zoo_vision)