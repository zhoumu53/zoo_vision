cmake_minimum_required(VERSION 3.24)

project(zoo_msgs)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msg_files
  "msg/String.msg"
  "msg/Header.msg"
  "msg/KeyValueArrayi64.msg"
  "msg/KeyValueArrayf.msg"
  "msg/Timings.msg"
  "msg/Image8k.msg"
  "msg/Image512k.msg"
  "msg/Image1m.msg"
  "msg/Image2m.msg"
  "msg/Image4m.msg"
  "msg/Image8m.msg"
  "msg/Image12m.msg"
  "msg/Image.msg"
  "msg/BoundingBox2D.msg"
  "msg/Tensor3b32m.msg"
  "msg/Detection.msg" 
  "msg/TrackState.msg" 
  "msg/TrackClosed.msg" 
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces
  ADD_LINTER_TESTS
)

rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
if(cpp_typesupport_target)
  add_library(${PROJECT_NAME}_library INTERFACE)
  target_include_directories(${PROJECT_NAME}_library INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
  target_link_libraries(${PROJECT_NAME}_library INTERFACE
  "${cpp_typesupport_target}")

  install(
    TARGETS ${PROJECT_NAME}_library EXPORT export_${PROJECT_NAME}
  )
endif()

rosidl_get_typesupport_target(c_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_c")
if(c_typesupport_target)
  add_library(${PROJECT_NAME}_library_c INTERFACE)
  target_include_directories(${PROJECT_NAME}_library_c INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
  target_link_libraries(${PROJECT_NAME}_library_c INTERFACE
  "${c_typesupport_target}")

  install(
    TARGETS ${PROJECT_NAME}_library_c EXPORT export_${PROJECT_NAME}
  )
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_export_targets(export_${PROJECT_NAME})
ament_package()
