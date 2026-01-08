cmake_minimum_required(VERSION 3.24)

project(zoo_grafana)

# Configure launcher
# TODO: this adds an app shortcut whenever the project is configured.
#       Maybe we should change this to build or install time?
configure_file(start_track_images_server.sh.in ${CMAKE_BINARY_DIR}/start_track_images_server.sh)
