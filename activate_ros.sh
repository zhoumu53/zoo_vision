# Python env
source ./env/bin/activate    
# Base jazzy ros2 workspace
source /opt/ros/jazzy/setup.zsh
# ros2_rust ros2 workspace
source ~/git/ros2_rust_ws/install/setup.sh
# zoo_vision ros2 workspace
# source zoo_vision_ros2/install/setup.sh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/git/zoo_vision/zoo_vision_ros2/src/fast_dds_config.xml"
export LD_LIBRARY_PATH=~/git/zoo_vision/build/RelWithDebInfo/src/zoo_msgs:${LD_LIBRARY_PATH}
export CLICOLOR_FORCE=1 # Enable color on ninja build 