# Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install Ros2 Jazzy
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-dev-tools ros-jazzy-desktop
sudo rosdep init
rosdep update

# Rust
sudo apt install rustup
rustup default stable

# ros2_rust
sudo apt install libclang-dev python3-vcstool
cargo install cargo-ament-build

mkdir -p ~/git/ros2_rust_ws/src
cd ~/git/ros2_rust_ws/
git clone https://github.com/ros2-rust/ros2_rust.git src/ros2_rust
vcs import src < src/ros2_rust/ros2_rust_jazzy.repos
pip3 install lark
colcon build

# Bazel (for tensorflow)
# sudo apt install apt-transport-https curl gnupg -y
# curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
# sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
# echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# sudo apt update
# sudo apt install bazel

# Torch
#curl -o ~/Downloads/libtorch-2.5.1cpu.zip https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
curl -o ~/Downloads/libtorch-2.5.1.zip https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
mkdir ~/install
unzip ~/Downloads/libtorch-2.5.1.zip -d ~/install/libtorch-2.5.1

# Torchvision
cd ~/git
git clone git@github.com:pytorch/vision.git
mkdir -p ~/build/vision
cd ~/build/vision
CXXFLAGS=-D__CUDA_NO_HALF_CONVERSIONS__,-D__CUDA_NO_BFLOAT16_CONVERSIONS__,-D__CUDA_NO_HALF2_OPERATORS__ cmake ~/git/vision -G Ninja -DWITH_CUDA=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/install/libtorch-2.5.1/libtorch/share/cmake/Torch -DCMAKE_INSTALL_PREFIX=~/install/vision -DCUDA_NVCC_FLAGS="-D__CUDA_NO_HALF_CONVERSIONS__;-D__CUDA_NO_BFLOAT16_CONVERSIONS__;-D__CUDA_NO_HALF2_OPERATORS__"
ninja install

# Sam2
cd ~/git/zoo_vision
mkdir -p models/sam2
wget -P models/sam2/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# OCR for video parsing
sudo apt install of tesseract-ocr

# Database
sudo apt install postgresql libpq-dev odbc-postgresql
