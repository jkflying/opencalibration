#! /bin/bash

set -ex
sudo apt update || (apt-get update && apt-get install sudo)
sudo apt upgrade -y
DEBIAN_FRONTEND=noninteractive sudo apt install -y cmake ccache ninja-build libeigen3-dev libopencv-dev libgdal-dev libceres-dev libgtest-dev rapidjson-dev libspdlog-dev libtinyxml2-dev clang-format-11 lcov g++-11 build-essential
