#! /bin/bash

set -ex
sudo apt-get update
sudo apt-get install ccache ninja-build libeigen3-dev libopencv-dev libgdal-dev libceres-dev libgtest-dev rapidjson-dev libspdlog-dev libtinyxml2-dev clang-format-11 lcov
