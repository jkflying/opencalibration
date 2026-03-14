#! /bin/bash

set -ex
sudo apt-get update || (apt-get update && apt-get install sudo)
sudo apt-get install -y cmake ccache ninja-build libeigen3-dev libgdal-dev libceres-dev libgtest-dev rapidjson-dev libspdlog-dev libtinyxml2-dev clang-format-14 clang-tidy-18 clang-18 libomp-18-dev lcov g++-11 build-essential libunwind-dev git libmimalloc-dev libopencv-dev

rm -rf external/venv/
python3 -m venv external/venv
external/venv/bin/pip install fastcov cltcache

rm -rf external/install/
