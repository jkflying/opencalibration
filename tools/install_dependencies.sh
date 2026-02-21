#! /bin/bash

set -ex
sudo apt-get update || (apt-get update && apt-get install sudo)
sudo apt-get install -y cmake ccache ninja-build libeigen3-dev libgdal-dev libceres-dev libgtest-dev rapidjson-dev libspdlog-dev libtinyxml2-dev clang-format-14 clang-tidy-18 clang-18 libomp-18-dev lcov g++-11 build-essential libunwind-dev git

rm -rf external/venv/
python3 -m venv external/venv
external/venv/bin/pip install fastcov cltcache

rm -rf external/install/

mkdir -p external/build/opencv
rm -rf external/build/opencv/* || echo "No files found"
pushd external/build/opencv/
cmake ../../opencv -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_gapi=OFF -DBUILD_opencv_highgui=OFF -DBUILD_opencv_js_bindings_generator=OFF -DBUILD_opencv_ml=OFF -DBUILD_opencv_objdetect=OFF -DBUILD_opencv_photo=OFF -DBUILD_opencv_python_bindings_generator=OFF -DBUILD_opencv_stitching=OFF -DBUILD_opencv_ts=OFF -DBUILD_opencv_video=OFF -DBUILD_opencv_videoio=OFF  -DCMAKE_INSTALL_PREFIX=`pwd`/../../install && ninja && ninja install
popd
