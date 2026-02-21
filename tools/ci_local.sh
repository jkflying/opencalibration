#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

group()    { echo "::group::$1"; date; }
endgroup() { echo "::endgroup::"; }

group "Install dependencies"
tools/install_dependencies.sh
endgroup

group "Validate state machine"
python3 external/usm/generate_flow_diagram.py -v src/pipeline/pipeline.cpp src/pipeline/pipeline.cpp.dot
endgroup

group "Check formatting"
tools/check_style.sh
endgroup

group "clang-tidy"
mkdir -p build_tidy
cmake -S . -B build_tidy -G Ninja -DCMAKE_BUILD_TYPE=Debug -DOPENCALIBRATION_CLANG_TIDY=ON
ninja -C build_tidy
endgroup

group "CMake (asan)"
mkdir -p build_asan
cmake -S . -B build_asan -G Ninja -DCMAKE_BUILD_TYPE=asan
endgroup

group "Address Sanitizer Build"
ninja -C build_asan
endgroup

group "Address Sanitizer Test"
ASAN_OPTIONS=detect_leaks=0 ninja -C build_asan test
endgroup

group "Coverage Build"
mkdir -p build_coverage
cmake -S . -B build_coverage -G Ninja -DCMAKE_BUILD_TYPE=Coverage
ninja -C build_coverage
endgroup

group "Coverage Test"
ninja -C build_coverage test_coverage_html
endgroup

echo "All CI steps passed."
