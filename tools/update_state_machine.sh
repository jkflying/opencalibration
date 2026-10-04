#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 external/usm/generate_flow_diagram.py src/pipeline/pipeline.cpp src/pipeline/pipeline.cpp.dot
