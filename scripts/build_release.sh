#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -lineinfo" -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/release
