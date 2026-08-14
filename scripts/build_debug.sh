#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
cmake -S . -B build/debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTOPK_DEBUG_SYNC=ON -DCMAKE_CUDA_FLAGS_DEBUG="-O0 -G -g -lineinfo" -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/debug
