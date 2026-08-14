#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p reports/benchmark
./build/release/bench_topk "${1:-1024}" "${2:-4096}" "${3:-32}" "${4:-50}" "${5:-all}" | tee reports/benchmark/stdout_latest.log
