#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p reports/correctness
./build/debug/test_topk "${1:-all}" "${2:-all}" | tee reports/correctness/stdout_latest.log
