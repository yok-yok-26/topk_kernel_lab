#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p reports/nsys
mode="${5:-warp}"
out="reports/nsys/topk_${mode}_latest"
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --stats=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output="$out" \
  env TOPK_PROFILE_SINGLE_OP=1 TOPK_BENCH_CSV="reports/benchmark/nsys_${mode}_latest.csv" \
  ./build/release/bench_topk "${1:-1024}" "${2:-4096}" "${3:-32}" "${4:-1}" "$mode" 0 \
  | tee "${out}_stdout.log"
cp "${out}_stdout.log" reports/nsys/latest_stdout.log
cp "${out}.nsys-rep" reports/nsys/topk_latest.nsys-rep
if [ -f "${out}.sqlite" ]; then
  cp "${out}.sqlite" reports/nsys/topk_latest.sqlite
fi
