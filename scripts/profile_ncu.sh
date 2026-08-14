#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p reports/ncu
mode="${5:-warp}"
out="reports/ncu/topk_${mode}_latest"
ncu --force-overwrite --set full --target-processes all --profile-from-start off --export "$out" \
  env TOPK_PROFILE_SINGLE_OP=1 TOPK_BENCH_CSV="reports/benchmark/ncu_${mode}_latest.csv" \
  ./build/release/bench_topk "${1:-1024}" "${2:-4096}" "${3:-32}" "${4:-1}" "$mode" 0 \
  | tee "${out}_raw.txt"
ncu --import "${out}.ncu-rep" --page details > "${out}_details.txt" || true
cp "${out}_raw.txt" reports/ncu/topk_latest_raw.txt
cp "${out}_details.txt" reports/ncu/topk_latest_details.txt
cp "${out}.ncu-rep" reports/ncu/topk_latest.ncu-rep
