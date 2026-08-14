#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p reports/benchmark reports/trends
out="reports/benchmark/size_sweep_latest.csv"
echo "mode,batch,n,k,avg_ms,effective_gbs" > "$out"
# At least five representative sizes. K stays <=32 because v2/v3 are defined only for k<=32.
cases=(
  "256 1024 16 80"
  "1024 2048 32 60"
  "1024 4096 32 50"
  "2048 4096 32 30"
  "1024 8192 32 30"
  "256 16384 32 30"
)
for c in "${cases[@]}"; do
  read -r b n k iters <<< "$c"
  tmp="reports/benchmark/size_sweep_${b}_${n}_${k}.csv"
  TOPK_BENCH_CSV="$tmp" ./build/release/bench_topk "$b" "$n" "$k" "$iters" all 5 >/dev/null
  tail -n +2 "$tmp" >> "$out"
done
python3 scripts/summarize_size_sweep.py "$out"
echo "wrote $out"
