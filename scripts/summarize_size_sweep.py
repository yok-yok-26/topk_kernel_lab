#!/usr/bin/env python3
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path

inp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/benchmark/size_sweep_latest.csv")
out_dir = Path("reports/trends")
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
with inp.open(newline="") as f:
    for r in csv.DictReader(f):
        r["batch"] = int(r["batch"])
        r["n"] = int(r["n"])
        r["k"] = int(r["k"])
        r["avg_ms"] = float(r["avg_ms"])
        r["effective_gbs"] = float(r["effective_gbs"])
        rows.append(r)

by_shape = defaultdict(list)
for r in rows:
    by_shape[(r["batch"], r["n"], r["k"])].append(r)

summary = []
user_modes = {"v1", "v2", "v3"}
for shape, group in sorted(by_shape.items()):
    baseline = next((r for r in group if r["mode"] == "library_tensorrt_topk"), None)
    cub = next((r for r in group if r["mode"] in {
        "library_cub_stable_segmented_sort", "library_cub_segmented_sort",
        "library_cub_segmented_radix_sort", "cub_sort"
    }), None)
    naive = next((r for r in group if r["mode"] == "naive"), None)
    users = [r for r in group if r["mode"] in user_modes]
    best = min(users, key=lambda r: r["avg_ms"])
    rec = {
        "batch": shape[0],
        "n": shape[1],
        "k": shape[2],
        "best_user_mode": best["mode"],
        "best_user_ms": best["avg_ms"],
        "best_user_gbs": best["effective_gbs"],
        "baseline_mode": baseline["mode"] if baseline else "",
        "baseline_ms": baseline["avg_ms"] if baseline else "",
        "baseline_gbs": baseline["effective_gbs"] if baseline else "",
        "user_vs_baseline_speedup": (baseline["avg_ms"] / best["avg_ms"]) if baseline else "",
        "cub_full_sort_ms": cub["avg_ms"] if cub else "",
        "naive_ms": naive["avg_ms"] if naive else "",
        "user_vs_naive_speedup": (naive["avg_ms"] / best["avg_ms"]) if naive else "",
    }
    summary.append(rec)

summary_csv = out_dir / "topk_size_sweep_best_vs_baseline.csv"
with summary_csv.open("w", newline="") as f:
    keys = ["batch", "n", "k", "best_user_mode", "best_user_ms", "best_user_gbs",
            "baseline_mode", "baseline_ms", "baseline_gbs", "user_vs_baseline_speedup",
            "cub_full_sort_ms", "naive_ms", "user_vs_naive_speedup"]
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(summary)

md = out_dir / "topk_size_sweep_latest.md"
lines = [
    "# TopK size sweep", "",
    "Benchmark source: `reports/benchmark/size_sweep_latest.csv`.",
    "Main L1 baseline: `library_tensorrt_topk` uses one public TensorRT TopK layer for the complete batch. Engine build, context creation, and allocations are outside timing; `enqueueV3` is timed on the same default stream. Values are sorted and indices point to matching inputs. Tie index ordering is not guaranteed, so tie-heavy cases are a documented semantic exception.",
    "Strict-contract auxiliary L1 baseline: `library_cub_stable_segmented_sort` fully sorts every row with one public CUB primitive. Stable descending order preserves the original lower index for equal values. It is retained as an exact-semantics full-sort comparison, not the main algorithm-matched TopK baseline.",
    "User kernels: `v1`, `v2`, `v3`. `K` is kept <=32 so `v2/v3` are in their legal domain.", "",
    "| B | N | K | best user | best user ms | TensorRT TopK ms | user/TRT speedup | CUB full sort ms | naive ms | user/naive speedup |",
    "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
]
for r in summary:
    lines.append(
        f"| {r['batch']} | {r['n']} | {r['k']} | {r['best_user_mode']} | "
        f"{float(r['best_user_ms']):.6g} | {float(r['baseline_ms']):.6g} | "
        f"{float(r['user_vs_baseline_speedup']):.3g}x | {float(r['cub_full_sort_ms']):.6g} | "
        f"{float(r['naive_ms']):.6g} | {float(r['user_vs_naive_speedup']):.3g}x |"
    )
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_csv)
print(md)
