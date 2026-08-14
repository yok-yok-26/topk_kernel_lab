#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "reports" / "benchmark" / "latest.csv"
NCU_DIR = ROOT / "reports" / "ncu"
OUT = ROOT / "reports" / "trends"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "naive": "#75808a",
    "v1": "#2267b0",
    "v2": "#15946b",
    "v3": "#b96b1e",
    "seme": "#2267b0",
    "warp": "#15946b",
    "soa_warp": "#b96b1e",
    "cub_sort": "#8445b2",
    "library_cub_segmented_sort": "#8445b2",
    "library_cub_stable_segmented_sort": "#8445b2",
    "library_cub_segmented_radix_sort": "#59636e",
    "library_tensorrt_topk": "#c23b73",
}
IDEAS = {
    "naive": "one thread per row reference",
    "v1": "shared-memory bitonic per block",
    "v2": "warp-local topK plus block merge",
    "v3": "SoA shared layout experiment",
    "seme": "shared-memory bitonic per block",
    "warp": "warp-local topK plus block merge",
    "soa_warp": "SoA shared layout experiment",
    "cub_sort": "CUB segmented radix sort baseline",
    "library_cub_segmented_sort": "CUB segmented radix full-sort auxiliary baseline",
    "library_cub_stable_segmented_sort": "CUB stable segmented full-sort strict baseline",
    "library_cub_segmented_radix_sort": "CUB fixed segmented radix full-sort diagnostic baseline",
    "library_tensorrt_topk": "TensorRT public TopK main L1 baseline",
}


def read_benchmark():
    if not BENCH.exists():
        return []
    with BENCH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["avg_ms"] = float(r["avg_ms"])
        r["effective_gbs"] = float(r["effective_gbs"])
        r["batch"] = int(r["batch"])
        r["n"] = int(r["n"])
        r["k"] = int(r["k"])
    return rows


def first_kernel_text(mode: str) -> str:
    path = NCU_DIR / f"topk_{mode}_latest_details.txt"
    if not path.exists():
        return ""
    lines = path.read_text(errors="ignore").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("    Section: GPU Speed Of Light Throughput")]
    if not starts:
        return "\n".join(lines)
    end = starts[1] if len(starts) > 1 else len(lines)
    return "\n".join(lines[starts[0]:end])


def metric_value(text: str, label: str):
    for line in text.splitlines():
        if label in line:
            parts = line.split()
            for token in reversed(parts):
                try:
                    return float(token)
                except ValueError:
                    pass
    return None


def collect_ncu(rows):
    out = []
    for r in rows:
        mode = r["mode"]
        if mode == "naive":
            continue
        text = first_kernel_text(mode)
        if not text:
            continue
        metrics = {
            "duration_us": metric_value(text, "Duration"),
            "sm_throughput_pct": metric_value(text, "Compute (SM) Throughput"),
            "memory_throughput_pct": metric_value(text, "Memory Throughput                 %"),
            "dram_throughput_pct": metric_value(text, "DRAM Throughput"),
            "l1tex_throughput_pct": metric_value(text, "L1/TEX Cache Throughput"),
            "eligible_warps_per_sched": metric_value(text, "Eligible Warps Per Scheduler"),
            "warp_cycles_per_issued": metric_value(text, "Warp Cycles Per Issued Instruction"),
            "active_threads_per_warp": metric_value(text, "Avg. Active Threads Per Warp"),
            "not_predicated_threads_per_warp": metric_value(text, "Avg. Not Predicated Off Threads Per Warp"),
            "executed_instructions": metric_value(text, "Executed Instructions"),
            "registers_per_thread": metric_value(text, "Registers Per Thread"),
            "achieved_occupancy_pct": metric_value(text, "Achieved Occupancy"),
        }
        for line in text.splitlines():
            if "Duration" in line:
                if " ms " in f" {line} ":
                    metrics["duration_us"] = metrics["duration_us"] * 1000 if metrics["duration_us"] is not None else None
                break
        out.append({"mode": mode, **metrics})
    return out


def svg_bar(path: Path, rows, value_key, title, unit, lower_is_better=False):
    width = 980
    row_h = 44
    top = 70
    left = 170
    right = 60
    height = top + len(rows) * row_h + 50
    values = [float(r[value_key]) for r in rows if r.get(value_key) is not None]
    vmax = max(values) if values else 1.0
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="24" y="34" font-family="sans-serif" font-size="22" font-weight="700" fill="#17212b">{html.escape(title)}</text>')
    direction = "Lower is better." if lower_is_better else "Higher is better."
    lines.append(f'<text x="24" y="56" font-family="sans-serif" font-size="13" fill="#64707d">Release benchmark timing only. {direction}</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h
        val = float(r[value_key])
        color = COLORS.get(r["mode"], "#64707d")
        bar_w = max(2, (width - left - right) * val / vmax)
        lines.append(f'<line x1="{left}" y1="{y+22}" x2="{width-right}" y2="{y+22}" stroke="#edf1f5"/>')
        lines.append(f'<text x="24" y="{y+18}" font-family="sans-serif" font-size="15" fill="#17212b">{html.escape(r["mode"])}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="24" rx="3" fill="{color}"/>')
        lines.append(f'<text x="{left + bar_w + 8:.1f}" y="{y+18}" font-family="sans-serif" font-size="13" fill="#17212b">{val:.6g} {html.escape(unit)}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def svg_ncu_by_algorithm(path: Path, rows):
    metrics = [
        ("duration_us", "Duration us"),
        ("sm_throughput_pct", "SM throughput %"),
        ("memory_throughput_pct", "Memory throughput %"),
        ("dram_throughput_pct", "DRAM throughput %"),
        ("l1tex_throughput_pct", "L1/TEX throughput %"),
        ("eligible_warps_per_sched", "Eligible warps/sched"),
        ("warp_cycles_per_issued", "Warp cycles/issued"),
        ("executed_instructions", "Executed inst"),
        ("registers_per_thread", "Registers/thread"),
        ("achieved_occupancy_pct", "Achieved occupancy %"),
    ]
    width = 1280
    metric_h = 22
    block_gap = 26
    top = 112
    left = 250
    bar_w = 560
    raw_x = left + bar_w + 38
    block_h = len(metrics) * metric_h + block_gap
    height = top + len(rows) * block_h + 45
    max_by_metric = {k: max([r.get(k) or 0 for r in rows] or [1]) or 1 for k, _ in metrics}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append('<text x="24" y="34" font-family="sans-serif" font-size="22" font-weight="700" fill="#17212b">Combined NCU metrics grouped by algorithm</text>')
    lines.append('<text x="24" y="58" font-family="sans-serif" font-size="13" fill="#64707d">Bar length is normalized per metric across algorithms; raw values are shown on the right.</text>')
    for idx, (_, label) in enumerate(metrics, 1):
        x = 24 + ((idx - 1) % 5) * 235
        y = 84 + ((idx - 1) // 5) * 18
        lines.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="12" fill="#64707d">{idx}. {html.escape(label)}</text>')
    for bi, r in enumerate(rows):
        y0 = top + bi * block_h
        mode = r["mode"]
        color = COLORS.get(mode, "#64707d")
        lines.append(f'<text x="24" y="{y0+16}" font-family="sans-serif" font-size="17" font-weight="700" fill="{color}">{html.escape(mode)}</text>')
        lines.append(f'<text x="24" y="{y0+34}" font-family="sans-serif" font-size="12" fill="#64707d">{html.escape(IDEAS.get(mode, ""))}</text>')
        for mi, (key, label) in enumerate(metrics, 1):
            y = y0 + mi * metric_h
            val = r.get(key)
            norm = 0 if val is None else val / max_by_metric[key]
            bw = max(2, bar_w * norm)
            lines.append(f'<text x="196" y="{y+13}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#64707d">{mi}</text>')
            lines.append(f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="14" rx="2" fill="{color}" opacity="0.78"/>')
            raw = "NA" if val is None else (f"{val:.3g}" if abs(val) >= 10000 else f"{val:.4g}")
            lines.append(f'<text x="{raw_x}" y="{y+12}" font-family="monospace" font-size="12" fill="#17212b">{html.escape(raw)}</text>')
            lines.append(f'<text x="{raw_x+110}" y="{y+12}" font-family="sans-serif" font-size="12" fill="#64707d">{html.escape(label)}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows, keys):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def main():
    bench = read_benchmark()
    if not bench:
        raise SystemExit(f"missing {BENCH}")
    svg_bar(OUT / "topk_release_latency.svg", bench, "avg_ms", "TopK release latency by mode", "ms", lower_is_better=True)
    svg_bar(OUT / "topk_release_throughput.svg", bench, "effective_gbs", "TopK release effective throughput by mode", "GB/s")
    write_csv(OUT / "topk_release_summary.csv", bench, ["mode", "batch", "n", "k", "avg_ms", "effective_gbs"])
    ncu = collect_ncu(bench)
    if ncu:
        keys = ["mode", "duration_us", "sm_throughput_pct", "memory_throughput_pct", "dram_throughput_pct", "l1tex_throughput_pct", "eligible_warps_per_sched", "warp_cycles_per_issued", "active_threads_per_warp", "not_predicated_threads_per_warp", "executed_instructions", "registers_per_thread", "achieved_occupancy_pct"]
        write_csv(OUT / "topk_ncu_summary.csv", ncu, keys)
        svg_ncu_by_algorithm(OUT / "topk_ncu_metrics_by_algorithm.svg", ncu)
    note = OUT / "topk_trends_latest.md"
    note.write_text(
        "# TopK trend plots\n\n"
        "Data sources: `reports/benchmark/latest.csv` and per-mode `reports/ncu/topk_<mode>_latest_details.txt`.\n\n"
        "Generated files:\n"
        "- `topk_release_latency.svg`: release benchmark latency, lower is better.\n"
        "- `topk_release_throughput.svg`: release benchmark effective logical throughput, higher is better.\n"
        "- `topk_release_summary.csv`: raw release benchmark table.\n"
        "- `topk_ncu_summary.csv`: first-kernel NCU metric summary for profiled modes.\n"
        "- `topk_ncu_metrics_by_algorithm.svg`: per-metric normalized NCU diagnostic chart grouped by algorithm.\n\n"
        "Benchmark timings and profiler replay timings are intentionally kept separate.\n",
        encoding="utf-8",
    )
    for p in sorted(OUT.iterdir()):
        print(p)

if __name__ == "__main__":
    main()
