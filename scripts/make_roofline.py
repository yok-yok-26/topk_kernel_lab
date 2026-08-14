#!/usr/bin/env python3
from pathlib import Path
import csv
import math

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
bench = root / "reports/benchmark/latest.csv"
out_dir = root / "reports/roofline"
out_dir.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(bench.open())) if bench.exists() else []

W, H = 1360, 860
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 130, 70, 86, 126
COLORS = {
    "naive": (130, 130, 130),
    "v1": (22, 108, 180),
    "v2": (0, 150, 105),
    "v3": (190, 92, 20),
    "seme": (22, 108, 180),
    "warp": (0, 150, 105),
    "soa_warp": (190, 92, 20),
    "cub_sort": (132, 69, 178),
    "library_cub_segmented_sort": (132, 69, 178),
    "library_cub_stable_segmented_sort": (132, 69, 178),
    "library_cub_segmented_radix_sort": (89, 99, 110),
    "library_tensorrt_topk": (194, 59, 115),
}

# Estimated guide rails for this RTX 5070 lab. They are intentionally labeled
# as estimates; NCU details remain the source of truth for measured metrics.
EST_PEAK_BW_GBPS = 672.0
EST_PEAK_COMPUTE_GOPS = 30000.0

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    TINY = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    TITLE = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
except Exception:
    FONT = SMALL = TINY = TITLE = ImageFont.load_default()

# TopK is comparison-heavy. Use estimated comparison work for roofline placement
# until a richer NCU-derived operation model is added.
def ops_and_bytes(row):
    batch = float(row["batch"])
    n = float(row["n"])
    k = float(row["k"])
    bytes_moved = batch * n * 4.0 + batch * k * 8.0
    # Approximate comparison/sort work: selection/merge work scales with N and log2(K).
    ops = batch * n * max(1.0, math.log2(max(2.0, k)))
    return ops, bytes_moved

def point_for(row):
    ops, bytes_moved = ops_and_bytes(row)
    ms = float(row["avg_ms"])
    ai = ops / max(1.0, bytes_moved)
    perf = ops / (ms / 1000.0) / 1.0e9
    eff_bw = bytes_moved / (ms / 1000.0) / 1.0e9
    return ai, perf, eff_bw

def nice_log_ticks(vmin, vmax):
    start = math.floor(math.log10(vmin))
    stop = math.ceil(math.log10(vmax))
    return [10.0 ** p for p in range(start, stop + 1)]

def draw_plot(path, title_text, x_range, y_range, log_y, show_roofline):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    plot_l, plot_t = MARGIN_L, MARGIN_T
    plot_r, plot_b = W - MARGIN_R, H - MARGIN_B
    d.rectangle([plot_l, plot_t, plot_r, plot_b], outline=(35, 35, 35), width=2)
    d.text((W // 2, 30), title_text, anchor="mm", fill=(20, 20, 20), font=TITLE)
    d.text((W // 2, H - 45), "Arithmetic intensity, estimated comparison-ops/byte", anchor="mm", fill=(20, 20, 20), font=FONT)
    d.text((34, H // 2), "Achieved performance, estimated GOP/s", anchor="mm", fill=(20, 20, 20), font=FONT)

    x_min, x_max = x_range
    y_min, y_max = y_range

    def xmap(x):
        x = max(x, 1e-12)
        return plot_l + (math.log10(x) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min)) * (plot_r - plot_l)

    def ymap(y):
        y = max(y, 1e-12)
        if log_y:
            return plot_b - (math.log10(y) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min)) * (plot_b - plot_t)
        return plot_b - (y - y_min) / (y_max - y_min) * (plot_b - plot_t)

    for x in nice_log_ticks(x_min, x_max):
        if x_min <= x <= x_max:
            px = xmap(x)
            d.line([px, plot_t, px, plot_b], fill=(235, 235, 235))
            d.text((px, plot_b + 12), f"{x:g}", anchor="mt", fill=(50, 50, 50), font=SMALL)

    y_ticks = nice_log_ticks(y_min, y_max) if log_y else [y_min + (y_max - y_min) * i / 5 for i in range(6)]
    for y in y_ticks:
        if y_min <= y <= y_max:
            py = ymap(y)
            d.line([plot_l, py, plot_r, py], fill=(225, 225, 225))
            d.text((plot_l - 12, py), f"{y:g}", anchor="rm", fill=(50, 50, 50), font=SMALL)

    if show_roofline:
        knee = EST_PEAK_COMPUTE_GOPS / EST_PEAK_BW_GBPS
        xs = []
        p = math.floor(math.log10(x_min * 10))
        while True:
            x = 10 ** (p / 8.0)
            if x > x_max * 1.01:
                break
            if x >= x_min:
                xs.append(x)
            p += 1
        if x_min <= knee <= x_max:
            xs.append(knee)
        xs = sorted(set(round(x, 10) for x in xs))
        line = []
        for x in xs:
            y = min(EST_PEAK_COMPUTE_GOPS, x * EST_PEAK_BW_GBPS)
            if y_min <= y <= y_max:
                line.append((xmap(x), ymap(y)))
        for a, b in zip(line, line[1:]):
            d.line([a[0], a[1], b[0], b[1]], fill=(170, 55, 55), width=3)
        if x_min <= knee <= x_max:
            px, py = xmap(knee), ymap(EST_PEAK_COMPUTE_GOPS)
            d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(170, 55, 55))
            d.text((px + 12, py - 10), f"estimated knee AI~{knee:.1f}", fill=(140, 45, 45), font=TINY)
        d.text((plot_r - 10, ymap(EST_PEAK_COMPUTE_GOPS) - 8), "est. compute ceiling", anchor="rb", fill=(140, 45, 45), font=TINY)
        d.text((xmap(max(x_min * 1.4, min(knee / 5, x_max))), ymap(max(y_min * 2, min(EST_PEAK_COMPUTE_GOPS / 8, y_max))) + 12), "est. bandwidth slope", fill=(140, 45, 45), font=TINY)

    if not rows:
        d.text(((plot_l + plot_r) // 2, (plot_t + plot_b) // 2), "No benchmark CSV found", anchor="mm", fill=(120, 0, 0), font=FONT)
    else:
        best_perf = max(point_for(row)[1] for row in rows)
        for i, row in enumerate(rows):
            mode = row["mode"]
            ai, perf, eff_bw = point_for(row)
            px, py = xmap(ai), ymap(perf)
            color = COLORS.get(mode, (80, 80, 180))
            radius = 10 if perf == best_perf else 8
            d.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color, outline=(0, 0, 0))
            label = f"{mode}\n{float(row['avg_ms']):.4f} ms\n{perf:.1f} GOP/s\n{eff_bw:.1f} GB/s"
            if show_roofline:
                offsets = {
                    "naive": (18, -64),
                    "v1": (18, -18),
                    "v2": (18, 32),
                    "v3": (18, 78),
                    "seme": (18, -18),
                    "warp": (18, 32),
                    "soa_warp": (18, 78),
                    "cub_sort": (18, 116),
                    "library_cub_segmented_sort": (18, 116),
                    "library_cub_stable_segmented_sort": (18, 116),
                    "library_tensorrt_topk": (18, 154),
                }
            else:
                offsets = {
                    "naive": (18, -46),
                    "v1": (18, -10),
                    "v2": (18, 28),
                    "v3": (18, 64),
                    "seme": (18, -10),
                    "warp": (18, 28),
                    "soa_warp": (18, 64),
                    "cub_sort": (18, 100),
                    "library_cub_segmented_sort": (18, 100),
                    "library_cub_stable_segmented_sort": (18, 100),
                    "library_tensorrt_topk": (18, 136),
                }
            dx, dy = offsets.get(mode, (18, -30 + (i % 3) * 26))
            tx = min(max(plot_l + 6, px + dx), plot_r - 180)
            ty = min(max(plot_t + 6, py + dy), plot_b - 76)
            d.multiline_text((tx, ty), label, fill=(20, 20, 20), font=SMALL, spacing=3)

    img.save(path)

points = [point_for(row) for row in rows]
if points:
    xs = [p[0] for p in points]
    ps = [p[1] for p in points]
    local_x = (max(1e-2, min(xs) / 1.8), max(xs) * 1.8)
    local_y = (0.0, max(1.0, max(ps) * 1.35))
else:
    local_x = (1e-2, 10.0)
    local_y = (0.0, 10.0)

# Global range is deliberately bounded and readable while showing the roofline elbow.
knee = EST_PEAK_COMPUTE_GOPS / EST_PEAK_BW_GBPS
global_x = (1e-2, max(128.0, knee * 2.0))
global_y = (1e-1, EST_PEAK_COMPUTE_GOPS * 1.8)

draw_plot(out_dir / "topk_latest.png", "TopK Roofline - Local View", local_x, local_y, log_y=False, show_roofline=False)
draw_plot(out_dir / "topk_global_latest.png", "TopK Roofline - Global View", global_x, global_y, log_y=True, show_roofline=True)

(out_dir / "topk_latest.md").write_text(
    "# TopK roofline note\n\n"
    "Data source: `reports/benchmark/latest.csv`. Points use benchmark timing, effective bytes moved, and estimated comparison work. "
    "TopK is comparison-heavy rather than a clean FLOP workload, so arithmetic intensity and GOP/s are labeled as estimated.\n\n"
    "- `topk_latest.png`: local view, zoomed around measured implementations for relative comparison.\n"
    "- `topk_global_latest.png`: global view with a visible estimated roofline elbow, using a bounded arithmetic-intensity range so the plot remains readable.\n\n"
    f"Estimated ceilings used for visual context: memory bandwidth {EST_PEAK_BW_GBPS:.0f} GB/s, compute {EST_PEAK_COMPUTE_GOPS:.0f} GOP/s, knee AI~{knee:.1f}. "
    "These are guide rails, not calibrated device peaks. Prefer NCU details for concrete kernel metrics.\n",
    encoding="utf-8",
)
print(out_dir / "topk_latest.png")
print(out_dir / "topk_global_latest.png")
