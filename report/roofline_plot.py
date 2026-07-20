#!/usr/bin/env python3
"""
cuCore/report/roofline_plot.py

Generates a roofline plot from measured kernel performance numbers. This
does NOT invent numbers for you -- fill in `MEASUREMENTS` below with your
own results from running the benchmarks + reading achieved GFLOP/s and
GB/s off the printed benchmark output (or Nsight Compute), then run:

    python3 roofline_plot.py --peak-flops 35.0 --peak-bw 1000 --out roofline.png

--peak-flops: your GPU's peak FP32 TFLOP/s (single precision, non-tensor-core;
              check your GPU's spec sheet, e.g. RTX 4090 ~= 82.6 TFLOP/s FP32,
              A100 ~= 19.5 TFLOP/s FP32 non-TC, but always verify from vendor
              docs for the exact SKU you're running on).
--peak-bw:    your GPU's peak memory bandwidth in GB/s (e.g. RTX 4090 ~1008
              GB/s, A100 40GB ~1555 GB/s -- again, verify against your card).

Requires: matplotlib, numpy (pip install matplotlib numpy)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# FILL THIS IN with your own measured (arithmetic_intensity, gflops) points,
# one per kernel/version you want to plot. Arithmetic intensity = FLOPs / byte
# moved from/to DRAM (use the `bytes_moved` / `flops` fields the C++ harness
# already prints for you where applicable).
# ---------------------------------------------------------------------------
MEASUREMENTS = [
    # (label,                  arithmetic_intensity (FLOP/byte), achieved GFLOP/s)
    # ("vector_add (naive)",     0.083,   0.0),   # <- replace 0.0 with your measured GFLOP/s
    # ("vector_add (optimized)", 0.083,   0.0),
    # ("saxpy (optimized)",      0.167,   0.0),
    # ("matmul (naive)",         0.0,     0.0),   # AI = N/6 roughly for large square GEMM; compute per-run
    # ("matmul (smem tiled)",    0.0,     0.0),
    # ("matmul (register tiled)",0.0,     0.0),
    # ("matmul (cuBLAS)",        0.0,     0.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-flops", type=float, required=True, help="Peak FP32 TFLOP/s of your GPU")
    ap.add_argument("--peak-bw", type=float, required=True, help="Peak memory bandwidth in GB/s of your GPU")
    ap.add_argument("--out", type=str, default="roofline.png")
    args = ap.parse_args()

    peak_flops_gflops = args.peak_flops * 1000.0  # TFLOP/s -> GFLOP/s
    peak_bw_gbs = args.peak_bw

    ridge_point = peak_flops_gflops / peak_bw_gbs  # AI at which memory-bound meets compute-bound

    ai = np.logspace(-3, 3, 500)
    roofline = np.minimum(peak_bw_gbs * ai, peak_flops_gflops)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(ai, roofline, color="black", linewidth=2, label="Roofline")
    ax.axvline(ridge_point, color="gray", linestyle="--", linewidth=1,
               label=f"Ridge point (AI={ridge_point:.2f})")

    if not MEASUREMENTS:
        ax.text(0.5, 0.5, "No measurements plotted yet.\nFill in MEASUREMENTS in this script\nwith your real benchmark numbers.",
                transform=ax.transAxes, ha="center", va="center", fontsize=11, color="red")
    for label, x, y in MEASUREMENTS:
        ax.scatter([x], [y], s=60, zorder=5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("cuCore Roofline")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved roofline plot to {args.out}")


if __name__ == "__main__":
    main()
