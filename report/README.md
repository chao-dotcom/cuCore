# cuCore Benchmark Report — July 30, 2026

This directory contains the complete benchmark results and optimization analysis for all cuCore kernels, generated from the comprehensive benchmark notebook (`cuCore_benchmark_(5).ipynb`).

## Hardware

- **GPU:** NVIDIA RTX PRO 6000 Black (97.9 GB HBM)
- **Driver:** 580.82.07
- **CUDA Toolkit:** 13.0
- **Date:** July 30, 2026

## Files

### Benchmark Results (CSV)

- **`benchmark_results.csv`** — Summary of all Stage 1, 2, and 4 kernels with CPU baseline, naive CUDA, and optimized CUDA timings. Shows speedup factors and throughput.
  - 11 kernels: vector_add, saxpy, dot_product, reduction, scan, histogram, softmax, layernorm, gelu, conv2d, transpose
  
- **`matmul_sweep.csv`** — GEMM performance across matrix sizes (N = 256 to 4096) comparing naive, shared-memory-tiled, register-tiled (4×4), and cuBLAS (original repo benchmark).
  - Register-tiled variant achieves 43% of cuBLAS at N=4096
  
- **`matmul_dbuf_sweep.csv`** — Double-buffered GEMM (8×8 tiles + vectorized loads + software pipelining).
  - Achieves 37.5% of cuBLAS at N=4096 (note: slightly slower than register-tiled due to shared memory ping-pong overhead)
  
- **`matmul_v2b_sweep.csv`** — Register-tiled v2b (64×64 block tile, occupancy fix for small/medium N).
  - Achieves 93.2% of cuBLAS at N=4096, **79.1%–97.4%** at N=1024–2048 (solves occupancy bottleneck)
  
- **`matmul_v2c_sweep.csv`** — v2b with split-K (K-dimension splitting for additional parallelism).
  - Achieves **85.6%** of cuBLAS at N=1024 (above the 85% occupancy target), 91.9% at N=4096
  
- **`matmul_tensorcore_v3_sweep.csv`** — Tensor Core WMMA kernel with 128×128 block tile.
  - Achieves **126.5%** of cuBLAS default (pedantic FP32) at N=4096 because it uses TF32 (10× higher peak throughput)
  - When measured against cuBLAS's explicit TF32 mode, reaches ~75% (vendor tuning gap)

### Optimization Reports

Detailed analysis for each major kernel, including implementation ladder, bottleneck diagnosis, and roadmap for further optimization.

- **`optimization_report_dot_product.md`** — **1007.65x speedup** via warp shuffles; achieves peak reduction bandwidth
- **`optimization_report_histogram.md`** — **206.52x speedup** via per-block shared-memory private histograms; near-peak-bandwidth-bound
- **`optimization_report_reduction.md`** — **6.17x speedup** via sequential addressing + warp shuffles; eliminates bank conflicts and divergence
- **`optimization_report_transpose.md`** — **6.14x speedup** via shared-memory tiling with +1 padding; eliminates write coalescing and bank conflicts
- **`optimization_report_matmul.md`** — **Multi-variant GEMM analysis** (naive → shared memory → register-tiled → double-buffered); roadmap to Tensor Core acceleration

### Templates & Utilities

- **`optimization_report_template.md`** — Markdown template for documenting kernel optimizations (Problem Statement, Implementation Ladder, Benchmark Results, Nsight Compute Analysis, Roofline Position, Bottleneck Diagnosis, Remaining Gap)
- **`roofline_plot.py`** — Python script to generate roofline plots (model-based visualizations of compute vs. bandwidth bottlenecks)

## Key Findings

### By Speedup Magnitude

| Kernel | Speedup (naive→optimized) | CPU baseline | Bottleneck |
|---|---|---|---|
| Dot Product | **1007.65x** | 441x | Warp shuffle reduction |
| Histogram | **206.52x** | 246x | Atomic operation locality |
| Reduction | **6.17x** | 533x | Bank conflicts + divergence |
| Transpose | **6.14x** | 4,930x | Write coalescing + shared memory |
| Vector Add | 1.0x | 31x | Memory-bound; nearly optimal |
| Softmax | 1.75x | 2,515x | Multi-stage reduction fusion |
| Layernorm | 1.83x | 381x | Welford's algorithm + shuffles |
| GELU | 0.97x | 2,197x | (Naive ≈ Optimized; need profiling) |
| Conv2D | 1.34x | 1,293x | (Bottleneck: shared memory footprint?) |
| Scan | 0.44x | 41x | (Optimized regressed; likely correctness bug or architectural mismatch) |
| SAXPY | 1.01x | 2.2x | Memory-bound; minimal room |

### GEMM (Across Matrix Sizes)

- **Register-tiled (4×4)** peaks at **43% of cuBLAS** (N=4096)
- **Double-buffered v2 (8×8)** achieves **37.5% of cuBLAS** (N=4096)
  - Gap to cuBLAS is primarily **Tensor Cores** (WMMA), not algorithmic improvements
  - Software pipelining alone (v1) showed minimal benefit; larger tiles and vectorization (v2) matter more
- **Arithmetic intensity is still the constraint** — register-tiled at 31.7 TFLOP/s despite being 43% of cuBLAS because cuBLAS uses Tensor Cores (~10x higher throughput per watt)

## Next Steps (Per Roadmap)

See `docs/roadmap.md` for full prioritization. Priority items for closing the GEMM gap:

1. **Implement Tensor Core GEMM (WMMA, tf32)** — Should yield 60–70% of cuBLAS
2. **Async shared memory prefetch** (cuda::memcpy_async) — 10–15% improvement
3. **Double-buffered shared memory** (ping-pong shared memory buffers) — Additional latency hiding
4. **Profile-guided tile size tuning** — Larger registers tiles if register file permits

## How to Use These Results

1. **Reproducing:** Run the benchmark notebook (`cuCore_benchmark_(5).ipynb`) on your own GPU to collect data.
2. **Understanding bottlenecks:** Use Nsight Compute (ncu) to collect the metrics listed in each report's "Nsight Compute Analysis" section.
3. **Filling in the template:** For new kernels, use `optimization_report_template.md` as a starting point and fill in actual numbers from your runs.
4. **Roofline visualization:** Use `roofline_plot.py` with your measured GFLOP/s and GB/s to generate a roofline plot and visualize your position on the compute-vs-bandwidth frontier.

## References

- **Roofline Model:** Williams, Waterman, Patterson (2009) — "Roofline: An Insightful Visual Performance Model for Floating-Point Programs"
- **Bank Conflict Elimination:** Harris et al. (2012) — "Optimizing Parallel Reduction in CUDA"
- **Tensor Cores:** NVIDIA Turing, Ampere, Hopper documentation
- **Double Buffering in CUDA:** [NVIDIA Blog: Optimizing CUDA](https://developer.nvidia.com/blog/cuda-code-samples-and-tips/)
