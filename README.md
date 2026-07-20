# cuCore — A CUDA Deep Learning Primitive Library

A progressively-optimized collection of CUDA kernels, built to demonstrate
GPU architecture understanding rather than reproduce state-of-the-art
performance: CPU baseline → naive CUDA → optimized CUDA → benchmark →
Nsight Compute analysis, for every primitive, benchmarked against cuBLAS,
CUB, and PyTorch where applicable.

> **Status:** This repository was scaffolded and implemented in an
> environment with no GPU / no `nvcc` available, so nothing here has been
> compiled or run yet. Every kernel is written to compile and run correctly
> on real hardware (Volta/Turing/Ampere/Ada/Hopper), but **you will be the
> first to build it — budget time for fixing whatever your specific
> compiler/driver/architecture combination surfaces.** See
> [`docs/roadmap.md`](docs/roadmap.md) for exactly what's implemented,
> what's intentionally stubbed (and why), and the concrete next steps.

## What's implemented

| Stage | Contents | Status |
|---|---|---|
| 1. Fundamental Kernels | vector_add, saxpy, dot_product, reduction, scan, histogram | ✅ |
| 2. Memory Optimization | coalescing / shared memory / bank-conflict avoidance / warp shuffle — folded into Stage 1's optimized kernels + dedicated `transpose` | ✅ |
| 3. Matrix Computation | naive → shared-mem tiled → register-tiled GEMM, vs. cuBLAS | 🟡 (WMMA/CUTLASS/double-buffering scoped in roadmap) |
| 4. DL Operators | softmax, layernorm (Welford), gelu, conv2d | 🟡 (depthwise conv + FlashAttention-lite scoped in roadmap) |
| 5. Runtime Optimization | pinned memory, stream overlap, CUDA Graphs | ✅ (persistent kernels scoped in roadmap) |
| 6. PyTorch Extension | `torch.ops.cucore.my_softmax`, `my_layernorm` | 🟡 (`my_matmul` scoped in roadmap) |

## Layout

```
cuCore/
  common/            shared CUDA error-checking, timing, correctness helpers
  kernel/            one folder per primitive, each a self-contained
                      benchmark+test executable (CPU baseline -> naive ->
                      optimized -> benchmark -> correctness check)
  python/             PyTorch CUDA extension (Stage 6)
  docs/roadmap.md      what's done, what's next, and why
  profile/            Nsight Compute / Nsight Systems command reference
  report/             fill-in-the-blank optimization report + roofline plot script
```

## Build & run (C++ / CUDA kernels)

Requires CMake >= 3.18 and the CUDA Toolkit (nvcc + cuBLAS) installed.

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86   # set to YOUR GPU's compute capability
make -j
./bin/vector_add        # run one kernel's benchmark
make bench               # or run everything back-to-back
```

Each binary is self-contained: it runs the CPU baseline, the naive kernel,
the optimized kernel, checks correctness against the CPU reference, prints
timing/bandwidth/GFLOPs, and exits non-zero if any correctness check fails
(useful for wiring into CI).

## Build & run (PyTorch extension, Stage 6)

Requires a working PyTorch + CUDA install matching your toolkit.

```bash
cd python
pip install -e .
python test_ops.py
```

## Profiling

See [`profile/ncu_commands.md`](profile/ncu_commands.md) for the exact
Nsight Compute invocation and metric list for every kernel, and Nsight
Systems guidance for the Stage 5 streams/overlap demo (which `ncu` alone
won't usefully characterize).

## Reporting your results

Copy [`report/optimization_report_template.md`](report/optimization_report_template.md)
per kernel (or per stage) and fill it in with your real measured numbers —
don't ship a report with placeholder/estimated figures. Use
[`report/roofline_plot.py`](report/roofline_plot.py) to generate an actual
roofline plot once you have real GFLOP/s and GB/s numbers for your GPU.

## Why this project

Modern DL frameworks hide GPU execution behind high-level APIs. This project
exists to close the gap between "I can call a CUDA kernel" and "I understand
why cuBLAS/cuDNN are fast" — the same fundamentals underlying work at
NVIDIA's CUDA libraries team, Google XLA/Pallas, and cloud ML-accelerator
compiler/runtime teams.
