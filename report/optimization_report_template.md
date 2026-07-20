# cuCore Optimization Report — [Kernel Name]

> Copy this template once per kernel (or once per stage) and fill in the
> bracketed fields with your actual measured numbers. Don't leave estimates
> in a final report — every number below should come from a real run on your
> hardware.

## 1. Problem statement
- **Operation:** [e.g. C = A + B, elementwise]
- **Input size(s):** [N = ..., or rows x cols = ...]
- **Data type:** [fp32 / fp16 / bf16]
- **Hardware:** [GPU model, e.g. RTX 4090 / A100 40GB], driver [ver], CUDA [ver]

## 2. Implementation ladder

| Version | Description | Key technique |
|---|---|---|
| CPU baseline | single-threaded reference | — |
| Naive CUDA | one thread per element/output | — |
| Optimized CUDA | [describe] | [coalescing / shared mem / warp shuffle / register tiling / etc.] |

## 3. Benchmark results

| Version | Time (ms) | Throughput | Speedup vs. previous |
|---|---|---|---|
| CPU baseline | [ ] | — | — |
| Naive CUDA | [ ] | [GB/s or GFLOP/s] | [ ]x vs CPU |
| Optimized CUDA | [ ] | [GB/s or GFLOP/s] | [ ]x vs naive |
| Library reference (cuBLAS/CUB/PyTorch), if applicable | [ ] | [ ] | [ ]% of library throughput |

## 4. Nsight Compute analysis

| Metric | Naive | Optimized | Interpretation |
|---|---|---|---|
| Achieved occupancy (`sm__warps_active.avg.pct_of_peak_sustained_active`) | [ ] | [ ] | [ ] |
| DRAM throughput (`dram__throughput.avg.pct_of_peak_sustained_elapsed`) | [ ] | [ ] | [ ] |
| Global load/store efficiency (coalescing) | [ ] | [ ] | [ ] |
| Shared memory bank conflicts | [ ] | [ ] | [ ] |
| IPC / warp execution efficiency | [ ] | [ ] | [ ] |
| Registers per thread | [ ] | [ ] | [ ] |

Attach or link Nsight Compute screenshots / exported `.ncu-rep` summaries here.

## 5. Roofline position

- Arithmetic intensity (FLOPs / byte): **[ ]**
- Achieved GFLOP/s: **[ ]**
- Achieved GB/s: **[ ]**
- Bound by: **[ memory bandwidth / compute throughput / latency ]**
- See `report/roofline_plot.py` to generate the actual plot once you have
  real numbers; insert the resulting image here.

## 6. What limited performance, and what fixed it

[Narrative: e.g. "The naive kernel's shared-memory tree reduction used
interleaved addressing (`tid % (2*s) == 0`), which both diverges within a
warp and, once `s` exceeds 1, causes N-way bank conflicts because
threads `tid` and `tid + s` frequently alias the same bank. Switching to
sequential addressing (`if (tid < s) ...`) removed the divergence and the
bank conflicts simultaneously, measured via `l1tex__data_bank_conflicts_...`
dropping from X to Y."]

## 7. Remaining gap vs. library implementation

[Narrative: what would it take to close the remaining gap to cuBLAS/CUB/
PyTorch — e.g. double buffering, Tensor Core MMA, better tile size tuning
for this specific GPU's shared memory / register file size, etc. Reference
`docs/roadmap.md` for the concrete next steps already scoped out.]
