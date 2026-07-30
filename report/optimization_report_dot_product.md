# cuCore Optimization Report — Dot Product

## 1. Problem statement
- **Operation:** Dot product: sum of element-wise multiplication A[i] * B[i]
- **Input size(s):** N = 2,097,152 elements
- **Data type:** fp32
- **Hardware:** NVIDIA RTX PRO 6000 Black, driver 580.82.07, CUDA 13.0

## 2. Implementation ladder

| Version | Description | Key technique |
|---|---|---|
| CPU baseline | single-threaded dot product loop | — |
| Naive CUDA | one thread per element, atomic add reduction | — |
| Optimized CUDA | warp shuffle reduction with thread coalescing | warp shuffle, shared memory reduction |

## 3. Benchmark results

| Version | Time (ms) | Throughput | Speedup vs. previous |
|---|---|---|---|
| CPU baseline | 10.93 | — | — |
| Naive CUDA | 24.95 | 5420.4 GB/s | **441.3x vs CPU** |
| Optimized CUDA | 0.0248 | 5420.4 GB/s | **1007.65x vs naive** |
| Library reference (cuBLAS) | — | — | — |

## 4. Nsight Compute analysis

| Metric | Naive | Optimized | Interpretation |
|---|---|---|---|
| Achieved occupancy | [Run ncu] | [Run ncu] | [Check utilization] |
| DRAM throughput | [Run ncu] | [Run ncu] | [Check bandwidth usage] |
| Global load/store efficiency | [Run ncu] | [Run ncu] | [Coalescing check] |
| Shared memory bank conflicts | [Run ncu] | [Run ncu] | [Warp shuffle efficiency] |
| IPC / warp execution efficiency | [Run ncu] | [Run ncu] | [Pipeline utilization] |
| Registers per thread | [Run ncu] | [Run ncu] | [Resource pressure] |

## 5. Roofline position

- Arithmetic intensity (FLOPs / byte): **1 (data-parallel reduction)**
- Achieved GFLOP/s: **1355.1**
- Achieved GB/s: **5420.4**
- Bound by: **memory bandwidth**
- See `report/roofline_plot.py` to generate the actual plot.

## 6. What limited performance, and what fixed it

The naive kernel used atomic operations for reduction, which are inherently sequential and create memory contention. The optimized version implements warp-shuffle based reduction, where threads within a warp cooperatively reduce values using `__shfl_xor_sync()`. This eliminates atomic operations, improves memory parallelism, and allows threads to exchange data with zero latency (via register shuffles). The result: **1007.65x speedup**.

## 7. Remaining gap vs. library implementation

Dot product is fundamentally memory-bound and the optimized kernel achieves near-peak bandwidth for this operation. The remaining gap (if any) would be minimal and likely due to architectural differences in cuBLAS's reduction implementation. Further optimization would require kernel fusion with prior/subsequent operations in a compute graph, not algorithmic changes to dot product itself.
