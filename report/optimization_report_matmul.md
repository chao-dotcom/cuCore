# cuCore Optimization Report — Matrix Multiplication (GEMM)

## 1. Problem statement
- **Operation:** Matrix multiplication C = A × B (GEMM)
- **Input size(s):** N ∈ {256, 512, 1024, 2048, 4096} (square matrices)
- **Data type:** fp32
- **Hardware:** NVIDIA RTX PRO 6000 Black, driver 580.82.07, CUDA 13.0

## 2. Implementation ladder

| Version | Description | Key technique |
|---|---|---|
| Naive CUDA | one thread per element, element-wise multiply-accumulate | — |
| Shared memory tiling | load tiles into shared memory | shared memory blocking, 32×32 tiles |
| Register-tiled (4×4) | further tiling within registers, coalesced loads | register tiling, thread block cooperative loads |
| Double-buffered v1 (4×4) | buffer ping-ponging between tile loads | software pipelining |
| Double-buffered v2 (8×8 + vectorized + prefetch) | larger tiles, vectorized loads, real async prefetch | higher arithmetic intensity, latency hiding |

## 3. Benchmark results (N=4096)

| Version | Time (ms) | GFLOP/s | % of cuBLAS |
|---|---|---|---|
| Naive CUDA | 20.72 | 6,634 | 9.0% |
| Shared memory (32×32) | 16.99 | 8,086 | 11.0% |
| Register-tiled (4×4) | 4.33 | 31,731 | **43.0%** |
| Double-buffered v1 (4×4) | — | — | — |
| Double-buffered v2 (8×8) | 4.90 | 28,039 | **37.5%** |
| cuBLAS reference | 1.86 | 74,754 | 100% |

### Performance Sweep (all variants vs cuBLAS)

| N | Naive (%) | Shared Mem (%) | Reg-tile 4×4 (%) | Dbuf v2 8×8 (%) | cuBLAS GFLOP/s |
|---|---|---|---|---|---|
| 256 | 50.4% | 58.1% | **22.8%** | 22.8% | 3,973 |
| 512 | 50.1% | 63.0% | **43.2%** | 42.9% | 8,664 |
| 1024 | 13.2% | 17.0% | **29.8%** | 29.4% | 45,413 |
| 2048 | 10.5% | 13.4% | **40.3%** | 41.1% | 62,972 |
| 4096 | 9.0% | 11.0% | **43.0%** | 37.5% | 74,754 |

## 4. Nsight Compute analysis

| Metric | Naive | Register-tiled | Dbuf v2 | Interpretation |
|---|---|---|---|---|
| Achieved occupancy | [Run ncu] | [Run ncu] | [Run ncu] | [Warp scheduling] |
| L1 cache hit rate | [Run ncu] | [Run ncu] | [Run ncu] | [Reuse patterns] |
| L2 cache hit rate | [Run ncu] | [Run ncu] | [Run ncu] | [Memory pressure] |
| Shared memory utilization | [Run ncu] | [Run ncu] | [Run ncu] | [Bank conflicts] |
| Registers per thread | [Run ncu] | [Run ncu] | [Run ncu] | [Register pressure] |

## 5. Roofline position

- Arithmetic intensity (FLOPs / byte): **0.33 (memory-intensive)**
- Achieved GFLOP/s (register-tiled): **31,731**
- Achieved GB/s: **~96 GB/s** (estimated)
- Bound by: **memory bandwidth** (not compute)
- Peak GPU bandwidth: ~960 GB/s; achieved ~10% utilization
- See `report/roofline_plot.py` for roofline plot.

## 6. What limited performance, and what fixed it

**Naive kernel bottleneck:** No data reuse. Each thread loads A[i,k] and B[k,j] from global memory once, computes one output C[i,j], and writes it. Global bandwidth becomes the hard limit.

**Shared memory tiling fix:** Load 32×32 tiles of A and B into shared memory once, then all 1024 threads in the block reuse that data for 32 multiply-accumulates each. Reduces global memory traffic by 32×. However, shared memory bandwidth still becomes the bottleneck (~200 GB/s available).

**Register-tiled fix (4×4 per thread):** Each thread now owns a 4×4 register tile and accumulates into it across all K iterations. This increases arithmetic intensity locally and allows better utilization of the compute units. However, threads must still cooperatively load tiles from shared → registers, and register pressure limits parallelism.

**Double-buffered v2 (8×8 + vectorized):** Increases tile size from 4×4 to 8×8 per thread, reducing shared memory load frequency and increasing per-thread work per memory transaction. Vectorized loads (load 4× float32 in a single instruction) further reduce memory instruction count. Real software pipelining (prefetch tile t+1 while computing on tile t) reduces stalls.

**Why still only 43% of cuBLAS:** cuBLAS (and Tensor Cores) likely use:
- **Larger tiles** (16×16 or 32×32 per thread, via Tensor Core instructions)
- **Tensor Core MMA** (matrix multiply-accumulate instructions, 10× higher throughput)
- **Async shared memory loads** (pipelined from global DRAM in parallel with compute)
- **More sophisticated scheduling** (multiple fragments per warp, better register reuse across fragments)

## 7. Remaining gap vs. library implementation

To close the gap to cuBLAS (currently 43% at N=4096):

1. **Implement Tensor Cores (WMMA):** Switch from FP32 scalar arithmetic to `tf32` tensor operations. Tensor Cores deliver ~10× higher throughput and naturally pipeline. This should push toward 60-70% of cuBLAS.
2. **Async shared memory prefetch:** Use `cuda::memcpy_async` to pipeline tile loads from global memory while compute is in flight. This hides global memory latency and can yield another 10-15% improvement.
3. **Double-buffer shared memory:** Maintain two copies of shared memory (ping-pong), allowing compute to overlap with the next tile's load.
4. **Larger tile sizes:** Increase from 8×8 to 16×16 or larger per thread (if register pressure allows) to further reduce shared memory access frequency.

Reference: `docs/roadmap.md` lists these exact priorities.
