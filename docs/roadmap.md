# cuCore Roadmap

This tracks what's implemented, what's stubbed, and what the natural next
step is for each stage. Built to run on a real NVIDIA GPU (Volta/Turing/
Ampere/Ada/Hopper) — this repo was authored in a sandbox with no GPU
attached, so **you are the first person to compile and run it**. Expect to
spend your first session fixing whatever doesn't compile cleanly on your
specific `nvcc`/driver/arch combination before you start collecting numbers.

## Stage 1 — Fundamental Kernels ✅ implemented
`vector_add`, `saxpy`, `dot_product`, `reduction`, `scan`, `histogram` — each
has CPU baseline → naive CUDA → optimized CUDA → benchmark, with correctness
checked against the CPU reference.

**Next steps once running:**
- Sweep `N` across several orders of magnitude and plot achieved bandwidth
  vs. array size to find the "knee" where the GPU becomes compute/latency
  bound instead of bandwidth bound (small-N launch overhead dominates).
- Compare against `thrust::reduce` / `cub::DeviceReduce::Sum` and
  `cub::DeviceScan::InclusiveSum` as a "how close to a professional library
  did I get" ceiling — this is explicitly called out in the project brief.

## Stage 2 — Memory Optimization ✅ implemented (folded into Stage 1/transpose)
Coalescing, shared memory, bank-conflict avoidance, and warp shuffle are
demonstrated directly inside the Stage 1 kernels' "optimized" variants and in
the dedicated `transpose` kernel (the canonical bank-conflict teaching
example). Register tiling is demonstrated in Stage 3's GEMM.

**Next steps once running:**
- Use Nsight Compute (`profile/ncu_commands.md`) to directly measure
  `smsp__average_data_bytes_per_wavefront_mem_shared` before/after adding
  the `+1` padding in `transpose_opt_kernel` — you should see the "shared
  store bank conflict" metric drop from >1 to ~1 conflict-free.

## Stage 3 — Matrix Computation 🟡 implemented through register tiling
`matmul.cu` implements naive → shared-memory tiled → 2D register-tiled GEMM,
benchmarked against `cublasSgemm`.

**Explicitly NOT implemented yet (by design, not oversight):**
- **Double buffering** (prefetching tile `t+1` into shared memory while
  computing on tile `t`, ping-ponging two shared-memory buffers to hide
  global-memory latency behind compute). This is a mechanical extension of
  `matmul_regtile_kernel` — duplicate `As`/`Bs` into `As[2]`/`Bs[2]`, issue
  the next tile's global loads before the current tile's inner-product loop,
  and swap buffer indices each iteration.
- **WMMA Tensor Core path.** Requires Volta+ (`sm_70`) and needs data in
  `half`/`bf16`/`tf32` to actually engage the tensor cores — writing this
  blind (no GPU to validate numerics/perf on) risks shipping a kernel that
  either silently produces wrong results or "compiles but never actually
  hits the tensor core path," which would be worse than not including it.
  The concrete next step: convert A/B to `half`, use
  `nvcuda::wmma::fragment<matmul_a/b/accumulator, 16,16,16,...>`, load with
  `wmma::load_matrix_sync`, accumulate with `wmma::mma_sync`, and compare
  against `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
- **CUTLASS comparison.** Once WMMA is in, benchmark against a CUTLASS
  `Gemm` instantiation of matching tile shape as the "how close to a
  templated, tuned library did my hand-written kernel get" ceiling.

## Stage 4 — DL Operators ✅ implemented (minus FlashAttention-lite)
`softmax`, `layernorm` (Welford), `gelu`, `conv2d` (shared-memory + halo +
constant-memory filter) are all implemented with the naive→optimized→
benchmark pipeline.

**Explicitly NOT implemented yet:**
- **Depthwise Conv2D** — mechanically: change `conv2d_opt_kernel` so the
  filter and input both gain a channel dimension, and each block handles a
  single channel's spatial tile independently (no cross-channel reduction,
  which is what makes depthwise conv cheaper than full conv2d in the first
  place).
- **FlashAttention-lite** — this is the most involved item in the whole
  roadmap. Concretely: implement scaled dot-product attention
  `softmax(QK^T / sqrt(d)) V` for a single head, tiled over K/V blocks, using
  the **online softmax** trick (track running max `m` and running sum `l`,
  rescale the accumulated output `O` every time a new K/V tile shifts the
  max) so you never materialize the full `[seq_len, seq_len]` attention
  matrix. Structure: one block per query tile; loop over K/V tiles staged
  through shared memory; update `(m, l, O)` online per the FlashAttention
  paper's Algorithm 1. Validate against a naive
  `softmax(Q @ K.T / sqrt(d)) @ V` in PyTorch for a small sequence length
  first, then scale up and profile memory traffic reduction vs. the naive
  materialized-attention-matrix version.

## Stage 5 — Runtime Optimization ✅ implemented
`streams_demo.cu` covers pinned vs. pageable memory, single-stream vs.
multi-stream compute/copy overlap, and CUDA Graph replay vs. direct
per-iteration launch.

**Next steps once running:**
- **Persistent kernels** are the one item here not yet implemented: a
  kernel launched once that loops internally (grid-stride over a *work
  queue* rather than a fixed problem size, using `cuda::atomic` or
  `atomicAdd` on a global work-counter to pull the next chunk of work),
  staying resident on the SMs instead of being relaunched. Useful for
  latency-sensitive serving workloads where kernel-launch latency itself
  (even with graphs) is a bottleneck. Worth adding once you have a real
  workload (e.g. Stage 6's ops) to make persistent.

## Stage 6 — PyTorch Extension ✅ implemented (2 of 3 ops)
`python/csrc/ops.cu` exposes `my_softmax` and `my_layernorm` as both a
pybind11 module (`import cucore_ops`) and a `torch.ops.cucore.*` custom op
(via `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL`), reusing the exact warp-shuffle /
Welford kernels from Stages 4.

**Next step:** add `my_matmul` by wrapping `matmul_regtile_kernel` from
Stage 3 the same way (see `ops.cu`'s existing two ops as the template —
`CHECK_INPUT` macros, kernel launch on `at::cuda::getCurrentCUDAStream()`,
`TORCH_LIBRARY_IMPL` registration). This was left for last deliberately:
building it after double-buffering/WMMA lands means you can expose the
*fastest* version of GEMM you have, not the naive one.

## Suggested order of operations once you have a GPU box

1. `mkdir build && cd build && cmake .. && make -j` — fix any compiler
   version / architecture mismatches for your specific hardware.
2. Run every `Stage 1`/`Stage 2` binary, confirm `PASS` on all correctness
   checks, then start profiling with `profile/ncu_commands.md`.
3. Move to `matmul`, get correctness passing, THEN start tuning tile sizes
   for your specific GPU's shared memory size / register file.
4. Stage 4 kernels, same pattern.
5. Stage 5 demo — note that timings here are far more sensitive to your
   specific GPU's copy-engine count (1 vs. 2) and PCIe/NVLink generation
   than the compute kernels are.
6. `pip install -e python/` and run `python/test_ops.py` last, once you
   trust the underlying kernels.
7. Fill in `report/optimization_report_template.md` per kernel with your
   actual measured numbers and Nsight Compute screenshots/metrics.
