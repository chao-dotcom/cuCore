# Nsight Compute (`ncu`) Profiling Reference

Every kernel deliverable requires a profiler pass. This file gives ready-to-run
`ncu` invocations per kernel and explains which metrics matter for that
kernel's specific bottleneck (memory-bound vs. compute-bound vs. latency-bound).

General pattern:

```bash
ncu --set full -o profile/reports/<kernel_name> ./build/bin/<kernel_name>
```

`--set full` collects everything; for quick iteration use `--set basic` first,
then `--set full` for the final report you'll screenshot into
`report/optimization_report_template.md`.

To profile only the OPTIMIZED kernel and skip the (identically-named-pattern)
naive one when a binary launches multiple kernels, filter by kernel name regex:

```bash
ncu --set full -k regex:.*_opt_kernel.* -o profile/reports/vector_add_opt ./build/bin/vector_add
```

---

## Stage 1 — memory-bound kernels: focus on bandwidth + coalescing

```bash
ncu --set full -k regex:.*vector_add.* -o profile/reports/vector_add ./build/bin/vector_add
ncu --set full -k regex:.*saxpy.*      -o profile/reports/saxpy      ./build/bin/saxpy
ncu --set full -k regex:.*dot_.*       -o profile/reports/dot        ./build/bin/dot_product
ncu --set full -k regex:.*reduce_.*    -o profile/reports/reduction  ./build/bin/reduction
ncu --set full -k regex:.*scan_.*      -o profile/reports/scan       ./build/bin/scan
ncu --set full -k regex:.*histogram_.* -o profile/reports/histogram  ./build/bin/histogram
```

Key metrics to record for each (naive vs. optimized):
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — % of peak HBM bandwidth achieved.
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` — coalescing quality (100% = perfectly coalesced).
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` — shared-memory bank conflicts (reduction/scan naive vs optimized should show a big drop here).
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy.

## Stage 2 — transpose: THE bank-conflict / coalescing demo

```bash
ncu --set full -k regex:.*transpose_.* -o profile/reports/transpose ./build/bin/transpose
```

Record, naive vs. optimized:
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` — write coalescing (naive should be terrible, ~1/32 of optimal; optimized should be ~100%).
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` — should be ~0 for the padded (+1) shared-memory version, nonzero without padding.

## Stage 3 — GEMM: compute-bound territory, IPC + tensor pipe utilization

```bash
ncu --set full -k regex:.*matmul_naive.*   -o profile/reports/matmul_naive   ./build/bin/matmul
ncu --set full -k regex:.*matmul_smem.*    -o profile/reports/matmul_smem    ./build/bin/matmul
ncu --set full -k regex:.*matmul_regtile.* -o profile/reports/matmul_regtile ./build/bin/matmul
```

Key metrics:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — overall SM utilization.
- `smsp__inst_executed_pipe_fma.avg.per_cycle_active` (IPC on the FMA pipe) — should climb naive → smem → regtile.
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` — smem kernel should already be conflict-free (32x32 tile, no bank aliasing at that stride); confirm.
- `launch__registers_per_thread` — watch this for the register-tiled kernel; too many registers per thread hurts occupancy (classic tiling trade-off — this is the number to report in your optimization writeup).
- Achieved GFLOP/s (printed by the binary itself) as % of `cublasSgemm`'s GFLOP/s and of your GPU's theoretical FP32 peak (`docs/roadmap.md` peak-FLOPs reference, or `nvidia-smi -q` / vendor spec sheet).

## Stage 4 — DL operators: mixed memory/compute, watch reduction overhead

```bash
ncu --set full -k regex:.*softmax_opt.*   -o profile/reports/softmax   ./build/bin/softmax
ncu --set full -k regex:.*layernorm_opt.* -o profile/reports/layernorm ./build/bin/layernorm
ncu --set full -k regex:.*gelu_opt.*      -o profile/reports/gelu      ./build/bin/gelu
ncu --set full -k regex:.*conv2d_opt.*    -o profile/reports/conv2d    ./build/bin/conv2d
```

Key metrics:
- softmax/layernorm: `smsp__average_warps_issue_stalled_barrier_per_issue_active` — stalls from the `__syncthreads()` in the warp-merge step; if this is high relative to total cycles, consider increasing `cols`-per-thread work to amortize the sync.
- gelu: this one is a good "pure compute intrinsic cost" demo — compare `smsp__sass_thread_inst_executed_op_fp32_pred_on.sum` between the `erff`-based and `tanhf`-based kernels to quantify the transcendental cost difference directly.
- conv2d: `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` (naive) vs. shared-memory version — should show a clear reduction proportional to how much overlap exists between adjacent threads' K×K windows.

## Stage 5 — runtime demos: NOT profiled with ncu

Streams/pinned-memory/graph timings are dominated by host-side orchestration
and PCIe/copy-engine behavior, which `ncu` (a per-kernel compute profiler)
won't usefully characterize. Use **Nsight Systems** (`nsys`) instead for a
timeline view showing actual overlap between copy engines and compute:

```bash
nsys profile -o profile/reports/streams_demo --stats=true ./build/bin/streams_demo
```

Open the resulting `.nsys-rep` in the Nsight Systems GUI and visually confirm
overlapping H2D/kernel/D2H bars across streams in Demo 2, and a single
compressed graph-launch region in Demo 3.

## Stage 6 — PyTorch extension

Profile via PyTorch's own profiler (captures both the custom kernel and any
framework-side overhead) rather than raw `ncu` on the Python process:

```bash
python -c "
import torch, cucore_ops
from torch.profiler import profile, ProfilerActivity
x = torch.randn(4096, 4096, device='cuda')
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        cucore_ops.my_softmax(x)
print(prof.key_averages().table(sort_by='cuda_time_total'))
"
```

Or, to still get full Nsight Compute kernel-level metrics on the underlying
CUDA kernel launched from Python:

```bash
ncu --set full -k regex:.*softmax_fwd_kernel.* -o profile/reports/torch_softmax \
    python python/test_ops.py
```
