// cuCore/kernel/reduction/reduction.cu
//
// Stage 1 — Parallel Reduction (sum of an array), two-pass:
//   pass 1: each block reduces its chunk to one partial sum
//   pass 2: reduce the (small) array of partial sums on CPU or with a
//           second kernel launch (we do a second launch here to keep
//           everything on-device).
//
// This is deliberately the "classic Mark Harris reduction ladder" kernel:
//   Naive     -> interleaved addressing with %, divergent branches, and
//                shared-memory bank conflicts (2-way, 4-way, ... as offset
//                grows), the textbook "what NOT to do" reduction.
//   Optimized -> sequential addressing (no divergence, no bank conflicts),
//                first-add-during-load (halves the blocks needed), unrolled
//                last warp via warp shuffle (no __syncthreads needed once
//                we're down to 32 threads), grid-stride accumulation so one
//                thread sums many elements before the tree reduction begins.

#include "../../common/cuda_utils.cuh"

float reduce_cpu(const float* a, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) acc += a[i];
    return static_cast<float>(acc);
}

// ---------------------------------------------------------------------------
// Naive: interleaved addressing.
//   - `if (tid % (2*s) == 0)` -> half the threads in a warp are idle and
//     diverge every iteration (warp divergence).
//   - stride `s` doubling every step means threads access shared memory
//     addresses `tid, tid+s, tid+2s...` which collide into the same bank
//     for many values of s (bank conflicts), especially as s grows past 1.
// ---------------------------------------------------------------------------
__global__ void reduce_naive_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Optimized:
//   - Grid-stride accumulation: each thread first sums a whole strided
//     range of the input into a register before ever touching shared
//     memory, so we launch far fewer blocks than N/blockDim and still
//     touch every element (this is "first add during load", generalized).
//   - Sequential addressing tree (`s = blockDim/2` down to 1, halving):
//     thread `tid` reads `sdata[tid]` and `sdata[tid+s]` -- adjacent
//     threads read adjacent/non-colliding banks, no divergence, no
//     conflicts.
//   - Last warp (s <= 32) reduced with warp shuffle intrinsics instead of
//     shared memory: no __syncthreads() needed since a warp executes in
//     lockstep (register-to-register exchange).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <int BLOCK_SIZE>
__global__ void reduce_opt_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;

    // Grid-stride pre-accumulation: fold the whole array down using far
    // fewer blocks/threads than one-thread-per-element would need.
    size_t stride = (size_t)blockDim.x * gridDim.x;
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + tid; i < n; i += stride)
        acc += in[i];
    sdata[tid] = acc;
    __syncthreads();

    // Sequential-addressing tree reduction down to 1 warp.
#pragma unroll
    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Final warp: shuffle-based reduction, no shared memory / syncthreads.
    float val = (tid < 32) ? sdata[tid] + sdata[tid + 32] : 0.f;
    if (tid < 32) val = warp_reduce_sum(val);

    if (tid == 0) out[blockIdx.x] = val;
}

int main(int argc, char** argv) {
    const size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 25);
    std::printf("Reduction (sum) benchmark, N = %zu elements\n", N);

    std::vector<float> h_in(N);
    fill_random(h_in, -1.f, 1.f, 7);
    float cpu_result = 0.f;
    double cpu_ms = benchmark_cpu([&]() { cpu_result = reduce_cpu(h_in.data(), N); }, 1, 3);

    float* d_in;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;

    // ---- naive: one block per BLOCK elements, second pass reduces partials on host ----
    int naive_grid = static_cast<int>((N + BLOCK - 1) / BLOCK);
    float* d_partial_naive;
    CUDA_CHECK(cudaMalloc(&d_partial_naive, naive_grid * sizeof(float)));
    std::vector<float> h_partial_naive(naive_grid);
    float gpu_result_naive = 0.f;
    float naive_ms = benchmark_gpu([&]() {
        reduce_naive_kernel<<<naive_grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_partial_naive, N);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_partial_naive.data(), d_partial_naive, naive_grid * sizeof(float), cudaMemcpyDeviceToHost));
    for (float v : h_partial_naive) gpu_result_naive += v;

    // ---- optimized: few blocks (grid-stride), second pass reduces the small partial array ----
    int sm_count;
    { int dev; CUDA_CHECK(cudaGetDevice(&dev));
      cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
      sm_count = prop.multiProcessorCount; }
    int opt_grid = sm_count * 8;
    float* d_partial_opt;
    CUDA_CHECK(cudaMalloc(&d_partial_opt, opt_grid * sizeof(float)));
    std::vector<float> h_partial_opt(opt_grid);
    float gpu_result_opt = 0.f;
    float opt_ms = benchmark_gpu([&]() {
        reduce_opt_kernel<BLOCK><<<opt_grid, BLOCK>>>(d_in, d_partial_opt, N);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_partial_opt.data(), d_partial_opt, opt_grid * sizeof(float), cudaMemcpyDeviceToHost));
    for (float v : h_partial_opt) gpu_result_opt += v;

    double naive_err = std::fabs(gpu_result_naive - cpu_result);
    double opt_err = std::fabs(gpu_result_opt - cpu_result);
    double tol = 1e-2 * std::fabs(cpu_result) + 1e-2;

    BenchResult r;
    r.name = "reduction";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = static_cast<double>(N) * sizeof(float);
    print_bench_result(r);
    std::printf("  CPU=%.6f  naive=%.6f (err=%.6f)  optimized=%.6f (err=%.6f)\n",
                cpu_result, gpu_result_naive, naive_err, gpu_result_opt, opt_err);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                (naive_err < tol) ? "PASS" : "FAIL", (opt_err < tol) ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial_naive));
    CUDA_CHECK(cudaFree(d_partial_opt));
    return ((naive_err < tol) && (opt_err < tol)) ? 0 : 1;
}
