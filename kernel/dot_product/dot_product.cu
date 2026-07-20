// cuCore/kernel/dot_product/dot_product.cu
//
// Stage 1 — Dot Product: result = sum(a[i] * b[i])
//
// This kernel is the first one in the repo that requires a *reduction*
// across threads, so it foreshadows Stage 2's warp-shuffle techniques.
//
// Naive:     every thread does one global atomicAdd into the result.
//            Correct, but every single thread serializes on the same
//            memory location -> massive contention, terrible scaling.
// Optimized: each thread computes a partial product, block-level tree
//            reduction in shared memory down to one warp, then
//            warp-shuffle reduction (no shared memory, no __syncthreads)
//            for the final 32 values, then one atomicAdd per BLOCK
//            instead of one per THREAD.

#include "../../common/cuda_utils.cuh"

float dot_cpu(const float* a, const float* b, size_t n) {
    double acc = 0.0; // accumulate in double for a fair/stable CPU reference
    for (size_t i = 0; i < n; ++i) acc += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(acc);
}

// ---------------------------------------------------------------------------
// Naive: global atomicAdd per thread
// ---------------------------------------------------------------------------
__global__ void dot_naive_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ result, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(result, a[i] * b[i]);
}

// ---------------------------------------------------------------------------
// Optimized: grid-stride partials -> shared-mem tree reduce -> warp shuffle
// -> one atomicAdd per block.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Butterfly warp-shuffle reduction: no shared memory, no bank conflicts,
    // no __syncthreads(); XOR pattern folds the warp in log2(32)=5 steps.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <int BLOCK_SIZE>
__global__ void dot_opt_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ result, size_t n) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];

    size_t stride = (size_t)blockDim.x * gridDim.x;
    float partial = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride)
        partial += a[i] * b[i];

    // Reduce within each warp first (cheap, no synchronization needed).
    partial = warp_reduce_sum(partial);

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();

    // First warp reduces the per-warp partial sums.
    if (warp_id == 0) {
        float v = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) atomicAdd(result, v); // 1 atomic per BLOCK, not per THREAD
    }
}

int main(int argc, char** argv) {
    const size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 24);
    std::printf("Dot Product benchmark, N = %zu elements\n", N);

    std::vector<float> h_a(N), h_b(N);
    fill_random(h_a, -1.f, 1.f, 1);
    fill_random(h_b, -1.f, 1.f, 2);

    float cpu_result = 0.f;
    double cpu_ms = benchmark_cpu([&]() { cpu_result = dot_cpu(h_a.data(), h_b.data(), N); }, 1, 3);

    float *d_a, *d_b, *d_result;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;
    int grid = static_cast<int>((N + BLOCK - 1) / BLOCK);

    float gpu_result_naive = 0.f;
    float naive_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        dot_naive_kernel<<<grid, BLOCK>>>(d_a, d_b, d_result, N);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(&gpu_result_naive, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    int sm_count;
    { int dev; CUDA_CHECK(cudaGetDevice(&dev));
      cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
      sm_count = prop.multiProcessorCount; }
    int opt_grid = sm_count * 16;

    float gpu_result_opt = 0.f;
    float opt_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        dot_opt_kernel<BLOCK><<<opt_grid, BLOCK>>>(d_a, d_b, d_result, N);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(&gpu_result_opt, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    double naive_err = std::fabs(gpu_result_naive - cpu_result);
    double opt_err = std::fabs(gpu_result_opt - cpu_result);
    // Reduction order differs from CPU (float32 accumulation, non-associative),
    // so we use a looser relative tolerance here than the elementwise kernels.
    double tol = 1e-2 * std::fabs(cpu_result) + 1e-2;

    BenchResult r;
    r.name = "dot_product";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = 2.0 * N * sizeof(float);
    r.flops = 2.0 * N; // 1 mul + 1 add per element
    print_bench_result(r);
    std::printf("  CPU result=%.6f  naive=%.6f (err=%.6f)  optimized=%.6f (err=%.6f)\n",
                cpu_result, gpu_result_naive, naive_err, gpu_result_opt, opt_err);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                (naive_err < tol) ? "PASS" : "FAIL", (opt_err < tol) ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    return ((naive_err < tol) && (opt_err < tol)) ? 0 : 1;
}
