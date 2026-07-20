// cuCore/kernel/vector_add/vector_add.cu
//
// Stage 1 — Fundamental Kernels: Vector Addition  (C = A + B)
//
// Pipeline: CPU baseline -> Naive CUDA -> Optimized (grid-stride, coalesced) -> Benchmark
//
// This is the simplest kernel in the library and exists mainly to establish
// the benchmarking harness pattern used by every other kernel: memory-bound,
// so we report *effective bandwidth* rather than FLOP/s.

#include "../../common/cuda_utils.cuh"

// ---------------------------------------------------------------------------
// CPU baseline
// ---------------------------------------------------------------------------
void vector_add_cpu(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Naive CUDA kernel
//   - One thread per element.
//   - Global loads/stores ARE already coalesced here because consecutive
//     threads touch consecutive addresses -- the "naive" weakness in this
//     particular kernel isn't memory layout, it's that we launch exactly
//     ceil(n/block) threads with no grid-stride loop, so very large n
//     requires the launch to already know the full extent and we cannot
//     amortize kernel-launch overhead across multiple waves.
// ---------------------------------------------------------------------------
__global__ void vector_add_naive_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Optimized CUDA kernel
//   - Grid-stride loop: launch only as many threads as the GPU can resident
//     at once, let each thread process multiple elements. This decouples
//     problem size from launch configuration, avoids re-launch overhead for
//     huge n, and keeps all SMs saturated regardless of input size.
//   - float4 vectorized loads/stores: each thread moves 16B at a time instead
//     of 4B, cutting the instruction count for memory ops by 4x and better
//     saturating the memory bus width per transaction.
// ---------------------------------------------------------------------------
__global__ void vector_add_opt_kernel(const float4* __restrict__ a,
                                       const float4* __restrict__ b,
                                       float4* __restrict__ c,
                                       size_t n_vec4) {
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_vec4; i += stride) {
        float4 av = a[i];
        float4 bv = b[i];
        float4 cv;
        cv.x = av.x + bv.x;
        cv.y = av.y + bv.y;
        cv.z = av.z + bv.z;
        cv.w = av.w + bv.w;
        c[i] = cv;
    }
}

int main(int argc, char** argv) {
    const size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 25); // ~33M elems
    std::printf("Vector Add benchmark, N = %zu elements (%.2f MB per buffer)\n",
                N, N * sizeof(float) / 1e6);

    std::vector<float> h_a(N), h_b(N), h_c_ref(N), h_c_gpu(N);
    fill_random(h_a);
    fill_random(h_b);

    // ---- CPU baseline ----
    double cpu_ms = benchmark_cpu([&]() { vector_add_cpu(h_a.data(), h_b.data(), h_c_ref.data(), N); }, 1, 3);

    // ---- Device buffers ----
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Naive kernel ----
    int block = 256;
    int grid = static_cast<int>((N + block - 1) / block);
    float naive_ms = benchmark_gpu([&]() {
        vector_add_naive_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    });
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool naive_ok = allclose(h_c_gpu.data(), h_c_ref.data(), N);

    // ---- Optimized kernel (float4, grid-stride) ----
    CUDA_CHECK(cudaMemset(d_c, 0, N * sizeof(float)));
    size_t n_vec4 = N / 4; // assume N % 4 == 0 for the vectorized path
    int sm_count = 0;
    {
        int dev; CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        sm_count = prop.multiProcessorCount;
    }
    int opt_block = 256;
    int opt_grid = sm_count * 32; // enough blocks to fill the GPU, grid-stride handles the rest
    float opt_ms = benchmark_gpu([&]() {
        vector_add_opt_kernel<<<opt_grid, opt_block>>>(
            reinterpret_cast<const float4*>(d_a),
            reinterpret_cast<const float4*>(d_b),
            reinterpret_cast<float4*>(d_c),
            n_vec4);
    });
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool opt_ok = allclose(h_c_gpu.data(), h_c_ref.data(), N);

    BenchResult r;
    r.name = "vector_add";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = 3.0 * N * sizeof(float); // read A, read B, write C
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return (naive_ok && opt_ok) ? 0 : 1;
}
