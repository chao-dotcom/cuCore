// cuCore/kernel/saxpy/saxpy.cu
//
// Stage 1 — SAXPY: Y = alpha * X + Y
//
// Classic BLAS Level-1 op. Memory-bound (2 reads + 1 write per element,
// 2 FLOPs per element), so this is a good vehicle for discussing
// arithmetic intensity: AI = 2 FLOPs / 12 bytes ~= 0.17 FLOP/byte,
// meaning performance is governed almost entirely by DRAM bandwidth,
// not compute throughput -- a textbook roofline "memory-bound" case.

#include "../../common/cuda_utils.cuh"

void saxpy_cpu(float alpha, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = alpha * x[i] + y[i];
}

// Naive: one thread per element, scalar loads.
__global__ void saxpy_naive_kernel(float alpha, const float* __restrict__ x,
                                    float* __restrict__ y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];
}

// Optimized: grid-stride + float4 vectorized load/store, same rationale as
// vector_add. This lets each thread issue one 128-bit load/store transaction
// instead of four 32-bit ones, improving memory-controller efficiency.
__global__ void saxpy_opt_kernel(float alpha, const float4* __restrict__ x,
                                  float4* __restrict__ y, size_t n_vec4) {
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_vec4; i += stride) {
        float4 xv = x[i];
        float4 yv = y[i];
        yv.x = alpha * xv.x + yv.x;
        yv.y = alpha * xv.y + yv.y;
        yv.z = alpha * xv.z + yv.z;
        yv.w = alpha * xv.w + yv.w;
        y[i] = yv;
    }
}

int main(int argc, char** argv) {
    const size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 25);
    const float alpha = 2.5f;
    std::printf("SAXPY benchmark, N = %zu elements\n", N);

    std::vector<float> h_x(N), h_y(N), h_y_ref(N), h_y_gpu(N);
    fill_random(h_x);
    fill_random(h_y);
    h_y_ref = h_y;

    double cpu_ms = benchmark_cpu([&]() {
        h_y_ref = h_y; // reset since saxpy mutates y in place
        saxpy_cpu(alpha, h_x.data(), h_y_ref.data(), N);
    }, 1, 3);

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = static_cast<int>((N + block - 1) / block);
    float naive_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        saxpy_naive_kernel<<<grid, block>>>(alpha, d_x, d_y, N);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool naive_ok = allclose(h_y_gpu.data(), h_y_ref.data(), N);

    int sm_count;
    { int dev; CUDA_CHECK(cudaGetDevice(&dev));
      cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
      sm_count = prop.multiProcessorCount; }
    size_t n_vec4 = N / 4;
    int opt_grid = sm_count * 32;
    float opt_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        saxpy_opt_kernel<<<opt_grid, 256>>>(alpha, reinterpret_cast<const float4*>(d_x),
                                            reinterpret_cast<float4*>(d_y), n_vec4);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool opt_ok = allclose(h_y_gpu.data(), h_y_ref.data(), N);

    BenchResult r;
    r.name = "saxpy";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = 3.0 * N * sizeof(float); // read x, read y, write y
    r.flops = 2.0 * N;
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return (naive_ok && opt_ok) ? 0 : 1;
}
