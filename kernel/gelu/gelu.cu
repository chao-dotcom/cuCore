// cuCore/kernel/gelu/gelu.cu
//
// Stage 4 — DL Operator: GELU activation.
//   exact:  0.5 * x * (1 + erf(x / sqrt(2)))
//   approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))   (GPT-2/BERT's approximation)
//
// Naive:     one thread per element, scalar loads, exact erf-based formula
//            (erff() is a slow transcendental relative to tanh/exp).
// Optimized: grid-stride + float4 vectorized loads/stores (same rationale
//            as vector_add/saxpy) AND the tanh-approximation formula, which
//            is what PyTorch's `nn.GELU(approximate='tanh')` and most
//            production transformer stacks actually use, since it's
//            materially cheaper per element and the numerical difference
//            is negligible for training/inference purposes.

#include "../../common/cuda_utils.cuh"

void gelu_cpu(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        out[i] = 0.5f * x * (1.f + erff(x * 0.70710678f));
    }
}

__global__ void gelu_naive_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = 0.5f * x * (1.f + erff(x * 0.70710678f));
    }
}

__device__ __forceinline__ float gelu_tanh_approx(float x) {
    const float k0 = 0.7978845608f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x3)));
}

__global__ void gelu_opt_kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n_vec4) {
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_vec4; i += stride) {
        float4 v = in[i];
        float4 r;
        r.x = gelu_tanh_approx(v.x);
        r.y = gelu_tanh_approx(v.y);
        r.z = gelu_tanh_approx(v.z);
        r.w = gelu_tanh_approx(v.w);
        out[i] = r;
    }
}

int main(int argc, char** argv) {
    const size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 25);
    std::printf("GELU benchmark, N = %zu elements\n", N);

    std::vector<float> h_in(N), h_out_ref(N), h_out_gpu(N);
    fill_random(h_in, -4.f, 4.f, 55);
    double cpu_ms = benchmark_cpu([&]() { gelu_cpu(h_in.data(), h_out_ref.data(), N); }, 1, 3);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = static_cast<int>((N + block - 1) / block);
    float naive_ms = benchmark_gpu([&]() { gelu_naive_kernel<<<grid, block>>>(d_in, d_out, N); });
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    // Looser tolerance: exact-erf vs tanh-approximation formulas differ by design.
    bool naive_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-3, 1e-3);

    int sm_count;
    { int dev; CUDA_CHECK(cudaGetDevice(&dev));
      cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
      sm_count = prop.multiProcessorCount; }
    size_t n_vec4 = N / 4;
    int opt_grid = sm_count * 32;
    float opt_ms = benchmark_gpu([&]() {
        gelu_opt_kernel<<<opt_grid, 256>>>(reinterpret_cast<const float4*>(d_in),
                                           reinterpret_cast<float4*>(d_out), n_vec4);
    });
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool opt_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 3e-3, 3e-3); // approx formula: slightly looser

    BenchResult r;
    r.name = "gelu";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = 2.0 * N * sizeof(float);
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized(tanh-approx, looser tol)=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return (naive_ok && opt_ok) ? 0 : 1;
}
