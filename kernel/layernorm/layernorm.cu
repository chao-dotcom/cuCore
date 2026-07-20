// cuCore/kernel/layernorm/layernorm.cu
//
// Stage 4 — DL Operator: LayerNorm over the last dimension of a
// [rows, cols] matrix:  y = (x - mean) / sqrt(var + eps) * gamma + beta
//
// Naive:     two full passes over each row: pass 1 computes mean, pass 2
//            computes variance using that mean (needs mean already
//            finalized), pass 3 normalizes -- three reads of the row from
//            global memory across three kernel launches.
// Optimized: single kernel, one block per row, single pass over the row
//            using Welford's online algorithm to accumulate mean and
//            variance simultaneously in one sweep (no dependency on a
//            finalized mean before starting the variance accumulation),
//            with per-thread partial (mean, M2, count) triples merged
//            first via warp shuffle (Chan's parallel merge formula) and
//            then across warps via shared memory. Only 2 total passes over
//            the row (1 to compute stats, 1 to write normalized output),
//            versus the naive version's dependency chain across 3 kernels.

#include "../../common/cuda_utils.cuh"

void layernorm_cpu(const float* in, float* out, const float* gamma, const float* beta,
                    int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in = in + (size_t)r * cols;
        float* row_out = out + (size_t)r * cols;
        double mean = 0.0;
        for (int c = 0; c < cols; ++c) mean += row_in[c];
        mean /= cols;
        double var = 0.0;
        for (int c = 0; c < cols; ++c) { double d = row_in[c] - mean; var += d * d; }
        var /= cols;
        double inv_std = 1.0 / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c)
            row_out[c] = static_cast<float>((row_in[c] - mean) * inv_std) * gamma[c] + beta[c];
    }
}

// ---------------------------------------------------------------------------
// Naive: 2 reduction kernels (mean, then variance) + 1 normalize kernel.
// ---------------------------------------------------------------------------
__global__ void mean_kernel(const float* __restrict__ in, float* __restrict__ mean, int rows, int cols) {
    int r = blockIdx.x; if (r >= rows) return;
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += in[(size_t)r * cols + c];
    __shared__ float smem[256];
    smem[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) smem[threadIdx.x] += smem[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) mean[r] = smem[0] / cols;
}
__global__ void var_kernel(const float* __restrict__ in, const float* __restrict__ mean,
                            float* __restrict__ var, int rows, int cols) {
    int r = blockIdx.x; if (r >= rows) return;
    float m = mean[r], s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) { float d = in[(size_t)r * cols + c] - m; s += d * d; }
    __shared__ float smem[256];
    smem[threadIdx.x] = s; __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (threadIdx.x < o) smem[threadIdx.x] += smem[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) var[r] = smem[0] / cols;
}
__global__ void normalize_kernel(const float* __restrict__ in, float* __restrict__ out,
                                  const float* __restrict__ mean, const float* __restrict__ var,
                                  const float* __restrict__ gamma, const float* __restrict__ beta,
                                  int rows, int cols, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    float m = mean[r], inv_std = rsqrtf(var[r] + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        out[(size_t)r * cols + c] = (in[(size_t)r * cols + c] - m) * inv_std * gamma[c] + beta[c];
}

// ---------------------------------------------------------------------------
// Optimized: single-pass Welford, block-per-row, warp-shuffle merge.
// ---------------------------------------------------------------------------
struct WelfordState { float mean; float m2; float count; };

// Chan et al. parallel merge of two Welford accumulators.
__device__ __forceinline__ WelfordState welford_merge(WelfordState a, WelfordState b) {
    if (a.count == 0.f) return b;
    if (b.count == 0.f) return a;
    float count = a.count + b.count;
    float delta = b.mean - a.mean;
    float mean = a.mean + delta * (b.count / count);
    float m2 = a.m2 + b.m2 + delta * delta * (a.count * b.count / count);
    return {mean, m2, count};
}

__device__ __forceinline__ WelfordState warp_reduce_welford(WelfordState v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        WelfordState other;
        other.mean  = __shfl_xor_sync(0xffffffff, v.mean, o);
        other.m2    = __shfl_xor_sync(0xffffffff, v.m2, o);
        other.count = __shfl_xor_sync(0xffffffff, v.count, o);
        v = welford_merge(v, other);
    }
    return v;
}

template <int BLOCK_SIZE>
__global__ void layernorm_opt_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      const float* __restrict__ gamma, const float* __restrict__ beta,
                                      int rows, int cols, float eps) {
    __shared__ WelfordState warp_states[BLOCK_SIZE / 32];
    int r = blockIdx.x; if (r >= rows) return;
    const float* row_in = in + (size_t)r * cols;
    float* row_out = out + (size_t)r * cols;

    int lane = threadIdx.x % 32, warp_id = threadIdx.x / 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    // Single pass: accumulate (mean, M2, count) online -- no dependency on
    // a finalized mean before variance can start, unlike the naive version.
    WelfordState state{0.f, 0.f, 0.f};
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) {
        float x = row_in[c];
        state.count += 1.f;
        float delta = x - state.mean;
        state.mean += delta / state.count;
        state.m2 += delta * (x - state.mean);
    }
    state = warp_reduce_welford(state);
    if (lane == 0) warp_states[warp_id] = state;
    __syncthreads();
    if (warp_id == 0) {
        WelfordState v = (lane < NUM_WARPS) ? warp_states[lane] : WelfordState{0.f, 0.f, 0.f};
        v = warp_reduce_welford(v);
        if (lane == 0) warp_states[0] = v;
    }
    __syncthreads();
    float mean = warp_states[0].mean;
    float inv_std = rsqrtf(warp_states[0].m2 / cols + eps);
    __syncthreads();

    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE)
        row_out[c] = (row_in[c] - mean) * inv_std * gamma[c] + beta[c];
}

int main(int argc, char** argv) {
    const int rows = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int cols = (argc > 2) ? std::atoi(argv[2]) : 1024; // typical hidden size
    const float eps = 1e-5f;
    const size_t N = (size_t)rows * cols;
    std::printf("LayerNorm benchmark, [%d, %d] matrix\n", rows, cols);

    std::vector<float> h_in(N), h_out_ref(N), h_out_gpu(N), h_gamma(cols), h_beta(cols);
    fill_random(h_in, -3.f, 3.f, 44);
    fill_random(h_gamma, 0.5f, 1.5f, 45);
    fill_random(h_beta, -0.5f, 0.5f, 46);

    double cpu_ms = benchmark_cpu([&]() {
        layernorm_cpu(h_in.data(), h_out_ref.data(), h_gamma.data(), h_beta.data(), rows, cols, eps);
    }, 0, 1);

    float *d_in, *d_out, *d_gamma, *d_beta, *d_mean, *d_var;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;
    float naive_ms = benchmark_gpu([&]() {
        mean_kernel<<<rows, BLOCK>>>(d_in, d_mean, rows, cols);
        var_kernel<<<rows, BLOCK>>>(d_in, d_mean, d_var, rows, cols);
        normalize_kernel<<<rows, BLOCK>>>(d_in, d_out, d_mean, d_var, d_gamma, d_beta, rows, cols, eps);
    }, 3, 15);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool naive_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-3, 1e-3);

    float opt_ms = benchmark_gpu([&]() {
        layernorm_opt_kernel<BLOCK><<<rows, BLOCK>>>(d_in, d_out, d_gamma, d_beta, rows, cols, eps);
    }, 3, 15);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool opt_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-3, 1e-3);

    BenchResult r;
    r.name = "layernorm";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = 2.0 * N * sizeof(float);
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma)); CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_mean)); CUDA_CHECK(cudaFree(d_var));
    return (naive_ok && opt_ok) ? 0 : 1;
}
