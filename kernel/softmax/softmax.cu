// cuCore/kernel/softmax/softmax.cu
//
// Stage 4 — DL Operator: row-wise Softmax over a [rows, cols] matrix.
//
// Naive:     three separate kernel launches -- (1) find row max, (2) compute
//            exp(x - max) and row sum, (3) divide by sum -- each one reading
//            the whole [rows, cols] matrix from global memory again. 3x the
//            necessary global memory traffic, plus 2 extra kernel-launch
//            overheads.
// Optimized: ONE kernel, one block per row. Loads each row once into
//            registers via a grid-stride loop over columns, computes max
//            and sum using warp-shuffle reductions (+ a tiny shared-memory
//            step to combine across warps within the block), and writes the
//            row back out exactly once. This is the same "single-pass,
//            block-per-row, warp-reduce" structure used by PyTorch's own
//            softmax and FlashAttention's inner loop.

#include "../../common/cuda_utils.cuh"

void softmax_cpu(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row_in = in + (size_t)r * cols;
        float* row_out = out + (size_t)r * cols;
        float m = -INFINITY;
        for (int c = 0; c < cols; ++c) m = std::max(m, row_in[c]);
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += std::exp(row_in[c] - m);
        for (int c = 0; c < cols; ++c) row_out[c] = static_cast<float>(std::exp(row_in[c] - m) / sum);
    }
}

// ---------------------------------------------------------------------------
// Naive: 3 kernels, each doing a full pass over the matrix.
// ---------------------------------------------------------------------------
__global__ void row_max_kernel(const float* __restrict__ in, float* __restrict__ row_max, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        m = fmaxf(m, in[(size_t)r * cols + c]);
    __shared__ float smem[256];
    smem[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) row_max[r] = smem[0];
}

__global__ void exp_and_sum_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    const float* __restrict__ row_max, float* __restrict__ row_sum,
                                    int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float m = row_max[r];
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float e = __expf(in[(size_t)r * cols + c] - m);
        out[(size_t)r * cols + c] = e;
        s += e;
    }
    __shared__ float smem[256];
    smem[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) row_sum[r] = smem[0];
}

__global__ void divide_kernel(float* __restrict__ out, const float* __restrict__ row_sum, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    float s = row_sum[r];
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        out[(size_t)r * cols + c] /= s;
}

// ---------------------------------------------------------------------------
// Optimized: single-pass-per-row-in-registers, block-per-row, warp-shuffle
// reductions for both max and sum.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int BLOCK_SIZE>
__global__ void softmax_opt_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float warp_reduce_buf[BLOCK_SIZE / 32];
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row_in = in + (size_t)r * cols;
    float* row_out = out + (size_t)r * cols;

    int lane = threadIdx.x % 32, warp_id = threadIdx.x / 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    // Pass 1: row max, single global read of the row.
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) m = fmaxf(m, row_in[c]);
    m = warp_reduce_max(m);
    if (lane == 0) warp_reduce_buf[warp_id] = m;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < NUM_WARPS) ? warp_reduce_buf[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0) warp_reduce_buf[0] = v;
    }
    __syncthreads();
    m = warp_reduce_buf[0];
    __syncthreads(); // guard buf reuse below

    // Pass 2: exp(x - m) written directly to output, accumulate sum.
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) {
        float e = __expf(row_in[c] - m);
        row_out[c] = e;
        s += e;
    }
    s = warp_reduce_sum(s);
    if (lane == 0) warp_reduce_buf[warp_id] = s;
    __syncthreads();
    if (warp_id == 0) {
        float v = (lane < NUM_WARPS) ? warp_reduce_buf[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) warp_reduce_buf[0] = v;
    }
    __syncthreads();
    s = warp_reduce_buf[0];

    // Pass 3: normalize (this final pass over `out` is unavoidable in any
    // 2-pass-stat softmax; FlashAttention avoids even this via online
    // rescaling -- see docs/roadmap.md for the FlashAttention-lite plan).
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) row_out[c] /= s;
}

int main(int argc, char** argv) {
    const int rows = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int cols = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const size_t N = (size_t)rows * cols;
    std::printf("Softmax benchmark, [%d, %d] matrix\n", rows, cols);

    std::vector<float> h_in(N), h_out_ref(N), h_out_gpu(N);
    fill_random(h_in, -5.f, 5.f, 33);
    double cpu_ms = benchmark_cpu([&]() { softmax_cpu(h_in.data(), h_out_ref.data(), rows, cols); }, 0, 1);

    float *d_in, *d_out, *d_row_max, *d_row_sum;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_max, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_sum, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;
    float naive_ms = benchmark_gpu([&]() {
        row_max_kernel<<<rows, BLOCK>>>(d_in, d_row_max, rows, cols);
        exp_and_sum_kernel<<<rows, BLOCK>>>(d_in, d_out, d_row_max, d_row_sum, rows, cols);
        divide_kernel<<<rows, BLOCK>>>(d_out, d_row_sum, rows, cols);
    }, 3, 15);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool naive_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-4, 1e-3);

    float opt_ms = benchmark_gpu([&]() {
        softmax_opt_kernel<BLOCK><<<rows, BLOCK>>>(d_in, d_out, rows, cols);
    }, 3, 15);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool opt_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-4, 1e-3);

    BenchResult r;
    r.name = "softmax";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = 2.0 * N * sizeof(float); // 1 read + 1 write in the fused kernel
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_row_max));
    CUDA_CHECK(cudaFree(d_row_sum));
    return (naive_ok && opt_ok) ? 0 : 1;
}
