// cuCore/kernel/conv2d/conv2d.cu
//
// Stage 4 — DL Operator: 2D Convolution, single channel, square KxK filter,
// 'same' padding, stride 1. (Depthwise conv is the direct multi-channel
// generalization of this kernel -- see docs/roadmap.md.)
//
// Naive:     one thread per output pixel, each thread re-reads its own
//            K x K input neighborhood directly from global memory with NO
//            reuse across threads, even though adjacent output pixels'
//            neighborhoods overlap by (K-1) columns/rows. Filter weights
//            are also re-read from global memory by every thread.
// Optimized: (1) filter weights moved to __constant__ memory, which is
//            broadcast to every thread in a warp from a small cached
//            memory space instead of re-fetched from global DRAM;
//            (2) shared-memory tiling WITH a halo region: each block
//            cooperatively loads its output tile PLUS a (K/2)-pixel border
//            around it into shared memory once, so the overlapping input
//            reads across neighboring output pixels are served from
//            on-chip memory instead of global memory.

#include "../../common/cuda_utils.cuh"

constexpr int MAX_K = 7; // supports up to 7x7 filters in constant memory
__constant__ float c_filter[MAX_K * MAX_K];

void conv2d_cpu(const float* in, const float* filt, float* out, int H, int W, int K) {
    int r = K / 2;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double acc = 0.0;
            for (int ky = -r; ky <= r; ++ky) {
                for (int kx = -r; kx <= r; ++kx) {
                    int iy = y + ky, ix = x + kx;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                        acc += static_cast<double>(in[iy * W + ix]) * filt[(ky + r) * K + (kx + r)];
                }
            }
            out[y * W + x] = static_cast<float>(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Naive
// ---------------------------------------------------------------------------
__global__ void conv2d_naive_kernel(const float* __restrict__ in, const float* __restrict__ filt,
                                     float* __restrict__ out, int H, int W, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int r = K / 2;
    float acc = 0.f;
    for (int ky = -r; ky <= r; ++ky) {
        for (int kx = -r; kx <= r; ++kx) {
            int iy = y + ky, ix = x + kx;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                acc += in[iy * W + ix] * filt[(ky + r) * K + (kx + r)]; // filter re-read from global every time
        }
    }
    out[y * W + x] = acc;
}

// ---------------------------------------------------------------------------
// Optimized: constant-memory filter + shared-memory tile with halo.
// ---------------------------------------------------------------------------
constexpr int TILE = 16;

template <int K>
__global__ void conv2d_opt_kernel(const float* __restrict__ in, float* __restrict__ out, int H, int W) {
    constexpr int R = K / 2;
    constexpr int SMEM_DIM = TILE + 2 * R;
    __shared__ float tile[SMEM_DIM][SMEM_DIM];

    int out_x = blockIdx.x * TILE + threadIdx.x;
    int out_y = blockIdx.y * TILE + threadIdx.y;

    // Cooperatively load the (TILE + 2R) x (TILE + 2R) halo'd tile: every
    // thread may load more than one shared-memory cell so the whole tile
    // (including the border needed by edge threads' filter taps) gets
    // populated exactly once from global memory.
    for (int sy = threadIdx.y; sy < SMEM_DIM; sy += blockDim.y) {
        int gy = blockIdx.y * TILE + sy - R;
        for (int sx = threadIdx.x; sx < SMEM_DIM; sx += blockDim.x) {
            int gx = blockIdx.x * TILE + sx - R;
            tile[sy][sx] = (gy >= 0 && gy < H && gx >= 0 && gx < W) ? in[gy * W + gx] : 0.f;
        }
    }
    __syncthreads();

    if (out_x < W && out_y < H) {
        float acc = 0.f;
#pragma unroll
        for (int ky = 0; ky < K; ++ky)
#pragma unroll
            for (int kx = 0; kx < K; ++kx)
                acc += tile[threadIdx.y + ky][threadIdx.x + kx] * c_filter[ky * K + kx]; // broadcast from constant cache
        out[out_y * W + out_x] = acc;
    }
}

int main(int argc, char** argv) {
    const int H = (argc > 1) ? std::atoi(argv[1]) : 2048;
    const int W = (argc > 2) ? std::atoi(argv[2]) : 2048;
    constexpr int K = 5; // 5x5 filter
    std::printf("Conv2D benchmark, %dx%d image, %dx%d filter\n", H, W, K, K);

    size_t N = (size_t)H * W;
    std::vector<float> h_in(N), h_filt(K * K), h_out_ref(N), h_out_gpu(N);
    fill_random(h_in, -1.f, 1.f, 66);
    fill_random(h_filt, -0.2f, 0.2f, 67);

    double cpu_ms = benchmark_cpu([&]() { conv2d_cpu(h_in.data(), h_filt.data(), h_out_ref.data(), H, W, K); }, 0, 1);

    float *d_in, *d_filt, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filt, K * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filt, h_filt.data(), K * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_filter, h_filt.data(), K * K * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    float naive_ms = benchmark_gpu([&]() { conv2d_naive_kernel<<<grid, block>>>(d_in, d_filt, d_out, H, W, K); });
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool naive_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-3, 1e-3);

    dim3 block_opt(TILE, TILE);
    dim3 grid_opt((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    float opt_ms = benchmark_gpu([&]() { conv2d_opt_kernel<K><<<grid_opt, block_opt>>>(d_in, d_out, H, W); });
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool opt_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-3, 1e-3);

    double flops = 2.0 * N * K * K;
    BenchResult r;
    r.name = "conv2d (5x5)";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.flops = flops;
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_filt));
    CUDA_CHECK(cudaFree(d_out));
    return (naive_ok && opt_ok) ? 0 : 1;
}
