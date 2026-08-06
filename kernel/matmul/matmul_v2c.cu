// cuCore/kernel/matmul/matmul_v2c.cu
//
// v2b (64x64 tile, float4 vectorized loads, transposed A tile, real
// pipelining) plus split-K: when the (M/BM)*(N/BN) block grid is too small to
// keep the GPU's SMs busy for enough waves (small/medium N), the K dimension
// is also split across gridDim.z extra blocks, each accumulating a partial
// sum into C via atomicAdd. This adds parallelism without touching the
// per-block register tile (unlike shrinking BM/BN further, which would cut
// arithmetic intensity again). split_k is chosen at runtime from the block
// count alone -- it is a launch-parameter heuristic on the *same* kernel,
// not a different kernel for different sizes.
//
// Note: Assumes N is a multiple of 64 and, for the split_k values this
// heuristic can choose (1/2/4/8), that N/split_k is a multiple of BK.

#include "../../common/cuda_utils.cuh"
#include <cublas_v2.h>

void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k) acc += (double)A[i*N+k] * B[k*N+j];
            C[i*N+j] = (float)acc;
        }
}

constexpr int BM = 64, BN = 64, BK = 16;
constexpr int TM = 4, TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256

__global__ void matmul_v2c_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C, int N, int split_k) {
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int k_per_slice = N / split_k;
    const int k_base = blockIdx.z * k_per_slice;

    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN);
    const int ty = tid / (BN / TN);

    const int a_row = tid / 4, a_col_grp = tid % 4;
    const int b_row = tid / 16, b_col_grp = tid % 16;

    auto load_store_tile = [&](int k0, int buf) {
        float4 a4 = *reinterpret_cast<const float4*>(&A[(size_t)(block_row + a_row) * N + k0 + a_col_grp * 4]);
        As[buf][a_col_grp * 4 + 0][a_row] = a4.x;
        As[buf][a_col_grp * 4 + 1][a_row] = a4.y;
        As[buf][a_col_grp * 4 + 2][a_row] = a4.z;
        As[buf][a_col_grp * 4 + 3][a_row] = a4.w;

        float4 b4 = *reinterpret_cast<const float4*>(&B[(size_t)(k0 + b_row) * N + block_col + b_col_grp * 4]);
        *reinterpret_cast<float4*>(&Bs[buf][b_row][b_col_grp * 4]) = b4;
    };

    float acc[TM][TN] = {0.f};
    const int num_k_tiles = k_per_slice / BK;

    int cur = 0;
    load_store_tile(k_base, cur);
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        const int nxt = cur ^ 1;
        const bool has_next = (t + 1 < num_k_tiles);

        float4 a4_next, b4_next;
        if (has_next) {
            const int k0_next = k_base + (t + 1) * BK;
            a4_next = *reinterpret_cast<const float4*>(&A[(size_t)(block_row + a_row) * N + k0_next + a_col_grp * 4]);
            b4_next = *reinterpret_cast<const float4*>(&B[(size_t)(k0_next + b_row) * N + block_col + b_col_grp * 4]);
        }

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM], b_reg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[cur][k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[cur][k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        if (has_next) {
            As[nxt][a_col_grp * 4 + 0][a_row] = a4_next.x;
            As[nxt][a_col_grp * 4 + 1][a_row] = a4_next.y;
            As[nxt][a_col_grp * 4 + 2][a_row] = a4_next.z;
            As[nxt][a_col_grp * 4 + 3][a_row] = a4_next.w;
            *reinterpret_cast<float4*>(&Bs[nxt][b_row][b_col_grp * 4]) = b4_next;
        }
        __syncthreads();
        cur = nxt;
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int g_row = block_row + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int g_col = block_col + tx * TN + j;
            if (split_k == 1) {
                C[(size_t)g_row * N + g_col] = acc[i][j];
            } else {
                atomicAdd(&C[(size_t)g_row * N + g_col], acc[i][j]);
            }
        }
    }
}

// Chooses split_k from block count alone: a launch-parameter heuristic on
// the same kernel, aiming for enough blocks to cover ~4 waves across an
// SM count passed in (so it degrades gracefully on non-A100 GPUs too).
int choose_split_k(int N, int sm_count) {
    const int base_blocks = (N / BM) * (N / BN);
    const int target_blocks = 4 * sm_count;
    int split_k = 1;
    while (split_k < 8 && base_blocks * split_k < target_blocks) split_k *= 2;
    return split_k;
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (N % BM != 0) {
        std::printf("This benchmark kernel assumes N is a multiple of %d; got N=%d. Skipping.\n", BM, N);
        return 1;
    }
    const bool check = (N <= 512);
    std::printf("matmul_v2c benchmark, N = %d\n", N);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const int split_k = choose_split_k(N, prop.multiProcessorCount);

    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> h_A(N*N), h_B(N*N), h_C_ref, h_C_gpu(N*N);
    fill_random(h_A, -1.f, 1.f, 31);
    fill_random(h_B, -1.f, 1.f, 32);

    if (check) {
        h_C_ref.resize(N*N);
        benchmark_cpu([&]() { matmul_cpu(h_A.data(), h_B.data(), h_C_ref.data(), N); }, 0, 1);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid(N / BN, N / BM, split_k);
    dim3 block(THREADS);
    float v2c_ms = benchmark_gpu([&]() {
        if (split_k > 1) CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        matmul_v2c_kernel<<<grid, block>>>(d_A, d_B, d_C, N, split_k);
    }, 2, 10);
    CUDA_CHECK_LAST();

    bool ok = true;
    if (check) {
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        ok = allclose(h_C_gpu.data(), h_C_ref.data(), (size_t)N*N, 1e-1, 1e-2);
    }

    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    float cublas_ms = benchmark_gpu([&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    }, 3, 20);
    cublasDestroy(handle);

    double flops = 2.0 * N * N * N;
    std::printf("\n================ matmul_v2c (GEMM, 64x64 tile, split-K), N=%d ================\n", N);
    std::printf("  split_k chosen: %d\n", split_k);
    std::printf("  V2c (64x64 tile, split-K=%d, pipelined): %10.4f ms  (%.2f GFLOP/s)\n", split_k, v2c_ms, flops/1e9/(v2c_ms/1e3));
    std::printf("  cuBLAS SGEMM                       : %10.4f ms  (%.2f GFLOP/s)\n", cublas_ms, flops/1e9/(cublas_ms/1e3));
    std::printf("  Fraction of cuBLAS achieved: %.1f%%\n", 100.0 * cublas_ms / v2c_ms);
    std::printf("  Correctness: v2c=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
