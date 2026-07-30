// cuCore/kernel/matmul/matmul_tensorcore_v2.cu
//
// Fixes two problems found in the basic WMMA kernel (Section 10c):
//
// 1) Under-sized warp tile: one 16x16 fragment per warp is too small. Tensor
//    cores consume shared-memory data far faster than CUDA cores do, so
//    data-reuse ratio that was enough for v2 (pure CUDA) starves tensor
//    cores instead.
// 2) No real pipelining: single-buffered shared memory loads stall the
//    tensor cores between k-steps.
//
// Fixes (mirroring the pure-CUDA v2 approach):
//   - Bigger warp tiles: each warp computes a 32x32 output tile as a 2x2
//     grid of 16x16 fragments (4 accumulators per warp) instead of just
//     one. This quadruples data reuse per shared-memory load.
//   - Real double buffering: next tile's data is read into registers
//     (via vectorized float4 loads) *before* the compute loop runs, and
//     only written into the next shared-memory buffer *after* compute
//     finishes, so the global-memory latency has the whole compute phase
//     to hide behind.
//
// Requirements: Compute capability 8.0+ (Ampere or newer)

#include "../../common/cuda_utils.cuh"
#include <cublas_v2.h>
#include <mma.h>
using namespace nvcuda;

void matmul_cpu(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k) acc += (double)A[i*N+k] * B[k*N+j];
            C[i*N+j] = (float)acc;
        }
}

constexpr int BM = 64, BN = 64, BK = 32;   // block-level tile
constexpr int WM = 16, WN = 16, WK = 8;    // wmma tf32 fragment shape (fixed)
constexpr int WARP_TILE = 32;              // each warp computes WARP_TILE x WARP_TILE
constexpr int WARPS_M = BM / WARP_TILE;    // 2
constexpr int WARPS_N = BN / WARP_TILE;    // 2
constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 4
constexpr int THREADS = NUM_WARPS * 32;    // 128
constexpr int FRAGS = WARP_TILE / WM;      // 2 fragments per dimension per warp

constexpr int A_ROW_F4 = BK / 4;                 // float4-groups per row of the A tile
constexpr int A_PASSES = (BM * BK / 4) / THREADS; // vectorized loads per thread for A
constexpr int B_ROW_F4 = BN / 4;                 // float4-groups per row of the B tile
constexpr int B_PASSES = (BK * BN / 4) / THREADS; // vectorized loads per thread for B

__global__ void matmul_tensorcore_v2_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                             float* __restrict__ C, int N) {
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[FRAGS][FRAGS];
#pragma unroll
    for (int m = 0; m < FRAGS; ++m)
#pragma unroll
        for (int n = 0; n < FRAGS; ++n) wmma::fill_fragment(acc[m][n], 0.0f);

    auto load_A_reg = [&](int k0, float4 (&dst)[A_PASSES]) {
#pragma unroll
        for (int p = 0; p < A_PASSES; ++p) {
            int idx = tid + p * THREADS;
            int row = idx / A_ROW_F4, colgrp = idx % A_ROW_F4;
            dst[p] = *reinterpret_cast<const float4*>(&A[(size_t)(block_row + row) * N + k0 + colgrp * 4]);
        }
    };
    auto store_A_reg = [&](int buf, const float4 (&src)[A_PASSES]) {
#pragma unroll
        for (int p = 0; p < A_PASSES; ++p) {
            int idx = tid + p * THREADS;
            int row = idx / A_ROW_F4, colgrp = idx % A_ROW_F4;
            *reinterpret_cast<float4*>(&As[buf][row][colgrp * 4]) = src[p];
        }
    };
    auto load_B_reg = [&](int k0, float4 (&dst)[B_PASSES]) {
#pragma unroll
        for (int p = 0; p < B_PASSES; ++p) {
            int idx = tid + p * THREADS;
            int row = idx / B_ROW_F4, colgrp = idx % B_ROW_F4;
            dst[p] = *reinterpret_cast<const float4*>(&B[(size_t)(k0 + row) * N + block_col + colgrp * 4]);
        }
    };
    auto store_B_reg = [&](int buf, const float4 (&src)[B_PASSES]) {
#pragma unroll
        for (int p = 0; p < B_PASSES; ++p) {
            int idx = tid + p * THREADS;
            int row = idx / B_ROW_F4, colgrp = idx % B_ROW_F4;
            *reinterpret_cast<float4*>(&Bs[buf][row][colgrp * 4]) = src[p];
        }
    };

    float4 a_tmp[A_PASSES], b_tmp[B_PASSES];
    load_A_reg(0, a_tmp); store_A_reg(0, a_tmp);
    load_B_reg(0, b_tmp); store_B_reg(0, b_tmp);
    __syncthreads();

    int cur = 0;
    const int num_k_tiles = N / BK;
    for (int t = 0; t < num_k_tiles; ++t) {
        const int nxt = cur ^ 1;
        const bool has_next = (t + 1 < num_k_tiles);

        float4 a_next[A_PASSES], b_next[B_PASSES];
        if (has_next) {
            const int k0n = (t + 1) * BK;
            load_A_reg(k0n, a_next);
            load_B_reg(k0n, b_next);
        }

#pragma unroll
        for (int sub_k = 0; sub_k < BK; sub_k += WK) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::precision::tf32, wmma::row_major> a_frag[FRAGS];
            wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::precision::tf32, wmma::row_major> b_frag[FRAGS];

#pragma unroll
            for (int m = 0; m < FRAGS; ++m) {
                wmma::load_matrix_sync(a_frag[m], &As[cur][warp_row * WARP_TILE + m * WM][sub_k], BK);
#pragma unroll
                for (int e = 0; e < a_frag[m].num_elements; ++e) a_frag[m].x[e] = wmma::__float_to_tf32(a_frag[m].x[e]);
            }
#pragma unroll
            for (int n = 0; n < FRAGS; ++n) {
                wmma::load_matrix_sync(b_frag[n], &Bs[cur][sub_k][warp_col * WARP_TILE + n * WN], BN);
#pragma unroll
                for (int e = 0; e < b_frag[n].num_elements; ++e) b_frag[n].x[e] = wmma::__float_to_tf32(b_frag[n].x[e]);
            }
#pragma unroll
            for (int m = 0; m < FRAGS; ++m)
#pragma unroll
                for (int n = 0; n < FRAGS; ++n)
                    wmma::mma_sync(acc[m][n], a_frag[m], b_frag[n], acc[m][n]);
        }

        if (has_next) {
            store_A_reg(nxt, a_next);
            store_B_reg(nxt, b_next);
        }
        __syncthreads();
        cur = nxt;
    }

#pragma unroll
    for (int m = 0; m < FRAGS; ++m)
#pragma unroll
        for (int n = 0; n < FRAGS; ++n) {
            const int c_row = block_row + warp_row * WARP_TILE + m * WM;
            const int c_col = block_col + warp_col * WARP_TILE + n * WN;
            wmma::store_matrix_sync(&C[(size_t)c_row * N + c_col], acc[m][n], N, wmma::mem_row_major);
        }
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (N % BM != 0 || N % BN != 0 || N % BK != 0) {
        std::printf("This benchmark kernel assumes N is a multiple of %d/%d/%d; got N=%d. Skipping.\n", BM, BN, BK, N);
        return 1;
    }
    const bool check = (N <= 512);
    std::printf("matmul_tensorcore_v2 (tf32, pipelined) benchmark, N = %d\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> h_A(N*N), h_B(N*N), h_C_ref, h_C_gpu(N*N);
    fill_random(h_A, -1.f, 1.f, 51);
    fill_random(h_B, -1.f, 1.f, 52);

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

    dim3 grid(N / BN, N / BM);
    dim3 block(THREADS);
    float tc2_ms = benchmark_gpu([&]() {
        matmul_tensorcore_v2_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }, 2, 10);
    CUDA_CHECK_LAST();

    bool ok = true;
    if (check) {
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        ok = allclose(h_C_gpu.data(), h_C_ref.data(), (size_t)N*N, 2e-1, 5e-2);
    }

    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    float cublas_ms = benchmark_gpu([&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    }, 3, 20);
    cublasDestroy(handle);

    double flops = 2.0 * N * N * N;
    std::printf("\n================ matmul_tensorcore_v2 (tf32, pipelined), N=%d ================\n", N);
    std::printf("  Tensor Core v2 (32x32 warp tile, pipelined): %10.4f ms  (%.2f GFLOP/s)\n", tc2_ms, flops/1e9/(tc2_ms/1e3));
    std::printf("  cuBLAS SGEMM                               : %10.4f ms  (%.2f GFLOP/s)\n", cublas_ms, flops/1e9/(cublas_ms/1e3));
    std::printf("  Fraction of cuBLAS achieved: %.1f%%\n", 100.0 * cublas_ms / tc2_ms);
    std::printf("  Correctness: tc_v2=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
