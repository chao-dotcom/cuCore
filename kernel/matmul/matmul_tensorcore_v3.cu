// cuCore/kernel/matmul/matmul_tensorcore_v3.cu
//
// Doubles the block tile to 128x128 (from matmul_tensorcore_v2's 64x64),
// keeping the same 32x32-per-warp fragment tiling and real double-buffered
// pipelining -- this roughly doubles arithmetic intensity (BM*BN/(BM+BN))
// for the same technique. BK is trimmed from 32 to 16 so double-buffered
// shared memory still fits the default 48KB static limit; this doesn't
// reduce total global memory traffic, only how finely it's chunked across
// k-tiles. With a 512-thread block, both tile loaders land on exactly one
// vectorized float4 per thread (no loop passes needed).
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

constexpr int BM = 128, BN = 128, BK = 16;
constexpr int WM = 16, WN = 16, WK = 8;
constexpr int WARP_TILE = 32;
constexpr int WARPS_M = BM / WARP_TILE;   // 4
constexpr int WARPS_N = BN / WARP_TILE;   // 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 16
constexpr int THREADS = NUM_WARPS * 32;   // 512
constexpr int FRAGS = WARP_TILE / WM;     // 2

constexpr int A_ROW_F4 = BK / 4;  // float4-groups per row of the A tile (= 4)
constexpr int B_ROW_F4 = BN / 4;  // float4-groups per row of the B tile (= 32)
// With THREADS=512 these tiles need exactly one float4 per thread each --
// no loop passes required (checked: BM*BK/4 == THREADS == BK*BN/4).

__global__ void matmul_tensorcore_v3_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                             float* __restrict__ C, int N) {
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Loader index mapping: exactly one float4 per thread per tile.
    const int a_row = tid / A_ROW_F4, a_colgrp = tid % A_ROW_F4;
    const int b_row = tid / B_ROW_F4, b_colgrp = tid % B_ROW_F4;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[FRAGS][FRAGS];
#pragma unroll
    for (int m = 0; m < FRAGS; ++m)
#pragma unroll
        for (int n = 0; n < FRAGS; ++n) wmma::fill_fragment(acc[m][n], 0.0f);

    auto load_A = [&](int k0) -> float4 {
        return *reinterpret_cast<const float4*>(&A[(size_t)(block_row + a_row) * N + k0 + a_colgrp * 4]);
    };
    auto store_A = [&](int buf, float4 v) {
        *reinterpret_cast<float4*>(&As[buf][a_row][a_colgrp * 4]) = v;
    };
    auto load_B = [&](int k0) -> float4 {
        return *reinterpret_cast<const float4*>(&B[(size_t)(k0 + b_row) * N + block_col + b_colgrp * 4]);
    };
    auto store_B = [&](int buf, float4 v) {
        *reinterpret_cast<float4*>(&Bs[buf][b_row][b_colgrp * 4]) = v;
    };

    store_A(0, load_A(0));
    store_B(0, load_B(0));
    __syncthreads();

    int cur = 0;
    const int num_k_tiles = N / BK;
    for (int t = 0; t < num_k_tiles; ++t) {
        const int nxt = cur ^ 1;
        const bool has_next = (t + 1 < num_k_tiles);

        float4 a_next, b_next;
        if (has_next) {
            const int k0n = (t + 1) * BK;
            a_next = load_A(k0n);
            b_next = load_B(k0n);
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
            store_A(nxt, a_next);
            store_B(nxt, b_next);
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
    std::printf("matmul_tensorcore_v3 (128x128 tile) benchmark, N = %d\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> h_A(N*N), h_B(N*N), h_C_ref, h_C_gpu(N*N);
    fill_random(h_A, -1.f, 1.f, 61);
    fill_random(h_B, -1.f, 1.f, 62);

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
    float tc3_ms = benchmark_gpu([&]() {
        matmul_tensorcore_v3_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
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
    std::printf("\n================ matmul_tensorcore_v3 (128x128 tile), N=%d ================\n", N);
    std::printf("  Tensor Core v3 (128x128 tile, pipelined): %10.4f ms  (%.2f GFLOP/s)\n", tc3_ms, flops/1e9/(tc3_ms/1e3));
    std::printf("  cuBLAS SGEMM                            : %10.4f ms  (%.2f GFLOP/s)\n", cublas_ms, flops/1e9/(cublas_ms/1e3));
    std::printf("  Fraction of cuBLAS achieved: %.1f%%\n", 100.0 * cublas_ms / tc3_ms);
    std::printf("  Correctness: tc_v3=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
