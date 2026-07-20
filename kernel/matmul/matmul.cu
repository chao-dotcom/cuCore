// cuCore/kernel/matmul/matmul.cu
//
// Stage 3 — Matrix Computation: GEMM  (C = A x B), all matrices row-major,
// square N x N for simplicity.
//
// Implements three progressively optimized kernels plus a cuBLAS reference:
//   1) naive        - one thread per output element, all operands read
//                      straight from global memory every iteration
//                      (arithmetic intensity ~ O(1), heavily memory-bound,
//                      re-reads each element of A and B O(N) times from
//                      global memory).
//   2) smem tiled   - classic shared-memory tiled GEMM: cooperatively load
//                      TILE x TILE blocks of A and B into shared memory
//                      once per tile-step and reuse them for every output
//                      element the block computes, cutting global memory
//                      traffic by ~TILE.
//   3) register     - on top of shared-memory tiling, each thread computes
//                      a small TM x TN micro-tile of the output held
//                      entirely in registers, amortizing every shared-memory
//                      load across TM*TN FMAs instead of 1 (2D register
//                      tiling / micro-kernel), which is the same structural
//                      idea CUTLASS and cuBLAS use internally (just without
//                      their double buffering / Tensor Core MMA path).
//   ref) cuBLAS     - SGEMM via cublasSgemm, our ceiling for comparison.
//
// WMMA / Tensor Core and full CUTLASS-style double buffering are the natural
// "Stage 3.5" extension once this is validated on real Volta+/Ampere+
// hardware -- see docs/roadmap.md for the concrete next steps and why we
// scaffold rather than guess at code we can't validate in this environment.

#include "../../common/cuda_utils.cuh"
#include <cublas_v2.h>

void matmul_cpu(const float* A, const float* B, float* C, int N) {
    // Simple triple loop; fine as a correctness oracle for modest N (used
    // only to validate GEMM at a small size -- see main()).
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < N; ++k) acc += static_cast<double>(A[i * N + k]) * B[k * N + j];
            C[i * N + j] = static_cast<float>(acc);
        }
    }
}

// ---------------------------------------------------------------------------
// 1) Naive GEMM
// ---------------------------------------------------------------------------
__global__ void matmul_naive_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col]; // A: coalesced-ish per row, B: strided global reads
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// 2) Shared-memory tiled GEMM
// ---------------------------------------------------------------------------
constexpr int TILE = 32;

__global__ void matmul_smem_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;
    int num_tiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // reused TILE times from on-chip memory
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 3) Register-tiled GEMM: each thread computes a TM x TN micro-tile of C.
//    Block computes a BM x BN tile of C using BK-deep shared-memory panels.
// ---------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, BK = 16; // block-level output tile & shared K-depth
constexpr int TM = 4, TN = 4;             // per-thread micro-tile (registers)
// threads per block = (BM/TM) * (BN/TN) = 16 * 16 = 256

__global__ void matmul_regtile_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                       float* __restrict__ C, int N) {
    __shared__ float As[BK][BM];  // transposed layout: coalesced writes during load, fast column reads during compute
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int tx = threadIdx.x % (BN / TN); // 0..15
    const int ty = threadIdx.x / (BN / TN); // 0..15
    const int tid = threadIdx.x;            // 0..255

    float acc[TM][TN] = {0.f};

    int num_k_tiles = (N + BK - 1) / BK;
    for (int t = 0; t < num_k_tiles; ++t) {
        // Cooperative load of A tile (BM x BK) into As, transposed to [BK][BM]
        // so downstream reads down a k-column are contiguous across threads.
#pragma unroll
        for (int i = tid; i < BM * BK; i += blockDim.x) {
            int a_row = i / BK, a_col = i % BK;
            int g_row = block_row + a_row, g_col = t * BK + a_col;
            As[a_col][a_row] = (g_row < N && g_col < N) ? A[g_row * N + g_col] : 0.f;
        }
        // Cooperative load of B tile (BK x BN) into Bs, kept row-major.
#pragma unroll
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int b_row = i / BN, b_col = i % BN;
            int g_row = t * BK + b_row, g_col = block_col + b_col;
            Bs[b_row][b_col] = (g_row < N && g_col < N) ? B[g_row * N + g_col] : 0.f;
        }
        __syncthreads();

        // Each thread's TM x TN accumulator update, reusing each shared-mem
        // value TM (or TN) times across the micro-tile -- this is the whole
        // point of register tiling: amortize shared-memory bandwidth over
        // more FMAs per load.
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM], b_reg[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = block_row + ty * TM + i;
        if (g_row >= N) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_col = block_col + tx * TN + j;
            if (g_col < N) C[g_row * N + g_col] = acc[i][j];
        }
    }
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const bool check = (N <= 512); // CPU triple-loop reference only for modest sizes
    std::printf("GEMM benchmark, N = %d (%s)\n", N, check ? "with CPU correctness check" : "perf-only, N too large for CPU ref");

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    std::vector<float> h_A(N * N), h_B(N * N), h_C_ref, h_C_gpu(N * N);
    fill_random(h_A, -1.f, 1.f, 21);
    fill_random(h_B, -1.f, 1.f, 22);

    double cpu_ms = -1.0;
    if (check) {
        h_C_ref.resize(N * N);
        cpu_ms = benchmark_cpu([&]() { matmul_cpu(h_A.data(), h_B.data(), h_C_ref.data(), N); }, 0, 1);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // ---- naive ----
    dim3 block2d(32, 32);
    dim3 grid2d((N + 31) / 32, (N + 31) / 32);
    float naive_ms = benchmark_gpu([&]() {
        matmul_naive_kernel<<<grid2d, block2d>>>(d_A, d_B, d_C, N);
    }, 2, 10);
    CUDA_CHECK_LAST();
    bool naive_ok = true;
    if (check) {
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        naive_ok = allclose(h_C_gpu.data(), h_C_ref.data(), (size_t)N * N, 1e-1, 1e-2);
    }

    // ---- shared-memory tiled ----
    dim3 grid_smem((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    float smem_ms = benchmark_gpu([&]() {
        matmul_smem_kernel<<<grid_smem, block2d>>>(d_A, d_B, d_C, N);
    }, 2, 10);
    CUDA_CHECK_LAST();
    bool smem_ok = true;
    if (check) {
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        smem_ok = allclose(h_C_gpu.data(), h_C_ref.data(), (size_t)N * N, 1e-1, 1e-2);
    }

    // ---- register tiled ----
    dim3 grid_reg((N + BN - 1) / BN, (N + BM - 1) / BM);
    dim3 block_reg((BM / TM) * (BN / TN)); // 256 threads
    float reg_ms = benchmark_gpu([&]() {
        matmul_regtile_kernel<<<grid_reg, block_reg>>>(d_A, d_B, d_C, N);
    }, 2, 10);
    CUDA_CHECK_LAST();
    bool reg_ok = true;
    if (check) {
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        reg_ok = allclose(h_C_gpu.data(), h_C_ref.data(), (size_t)N * N, 1e-1, 1e-2);
    }

    // ---- cuBLAS reference ----
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    float cublas_ms = benchmark_gpu([&]() {
        // cuBLAS is column-major; for row-major A,B,C this computes C^T = B^T * A^T,
        // which is equivalent to C = A * B when all matrices are square and we
        // simply swap the A/B operands -- a standard trick to reuse SGEMM as-is.
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_B, N, d_A, N, &beta, d_C, N);
    }, 3, 20);
    cublasDestroy(handle);

    double flops = 2.0 * N * N * N; // N^3 multiply-adds = 2*N^3 FLOPs

    std::printf("\n================ matmul (GEMM), N=%d ================\n", N);
    if (cpu_ms >= 0) std::printf("  CPU baseline     : %10.4f ms\n", cpu_ms);
    std::printf("  Naive CUDA       : %10.4f ms  (%.2f GFLOP/s)\n", naive_ms, flops / 1e9 / (naive_ms / 1e3));
    std::printf("  Shared-mem tiled : %10.4f ms  (%.2f GFLOP/s)\n", smem_ms, flops / 1e9 / (smem_ms / 1e3));
    std::printf("  Register tiled   : %10.4f ms  (%.2f GFLOP/s)\n", reg_ms, flops / 1e9 / (reg_ms / 1e3));
    std::printf("  cuBLAS SGEMM     : %10.4f ms  (%.2f GFLOP/s)\n", cublas_ms, flops / 1e9 / (cublas_ms / 1e3));
    std::printf("  Speedup naive->smem   : %.2fx\n", naive_ms / smem_ms);
    std::printf("  Speedup smem->regtile : %.2fx\n", smem_ms / reg_ms);
    std::printf("  Fraction of cuBLAS achieved by register-tiled kernel: %.1f%%\n", 100.0 * cublas_ms / reg_ms);
    std::printf("  Correctness: naive=%s  smem=%s  regtile=%s\n",
                naive_ok ? "PASS" : "FAIL", smem_ok ? "PASS" : "FAIL", reg_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return (naive_ok && smem_ok && reg_ok) ? 0 : 1;
}
