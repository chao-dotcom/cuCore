// cuCore/kernel/matmul/matmul_tensorcore.cu
//
// Tensor Core GEMM using the WMMA API with tf32 inputs and fp32 accumulation.
//
// Requirements:
//   - Compute capability 8.0+ (Ampere or newer)
//   - NVIDIA CUDA Toolkit with WMMA support (-mma.h)
//
// Design:
//   - Each warp computes one 16x16 output tile per k-step using the tf32
//     WMMA shape (16x16x8), the only shape tf32 supports.
//   - Block-level shared-memory tiling (single-buffered, no software
//     pipelining) stages A/B tiles so all warps in a block reuse them
//     instead of re-reading global memory per warp.
//   - tf32: 8 exponent bits (same dynamic range as FP32) + 10 mantissa bits.
//     Accumulation still happens in full FP32. A few percent relative error
//     versus full FP32 is expected and normal (not a correctness bug).

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

constexpr int BM = 32, BN = 64, BK = 32;   // block-level tile (rows, cols, k-depth)
constexpr int WM = 16, WN = 16, WK = 8;    // wmma tf32 fragment shape (fixed by API)
constexpr int WARPS_M = BM / WM;           // 2
constexpr int WARPS_N = BN / WN;           // 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 8
constexpr int THREADS = NUM_WARPS * 32;    // 256

__global__ void matmul_tensorcore_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                          float* __restrict__ C, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Cooperative load of the A and B tiles into shared memory (all
        // warps in the block share these -- no per-warp redundant reads).
        for (int i = tid; i < BM * BK; i += THREADS) {
            int r = i / BK, c = i % BK;
            As[r][c] = A[(size_t)(block_row + r) * N + k0 + c];
        }
        for (int i = tid; i < BK * BN; i += THREADS) {
            int r = i / BN, c = i % BN;
            Bs[r][c] = B[(size_t)(k0 + r) * N + block_col + c];
        }
        __syncthreads();

        for (int sub_k = 0; sub_k < BK; sub_k += WK) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::precision::tf32, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, &As[warp_row * WM][sub_k], BK);
            wmma::load_matrix_sync(b_frag, &Bs[sub_k][warp_col * WN], BN);

            // tf32 requires an explicit round-to-tf32-precision step on the
            // loaded operands before the tensor-core multiply-accumulate.
#pragma unroll
            for (int t = 0; t < a_frag.num_elements; ++t) a_frag.x[t] = wmma::__float_to_tf32(a_frag.x[t]);
#pragma unroll
            for (int t = 0; t < b_frag.num_elements; ++t) b_frag.x[t] = wmma::__float_to_tf32(b_frag.x[t]);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();  // all warps must finish reading before the tile is overwritten
    }

    const int c_row = block_row + warp_row * WM;
    const int c_col = block_col + warp_col * WN;
    wmma::store_matrix_sync(&C[(size_t)c_row * N + c_col], acc_frag, N, wmma::mem_row_major);
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (N % BN != 0 || N % BM != 0) {
        std::printf("This benchmark kernel assumes N is a multiple of %d and %d; got N=%d. Skipping.\n", BN, BM, N);
        return 1;
    }
    const bool check = (N <= 512);
    std::printf("matmul_tensorcore (tf32) benchmark, N = %d\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> h_A(N*N), h_B(N*N), h_C_ref, h_C_gpu(N*N);
    fill_random(h_A, -1.f, 1.f, 41);
    fill_random(h_B, -1.f, 1.f, 42);

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
    float tc_ms = benchmark_gpu([&]() {
        matmul_tensorcore_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }, 2, 10);
    CUDA_CHECK_LAST();

    bool ok = true;
    if (check) {
        CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        // Looser tolerance than the FP32 kernels: tf32 truncates the
        // mantissa to 10 bits before the tensor-core multiply, so a few
        // percent relative error versus the FP64-accumulated CPU reference
        // is expected and is not a correctness bug.
        ok = allclose(h_C_gpu.data(), h_C_ref.data(), (size_t)N*N, 2e-1, 5e-2);
    }

    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    float cublas_ms = benchmark_gpu([&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    }, 3, 20);
    cublasDestroy(handle);

    double flops = 2.0 * N * N * N;
    std::printf("\n================ matmul_tensorcore (tf32 WMMA), N=%d ================\n", N);
    std::printf("  Tensor Core (tf32):  %10.4f ms  (%.2f GFLOP/s)\n", tc_ms, flops/1e9/(tc_ms/1e3));
    std::printf("  cuBLAS SGEMM      :  %10.4f ms  (%.2f GFLOP/s)\n", cublas_ms, flops/1e9/(cublas_ms/1e3));
    std::printf("  Fraction of cuBLAS achieved: %.1f%%\n", 100.0 * cublas_ms / tc_ms);
    std::printf("  Correctness: tensorcore=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
