// cuCore/kernel/matmul/matmul_v2b.cu
//
// Same techniques as matmul_v2 (float4 vectorized loads, transposed A tile,
// genuine software-pipelined double buffering) but with a 64x64 block tile
// instead of 128x128. Goal: fix occupancy at N=1024/2048, where the 128x128
// tile leaves most of an A100's 108 SMs idle (only 64 blocks launch at
// N=1024). BK=16 (instead of 8) so float4 loads still divide evenly onto
// 256 threads at the smaller tile size.
//
// Note: Assumes N is a multiple of 64 (true for all sizes in the benchmark).

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

__global__ void matmul_v2b_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C, int N) {
    __shared__ float As[2][BK][BM];   // transposed: As[buf][k][m]
    __shared__ float Bs[2][BK][BN];   // natural:    Bs[buf][k][n]

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN);
    const int ty = tid / (BN / TN);

    // A tile: 64 rows x 16 cols -> 256 float4s (4 float4/row). One float4 per thread.
    const int a_row = tid / 4, a_col_grp = tid % 4;
    // B tile: 16 rows x 64 cols -> 256 float4s (16 float4/row). One float4 per thread.
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
    const int num_k_tiles = N / BK;

    int cur = 0;
    load_store_tile(0, cur);
    __syncthreads();

    for (int t = 0; t < num_k_tiles; ++t) {
        const int nxt = cur ^ 1;
        const bool has_next = (t + 1 < num_k_tiles);

        float4 a4_next, b4_next;
        int k0_next = 0;
        if (has_next) {
            k0_next = (t + 1) * BK;
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
            C[(size_t)g_row * N + g_col] = acc[i][j];
        }
    }
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (N % BM != 0) {
        std::printf("This benchmark kernel assumes N is a multiple of %d; got N=%d. Skipping.\n", BM, N);
        return 1;
    }
    const bool check = (N <= 512);
    std::printf("matmul_v2b benchmark, N = %d\n", N);

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

    dim3 grid(N / BN, N / BM);
    dim3 block(THREADS);
    float v2b_ms = benchmark_gpu([&]() {
        matmul_v2b_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
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
    std::printf("\n================ matmul_v2b (GEMM, 64x64 tile), N=%d ================\n", N);
    std::printf("  V2b (64x64 tile, BK=16, pipelined): %10.4f ms  (%.2f GFLOP/s)\n", v2b_ms, flops/1e9/(v2b_ms/1e3));
    std::printf("  cuBLAS SGEMM                       : %10.4f ms  (%.2f GFLOP/s)\n", cublas_ms, flops/1e9/(cublas_ms/1e3));
    std::printf("  Fraction of cuBLAS achieved: %.1f%%\n", 100.0 * cublas_ms / v2b_ms);
    std::printf("  Correctness: v2b=%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
