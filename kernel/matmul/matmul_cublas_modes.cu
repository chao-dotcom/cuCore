// cuCore/kernel/matmul/matmul_cublas_modes.cu
//
// Diagnostic: calls cuBLAS SGEMM three explicit ways to determine whether the
// "cuBLAS" reference used throughout the other matmul benchmarks has
// actually been tensor-core-accelerated, or just a well-tuned plain-FP32
// kernel. This matters because every "% of cuBLAS" number reported by those
// benchmarks is relative to whichever one the default call happens to be.

#include "../../common/cuda_utils.cuh"
#include <cublas_v2.h>

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    std::printf("cuBLAS math-mode comparison, N = %d\n", N);

    size_t bytes = (size_t)N * N * sizeof(float);
    std::vector<float> h_A(N*N), h_B(N*N);
    fill_random(h_A, -1.f, 1.f, 71);
    fill_random(h_B, -1.f, 1.f, 72);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    const double flops = 2.0 * N * N * N;

    // 1. Default -- no math-mode hint given, exactly what every other matmul
    //    benchmark in this repo compares against.
    float default_ms = benchmark_gpu([&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    }, 3, 20);

    // 2. Pedantic -- explicitly forbid tensor-core shortcuts, force plain FP32.
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    float pedantic_ms = benchmark_gpu([&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    }, 3, 20);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);  // reset

    // 3. Explicit TF32 -- force tensor-core acceleration via cublasGemmEx.
    float tf32_ms = benchmark_gpu([&]() {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                     &alpha, d_B, CUDA_R_32F, N, d_A, CUDA_R_32F, N,
                     &beta, d_C, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }, 3, 20);

    cublasDestroy(handle);

    std::printf("\n================ cuBLAS math modes, N=%d ================\n", N);
    std::printf("  Default       : %10.4f ms  (%.2f GFLOP/s)\n", default_ms, flops/1e9/(default_ms/1e3));
    std::printf("  Pedantic FP32 : %10.4f ms  (%.2f GFLOP/s)\n", pedantic_ms, flops/1e9/(pedantic_ms/1e3));
    std::printf("  Explicit TF32 : %10.4f ms  (%.2f GFLOP/s)\n", tf32_ms, flops/1e9/(tf32_ms/1e3));
    std::printf("  Default is %.1f%% of the way from Pedantic to TF32\n",
                100.0 * (flops/1e9/(default_ms/1e3) - flops/1e9/(pedantic_ms/1e3)) /
                        ((flops/1e9/(tf32_ms/1e3)) - (flops/1e9/(pedantic_ms/1e3)) + 1e-9));
    return 0;
}
