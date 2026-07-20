// cuCore/common/cuda_utils.cuh
//
// Shared utilities for every kernel in cuCore:
//   - CUDA error checking macros
//   - GPU event-based timer
//   - CPU wall-clock timer
//   - correctness comparison helpers (max abs / rel error)
//   - simple RNG-based array initializers
//
// Include this from any .cu / .cpp translation unit in the repo.

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr,                                               \
                "CUDA error at %s:%d code=%d(%s) \"%s\"\n",                    \
                __FILE__, __LINE__, static_cast<int>(err__),                   \
                cudaGetErrorString(err__), #call);                             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_LAST()                                                     \
    do {                                                                        \
        cudaError_t err__ = cudaGetLastError();                                \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr,                                               \
                "CUDA kernel launch error at %s:%d code=%d(%s)\n",             \
                __FILE__, __LINE__, static_cast<int>(err__),                   \
                cudaGetErrorString(err__));                                    \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// GPU timer (CUDA events) -- measures device-side kernel time only.
// ---------------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start_, stop_;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    // returns elapsed milliseconds
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// Runs `fn` `iters` times on the GPU and returns the mean elapsed ms.
// `warmup` iterations are run first and discarded (JIT / clock warm-up,
// cache warm-up). fn() must itself do the kernel launch(es).
template <typename Fn>
float benchmark_gpu(Fn&& fn, int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) fn();
    float total_ms = timer.stop();
    return total_ms / iters;
}

// ---------------------------------------------------------------------------
// CPU timer
// ---------------------------------------------------------------------------
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    // returns elapsed milliseconds
    double stop() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

template <typename Fn>
double benchmark_cpu(Fn&& fn, int warmup = 1, int iters = 5) {
    for (int i = 0; i < warmup; ++i) fn();
    CpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) fn();
    double total_ms = timer.stop();
    return total_ms / iters;
}

// ---------------------------------------------------------------------------
// Correctness helpers
// ---------------------------------------------------------------------------

// Returns max absolute error between two arrays.
inline double max_abs_err(const float* a, const float* b, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return max_err;
}

// Returns true if two arrays match within an absolute+relative tolerance.
inline bool allclose(const float* a, const float* b, size_t n,
                      double atol = 1e-3, double rtol = 1e-3) {
    for (size_t i = 0; i < n; ++i) {
        double diff = std::fabs(a[i] - b[i]);
        double tol = atol + rtol * std::fabs(b[i]);
        if (diff > tol) {
            std::fprintf(stderr,
                "Mismatch at index %zu: got=%.6f expected=%.6f diff=%.6f tol=%.6f\n",
                i, a[i], b[i], diff, tol);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Random init helpers
// ---------------------------------------------------------------------------
inline void fill_random(std::vector<float>& v, float lo = -1.f, float hi = 1.f,
                         unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
struct BenchResult {
    std::string name;
    double cpu_ms   = -1.0;
    double naive_ms = -1.0;
    double opt_ms   = -1.0;
    double bytes_moved = 0.0;   // bytes touched by the optimized kernel (for BW calc)
    double flops       = 0.0;   // FLOPs performed by the optimized kernel (for GFLOP/s calc)
};

inline void print_bench_result(const BenchResult& r) {
    std::printf("\n================ %s ================\n", r.name.c_str());
    if (r.cpu_ms   >= 0) std::printf("  CPU baseline : %10.4f ms\n", r.cpu_ms);
    if (r.naive_ms >= 0) std::printf("  Naive CUDA   : %10.4f ms\n", r.naive_ms);
    if (r.opt_ms   >= 0) std::printf("  Optimized    : %10.4f ms\n", r.opt_ms);
    if (r.naive_ms > 0 && r.opt_ms > 0)
        std::printf("  Speedup (opt vs naive): %.2fx\n", r.naive_ms / r.opt_ms);
    if (r.cpu_ms > 0 && r.opt_ms > 0)
        std::printf("  Speedup (opt vs CPU)  : %.2fx\n", r.cpu_ms / r.opt_ms);
    if (r.bytes_moved > 0 && r.opt_ms > 0) {
        double gbps = (r.bytes_moved / 1e9) / (r.opt_ms / 1e3);
        std::printf("  Effective BW (optimized): %.2f GB/s\n", gbps);
    }
    if (r.flops > 0 && r.opt_ms > 0) {
        double gflops = (r.flops / 1e9) / (r.opt_ms / 1e3);
        std::printf("  Throughput (optimized): %.2f GFLOP/s\n", gflops);
    }
}
