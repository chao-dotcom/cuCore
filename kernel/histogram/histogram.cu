// cuCore/kernel/histogram/histogram.cu
//
// Stage 1 — Histogram: bin `n` random byte values into NUM_BINS counters.
//
// Naive:     every thread does atomicAdd directly into GLOBAL memory. With
//            NUM_BINS small relative to the number of threads, many threads
//            collide on the same bin at the same time -> heavy atomic
//            contention on DRAM, serialized by the memory controller.
// Optimized: "privatization" -- each block keeps its own histogram in
//            SHARED memory (atomics there are far cheaper, on-chip, and
//            partitioned across banks), then each block adds its private
//            histogram into the global one at the end (NUM_BINS atomics
//            per block instead of N atomics total).

#include "../../common/cuda_utils.cuh"

constexpr int NUM_BINS = 256;

void histogram_cpu(const unsigned char* data, size_t n, unsigned int* hist) {
    std::fill(hist, hist + NUM_BINS, 0u);
    for (size_t i = 0; i < n; ++i) hist[data[i]]++;
}

// ---------------------------------------------------------------------------
// Naive: direct global-memory atomics.
// ---------------------------------------------------------------------------
__global__ void histogram_naive_kernel(const unsigned char* __restrict__ data, size_t n,
                                        unsigned int* __restrict__ hist) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[data[i]], 1u);
}

// ---------------------------------------------------------------------------
// Optimized: shared-memory privatized histogram per block, grid-stride load.
// ---------------------------------------------------------------------------
__global__ void histogram_opt_kernel(const unsigned char* __restrict__ data, size_t n,
                                      unsigned int* __restrict__ hist) {
    __shared__ unsigned int local_hist[NUM_BINS];

    // Cooperatively zero the shared histogram.
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) local_hist[b] = 0u;
    __syncthreads();

    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&local_hist[data[i]], 1u); // on-chip atomic: far cheaper than DRAM atomic
    __syncthreads();

    // Merge this block's private histogram into the global one:
    // NUM_BINS atomics per BLOCK, versus N atomics total in the naive version.
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        if (local_hist[b] != 0u) atomicAdd(&hist[b], local_hist[b]);
}

int main(int argc, char** argv) {
    const size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 26); // 64M bytes
    std::printf("Histogram benchmark, N = %zu bytes, %d bins\n", N, NUM_BINS);

    std::vector<unsigned char> h_data(N);
    {
        std::mt19937 rng(99);
        std::uniform_int_distribution<int> dist(0, NUM_BINS - 1);
        for (auto& v : h_data) v = static_cast<unsigned char>(dist(rng));
    }

    std::vector<unsigned int> h_hist_ref(NUM_BINS);
    double cpu_ms = benchmark_cpu([&]() { histogram_cpu(h_data.data(), N, h_hist_ref.data()); }, 1, 3);

    unsigned char* d_data;
    unsigned int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N));
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = static_cast<int>((N + block - 1) / block);
    std::vector<unsigned int> h_hist_naive(NUM_BINS);
    float naive_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_naive_kernel<<<grid, block>>>(d_data, N, d_hist);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_hist_naive.data(), d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    int sm_count;
    { int dev; CUDA_CHECK(cudaGetDevice(&dev));
      cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
      sm_count = prop.multiProcessorCount; }
    int opt_grid = sm_count * 8;
    std::vector<unsigned int> h_hist_opt(NUM_BINS);
    float opt_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
        histogram_opt_kernel<<<opt_grid, block>>>(d_data, N, d_hist);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_hist_opt.data(), d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    bool naive_ok = (h_hist_naive == h_hist_ref);
    bool opt_ok = (h_hist_opt == h_hist_ref);

    BenchResult r;
    r.name = "histogram";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = static_cast<double>(N); // read-only pass over bytes
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    return (naive_ok && opt_ok) ? 0 : 1;
}
