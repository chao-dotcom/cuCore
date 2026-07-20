// cuCore/kernel/streams_demo/streams_demo.cu
//
// Stage 5 — Runtime Optimization: CUDA Streams, Pinned Memory, Async Memcpy,
// Compute/Copy Overlap, and CUDA Graphs.
//
// Unlike Stages 1-4, there's no "naive vs optimized kernel" here -- the
// kernel itself (a simple elementwise op) is intentionally trivial. What's
// being benchmarked is entirely about the HOST-side orchestration of work:
//
//   Demo 1: pageable vs pinned host memory -- pinned (page-locked) memory
//           lets the DMA engine copy directly without an extra staging
//           copy through a pinned bounce buffer, so H2D/D2H bandwidth is
//           measurably higher.
//   Demo 2: single stream (serialized copy-then-compute-then-copy-back per
//           chunk) vs multiple streams (chunk N's D2H copy overlaps with
//           chunk N+1's kernel, which overlaps with chunk N+2's H2D copy)
//           -- classic "software pipelining" across the copy and compute
//           engines, which is exactly what real training/inference
//           pipelines (data loading, preprocessing, model exec) rely on.
//   Demo 3: CUDA Graph replay -- capturing the same sequence of operations
//           once and replaying it via cudaGraphLaunch, amortizing CPU-side
//           kernel-launch overhead across many iterations (important once
//           kernels get small/numerous, e.g. many tiny ops in a transformer
//           layer).

#include "../../common/cuda_utils.cuh"

__global__ void square_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * in[i];
}

// ---------------------------------------------------------------------------
// Demo 1: pinned vs pageable host memory copy bandwidth
// ---------------------------------------------------------------------------
void demo_pinned_vs_pageable(size_t n) {
    size_t bytes = n * sizeof(float);
    std::printf("\n--- Demo 1: Pageable vs Pinned host memory (H2D copy of %.2f MB) ---\n", bytes / 1e6);

    // Pageable
    std::vector<float> h_pageable(n, 1.0f);
    float* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));
    float pageable_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaMemcpy(d_buf, h_pageable.data(), bytes, cudaMemcpyHostToDevice));
    }, 3, 20);
    double pageable_gbps = (bytes / 1e9) / (pageable_ms / 1e3);

    // Pinned
    float* h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes)); // page-locked allocation
    std::fill(h_pinned, h_pinned + n, 1.0f);
    float pinned_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
    }, 3, 20);
    double pinned_gbps = (bytes / 1e9) / (pinned_ms / 1e3);

    std::printf("  Pageable H2D: %8.4f ms  (%.2f GB/s)\n", pageable_ms, pageable_gbps);
    std::printf("  Pinned   H2D: %8.4f ms  (%.2f GB/s)\n", pinned_ms, pinned_gbps);
    std::printf("  Speedup from pinning: %.2fx\n", pageable_ms / pinned_ms);

    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_buf));
}

// ---------------------------------------------------------------------------
// Demo 2: single-stream (serialized) vs multi-stream (overlapped) pipeline
// ---------------------------------------------------------------------------
void demo_stream_overlap(size_t total_n, int num_chunks) {
    size_t chunk_n = total_n / num_chunks;
    size_t chunk_bytes = chunk_n * sizeof(float);
    std::printf("\n--- Demo 2: Single stream vs %d-way stream overlap (total %.2f MB) ---\n",
                num_chunks, total_n * sizeof(float) / 1e6);

    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in, total_n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out, total_n * sizeof(float)));
    std::fill(h_in, h_in + total_n, 2.0f);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, total_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_n * sizeof(float)));

    int block = 256;
    int grid = static_cast<int>((chunk_n + block - 1) / block);

    // Serialized: default stream, full copy-compute-copy per chunk, no overlap.
    float serial_ms = benchmark_gpu([&]() {
        for (int c = 0; c < num_chunks; ++c) {
            size_t off = c * chunk_n;
            CUDA_CHECK(cudaMemcpy(d_in + off, h_in + off, chunk_bytes, cudaMemcpyHostToDevice));
            square_kernel<<<grid, block>>>(d_in + off, d_out + off, chunk_n);
            CUDA_CHECK(cudaMemcpy(h_out + off, d_out + off, chunk_bytes, cudaMemcpyDeviceToHost));
        }
    }, 2, 10);

    // Overlapped: each chunk gets its own stream, so while stream 0's kernel
    // runs, stream 1's H2D copy can proceed concurrently on the copy engine
    // (hardware permitting), and stream (c-1)'s D2H copy can overlap with
    // stream c's kernel -- a 3-stage software pipeline across chunks.
    std::vector<cudaStream_t> streams(num_chunks);
    for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

    float overlap_ms = benchmark_gpu([&]() {
        for (int c = 0; c < num_chunks; ++c) {
            size_t off = c * chunk_n;
            cudaStream_t s = streams[c];
            CUDA_CHECK(cudaMemcpyAsync(d_in + off, h_in + off, chunk_bytes, cudaMemcpyHostToDevice, s));
            square_kernel<<<grid, block, 0, s>>>(d_in + off, d_out + off, chunk_n);
            CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off, chunk_bytes, cudaMemcpyDeviceToHost, s));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }, 2, 10);

    std::printf("  Serialized (1 stream)      : %8.4f ms\n", serial_ms);
    std::printf("  Overlapped (%d streams)     : %8.4f ms\n", num_chunks, overlap_ms);
    std::printf("  Speedup from overlap: %.2fx\n", serial_ms / overlap_ms);

    for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ---------------------------------------------------------------------------
// Demo 3: CUDA Graph replay vs repeated stream launches
// ---------------------------------------------------------------------------
void demo_cuda_graph(size_t n, int num_kernels_in_chain) {
    std::printf("\n--- Demo 3: CUDA Graph replay vs per-iteration launch (%d chained kernels) ---\n",
                num_kernels_in_chain);

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_a, 0, n * sizeof(float)));

    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Baseline: launch the same chain of kernels directly on the stream every iteration.
    float direct_ms = benchmark_gpu([&]() {
        for (int k = 0; k < num_kernels_in_chain; ++k) {
            if (k % 2 == 0) square_kernel<<<grid, block, 0, stream>>>(d_a, d_b, n);
            else            square_kernel<<<grid, block, 0, stream>>>(d_b, d_a, n);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }, 3, 30);

    // Capture the identical chain into a graph once, then replay it repeatedly.
    // This amortizes CPU-side launch overhead (driver validation, queueing)
    // across every replay, which matters most when kernels are numerous/small.
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < num_kernels_in_chain; ++k) {
        if (k % 2 == 0) square_kernel<<<grid, block, 0, stream>>>(d_a, d_b, n);
        else            square_kernel<<<grid, block, 0, stream>>>(d_b, d_a, n);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    float graph_ms = benchmark_gpu([&]() {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }, 3, 30);

    std::printf("  Direct per-iteration launch: %8.4f ms\n", direct_ms);
    std::printf("  CUDA Graph replay          : %8.4f ms\n", graph_ms);
    std::printf("  Speedup from graph capture: %.2fx\n", direct_ms / graph_ms);

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

int main() {
    demo_pinned_vs_pageable(1ull << 24);      // 64 MB
    demo_stream_overlap(1ull << 24, 8);       // 64 MB total, 8 chunks/streams
    demo_cuda_graph(1ull << 16, 20);          // small buffer, 20 tiny chained kernels
    std::printf("\nAll Stage 5 runtime-optimization demos completed.\n");
    return 0;
}
