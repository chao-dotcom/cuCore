// cuCore/kernel/transpose/transpose.cu
//
// Stage 2 — Memory Optimization showcase: Matrix Transpose  (out = in^T)
//
// Transpose is the canonical kernel for teaching memory coalescing, because
// a naive implementation cannot have BOTH reads and writes coalesced at the
// same time -- one of the two directions always strides by `width`.
//
// Naive:     reads are coalesced (row-major, threadIdx.x walks contiguous
//            columns) but writes are NOT: out[x][y] = in[y][x] means
//            consecutive threads write to addresses `width` elements apart,
//            i.e. one element per cache line touched -> effectively N
//            separate 4-byte transactions instead of 1 coalesced 128-byte
//            transaction.
// Optimized: stage the tile through SHARED memory. Threads read a
//            TILE x TILE block with coalesced global reads, write it into
//            shared memory, syncthreads, then write it back out to global
//            memory ALSO with coalesced accesses (by swapping which index
//            reads from shared memory transposed instead of global memory).
//            The shared-memory tile is padded to TILE+1 columns so that
//            the transposed access pattern (column-major reads out of
//            shared memory) doesn't map every thread in a warp onto the
//            same memory bank (classic N-way bank conflict avoidance).

#include "../../common/cuda_utils.cuh"

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8; // each thread handles TILE/BLOCK_ROWS rows of the tile

void transpose_cpu(const float* in, float* out, int width, int height) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            out[x * height + y] = in[y * width + x];
}

// ---------------------------------------------------------------------------
// Naive: direct global-to-global transpose. Coalesced reads, scattered writes.
// ---------------------------------------------------------------------------
__global__ void transpose_naive_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int width, int height) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height)
            out[x * height + (y + j)] = in[(y + j) * width + x]; // scattered write, stride = height
    }
}

// ---------------------------------------------------------------------------
// Optimized: shared-memory tiled transpose with padding (+1) to avoid
// bank conflicts on the transposed shared-memory read.
// ---------------------------------------------------------------------------
__global__ void transpose_opt_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      int width, int height) {
    __shared__ float tile[TILE][TILE + 1]; // +1 padding: breaks the stride-32 bank collision

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Coalesced read from global memory into the shared-memory tile.
#pragma unroll
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height)
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
    __syncthreads();

    // Recompute output coordinates: swap block indices so this block now
    // writes the tile that corresponds to its transposed location.
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // Coalesced write to global memory, reading the tile transposed out of
    // shared memory (cheap on-chip, and conflict-free thanks to padding).
#pragma unroll
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width)
            out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

int main(int argc, char** argv) {
    const int width  = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int height = (argc > 2) ? std::atoi(argv[2]) : 4096;
    const size_t N = static_cast<size_t>(width) * height;
    std::printf("Transpose benchmark, %d x %d matrix (%.2f MB)\n", width, height, N * sizeof(float) / 1e6);

    std::vector<float> h_in(N), h_out_ref(N), h_out_gpu(N);
    fill_random(h_in);

    double cpu_ms = benchmark_cpu([&]() { transpose_cpu(h_in.data(), h_out_ref.data(), width, height); }, 1, 2);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    float naive_ms = benchmark_gpu([&]() {
        transpose_naive_kernel<<<grid, block>>>(d_in, d_out, width, height);
    });
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool naive_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N);

    float opt_ms = benchmark_gpu([&]() {
        transpose_opt_kernel<<<grid, block>>>(d_in, d_out, width, height);
    });
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool opt_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N);

    BenchResult r;
    r.name = "transpose";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = 2.0 * N * sizeof(float); // read once, write once
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return (naive_ok && opt_ok) ? 0 : 1;
}
