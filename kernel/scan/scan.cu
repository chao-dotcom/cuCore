// cuCore/kernel/scan/scan.cu
//
// Stage 1 — Prefix Scan (inclusive), implemented per-block (segment size =
// blockDim.x). A production scan would add a second pass to carry block
// sums forward (like CUB's device-wide scan); we keep it single-level here
// and benchmark per-segment throughput, which is where the two algorithms'
// architectural differences actually show up.
//
// Naive:     Hillis-Steele inclusive scan. O(n log n) work, but simple and
//            fully data-parallel every step -- good depth, bad work
//            efficiency. Uses double-buffering in shared memory (read from
//            buffer A, write to buffer B) to avoid needing to serialize
//            reads-after-writes within a step.
// Optimized: Blelloch work-efficient scan (up-sweep / reduce phase +
//            down-sweep phase). O(n) work instead of O(n log n), same
//            O(log n) depth. We also apply the classic conflict-free
//            offset trick (shifting shared-memory indices by
//            index >> LOG_NUM_BANKS) to avoid bank conflicts during the
//            tree phases, which the textbook version otherwise suffers
//            from once the stride exceeds the warp size.

#include "../../common/cuda_utils.cuh"

void scan_cpu_inclusive(const float* in, float* out, size_t n) {
    float acc = 0.f;
    for (size_t i = 0; i < n; ++i) { acc += in[i]; out[i] = acc; }
}

// ---------------------------------------------------------------------------
// Naive: Hillis-Steele, double buffered in shared memory. O(n log n) work.
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void scan_naive_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    __shared__ float buf_a[BLOCK_SIZE];
    __shared__ float buf_b[BLOCK_SIZE];
    float* cur = buf_a;
    float* nxt = buf_b;

    unsigned int tid = threadIdx.x;
    size_t gid = blockIdx.x * (size_t)blockDim.x + tid;
    cur[tid] = (gid < n) ? in[gid] : 0.f;
    __syncthreads();

    for (unsigned int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        if (tid >= offset) nxt[tid] = cur[tid] + cur[tid - offset];
        else               nxt[tid] = cur[tid];
        __syncthreads();
        float* tmp = cur; cur = nxt; nxt = tmp; // swap buffers, avoids RAW hazards
    }
    if (gid < n) out[gid] = cur[tid];
}

// ---------------------------------------------------------------------------
// Optimized: Blelloch work-efficient scan with conflict-free indexing.
// ---------------------------------------------------------------------------
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

template <int BLOCK_SIZE>
__global__ void scan_opt_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    // +extra padding slots to absorb the conflict-free offset shifts.
    __shared__ float temp[BLOCK_SIZE + (BLOCK_SIZE >> LOG_NUM_BANKS)];

    unsigned int tid = threadIdx.x;
    size_t gid = blockIdx.x * (size_t)blockDim.x + tid;

    unsigned int a_off = CONFLICT_FREE_OFFSET(tid);
    temp[tid + a_off] = (gid < n) ? in[gid] : 0.f;

    unsigned int offset = 1;
    // Up-sweep (reduce) phase: build partial sums up the implicit tree.
#pragma unroll
    for (unsigned int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    // Clear the last element to convert this into an exclusive-scan root
    // (standard Blelloch trick), we'll convert back to inclusive at the end.
    if (tid == 0) {
        unsigned int last = BLOCK_SIZE - 1;
        temp[last + CONFLICT_FREE_OFFSET(last)] = 0.f;
    }

    // Down-sweep phase: traverse back down, distributing partial sums.
#pragma unroll
    for (unsigned int d = 1; d < BLOCK_SIZE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            unsigned int ai = offset * (2 * tid + 1) - 1;
            unsigned int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // temp[] now holds the EXCLUSIVE scan; add the input back to get INCLUSIVE.
    if (gid < n) {
        unsigned int idx = tid + CONFLICT_FREE_OFFSET(tid);
        out[gid] = temp[idx] + in[gid];
    }
}

int main(int argc, char** argv) {
    constexpr int BLOCK = 256; // segment size; must be a power of 2 for Blelloch
    const size_t num_segments = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1 << 16);
    const size_t N = num_segments * BLOCK;
    std::printf("Prefix Scan benchmark, N = %zu elements (%zu segments of %d)\n", N, num_segments, BLOCK);

    std::vector<float> h_in(N), h_out_ref(N), h_out_gpu(N);
    fill_random(h_in, 0.f, 1.f, 11);

    // CPU reference is scanned per-segment too, to match the single-level GPU scan.
    double cpu_ms = benchmark_cpu([&]() {
        for (size_t s = 0; s < num_segments; ++s)
            scan_cpu_inclusive(h_in.data() + s * BLOCK, h_out_ref.data() + s * BLOCK, BLOCK);
    }, 1, 3);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float naive_ms = benchmark_gpu([&]() {
        scan_naive_kernel<BLOCK><<<(unsigned)num_segments, BLOCK>>>(d_in, d_out, N);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool naive_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-2, 1e-2);

    float opt_ms = benchmark_gpu([&]() {
        scan_opt_kernel<BLOCK><<<(unsigned)num_segments, BLOCK>>>(d_in, d_out, N);
    }, 3, 20);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool opt_ok = allclose(h_out_gpu.data(), h_out_ref.data(), N, 1e-2, 1e-2);

    BenchResult r;
    r.name = "scan (per-block inclusive)";
    r.cpu_ms = cpu_ms;
    r.naive_ms = naive_ms;
    r.opt_ms = opt_ms;
    r.bytes_moved = 2.0 * N * sizeof(float);
    print_bench_result(r);
    std::printf("  Correctness: naive=%s  optimized=%s\n",
                naive_ok ? "PASS" : "FAIL", opt_ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return (naive_ok && opt_ok) ? 0 : 1;
}
