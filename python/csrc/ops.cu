// cuCore/python/csrc/ops.cu
//
// Stage 6 — PyTorch Extension: exposes cuCore's optimized softmax and
// layernorm kernels as real torch ops (torch.ops.cucore.my_softmax /
// torch.ops.cucore.my_layernorm), callable directly on torch.Tensor from
// Python, dispatched through pybind11 + TORCH_LIBRARY.
//
// These reuse the exact single-pass, warp-shuffle-reduced kernel designs
// from kernel/softmax and kernel/layernorm (see those files for the
// detailed algorithmic commentary); this file is just the torch-facing
// wrapper: tensor validation, kernel launch, and op registration.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

// ---------------------------------------------------------------------------
// Softmax kernel (single-pass, block-per-row, warp-shuffle reductions)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int BLOCK_SIZE>
__global__ void softmax_fwd_kernel(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
    __shared__ float buf[BLOCK_SIZE / 32];
    int r = blockIdx.x;
    const float* row_in = in + (size_t)r * cols;
    float* row_out = out + (size_t)r * cols;
    int lane = threadIdx.x % 32, warp_id = threadIdx.x / 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) m = fmaxf(m, row_in[c]);
    m = warp_reduce_max(m);
    if (lane == 0) buf[warp_id] = m;
    __syncthreads();
    if (warp_id == 0) { float v = (lane < NUM_WARPS) ? buf[lane] : -INFINITY; v = warp_reduce_max(v); if (lane == 0) buf[0] = v; }
    __syncthreads();
    m = buf[0];
    __syncthreads();

    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) { float e = __expf(row_in[c] - m); row_out[c] = e; s += e; }
    s = warp_reduce_sum(s);
    if (lane == 0) buf[warp_id] = s;
    __syncthreads();
    if (warp_id == 0) { float v = (lane < NUM_WARPS) ? buf[lane] : 0.f; v = warp_reduce_sum(v); if (lane == 0) buf[0] = v; }
    __syncthreads();
    s = buf[0];

    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) row_out[c] /= s;
}

torch::Tensor my_softmax(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 2, "my_softmax expects a 2D tensor [rows, cols]");

    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    constexpr int BLOCK = 256;

    auto stream = at::cuda::getCurrentCUDAStream();
    softmax_fwd_kernel<BLOCK><<<rows, BLOCK, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);

    return output;
}

// ---------------------------------------------------------------------------
// LayerNorm kernel (single-pass Welford, block-per-row, warp-shuffle merge)
// ---------------------------------------------------------------------------
struct WelfordState { float mean; float m2; float count; };

__device__ __forceinline__ WelfordState welford_merge(WelfordState a, WelfordState b) {
    if (a.count == 0.f) return b;
    if (b.count == 0.f) return a;
    float count = a.count + b.count;
    float delta = b.mean - a.mean;
    float mean = a.mean + delta * (b.count / count);
    float m2 = a.m2 + b.m2 + delta * delta * (a.count * b.count / count);
    return {mean, m2, count};
}
__device__ __forceinline__ WelfordState warp_reduce_welford(WelfordState v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        WelfordState other;
        other.mean  = __shfl_xor_sync(0xffffffff, v.mean, o);
        other.m2    = __shfl_xor_sync(0xffffffff, v.m2, o);
        other.count = __shfl_xor_sync(0xffffffff, v.count, o);
        v = welford_merge(v, other);
    }
    return v;
}

template <int BLOCK_SIZE>
__global__ void layernorm_fwd_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      const float* __restrict__ gamma, const float* __restrict__ beta,
                                      int rows, int cols, float eps) {
    __shared__ WelfordState warp_states[BLOCK_SIZE / 32];
    int r = blockIdx.x;
    const float* row_in = in + (size_t)r * cols;
    float* row_out = out + (size_t)r * cols;
    int lane = threadIdx.x % 32, warp_id = threadIdx.x / 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    WelfordState state{0.f, 0.f, 0.f};
    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE) {
        float x = row_in[c];
        state.count += 1.f;
        float delta = x - state.mean;
        state.mean += delta / state.count;
        state.m2 += delta * (x - state.mean);
    }
    state = warp_reduce_welford(state);
    if (lane == 0) warp_states[warp_id] = state;
    __syncthreads();
    if (warp_id == 0) {
        WelfordState v = (lane < NUM_WARPS) ? warp_states[lane] : WelfordState{0.f, 0.f, 0.f};
        v = warp_reduce_welford(v);
        if (lane == 0) warp_states[0] = v;
    }
    __syncthreads();
    float mean = warp_states[0].mean;
    float inv_std = rsqrtf(warp_states[0].m2 / cols + eps);
    __syncthreads();

    for (int c = threadIdx.x; c < cols; c += BLOCK_SIZE)
        row_out[c] = (row_in[c] - mean) * inv_std * gamma[c] + beta[c];
}

torch::Tensor my_layernorm(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    TORCH_CHECK(input.dim() == 2, "my_layernorm expects a 2D tensor [rows, cols]");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == input.size(1), "gamma shape must match cols");
    TORCH_CHECK(beta.dim() == 1 && beta.size(0) == input.size(1), "beta shape must match cols");

    auto output = torch::empty_like(input);
    int rows = input.size(0);
    int cols = input.size(1);
    constexpr int BLOCK = 256;

    auto stream = at::cuda::getCurrentCUDAStream();
    layernorm_fwd_kernel<BLOCK><<<rows, BLOCK, 0, stream>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(), rows, cols, static_cast<float>(eps));

    return output;
}

// ---------------------------------------------------------------------------
// pybind11 module (import cucore_ops; cucore_ops.my_softmax(x))
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_softmax", &my_softmax, "cuCore single-pass warp-reduced softmax (CUDA)");
    m.def("my_layernorm", &my_layernorm, "cuCore single-pass Welford layernorm (CUDA)",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
}

// ---------------------------------------------------------------------------
// Also register as torch.ops.cucore.* so it works with torch.compile /
// custom-op dispatch, not just the direct pybind11 module import.
// ---------------------------------------------------------------------------
TORCH_LIBRARY(cucore, m) {
    m.def("my_softmax(Tensor input) -> Tensor");
    m.def("my_layernorm(Tensor input, Tensor gamma, Tensor beta, float eps=1e-5) -> Tensor");
}
TORCH_LIBRARY_IMPL(cucore, CUDA, m) {
    m.impl("my_softmax", &my_softmax);
    m.impl("my_layernorm", &my_layernorm);
}
