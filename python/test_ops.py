# cuCore/python/test_ops.py
#
# Correctness + benchmark harness for the PyTorch extension ops, checked
# against PyTorch's own (cuDNN/ATen-backed) implementations.
#
# Run with:  cd python && pip install -e . && python test_ops.py

import time
import torch

import cucore_ops  # built via `pip install -e .` in this directory


def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def test_softmax():
    print("\n=== my_softmax vs torch.softmax ===")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)

    y_ref = torch.softmax(x, dim=-1)
    y_ours = cucore_ops.my_softmax(x)

    max_err = (y_ref - y_ours).abs().max().item()
    print(f"  max abs error: {max_err:.6e}")
    assert torch.allclose(y_ref, y_ours, atol=1e-4, rtol=1e-4), "my_softmax mismatch vs torch.softmax!"
    print("  correctness: PASS")

    ms_ref = bench(lambda: torch.softmax(x, dim=-1))
    ms_ours = bench(lambda: cucore_ops.my_softmax(x))
    print(f"  torch.softmax : {ms_ref:.4f} ms")
    print(f"  my_softmax    : {ms_ours:.4f} ms")
    print(f"  ratio (ours / torch): {ms_ours / ms_ref:.2f}x")


def test_layernorm():
    print("\n=== my_layernorm vs torch.nn.functional.layer_norm ===")
    rows, cols = 4096, 1024
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
    gamma = torch.randn(cols, device="cuda", dtype=torch.float32)
    beta = torch.randn(cols, device="cuda", dtype=torch.float32)
    eps = 1e-5

    y_ref = torch.nn.functional.layer_norm(x, (cols,), weight=gamma, bias=beta, eps=eps)
    y_ours = cucore_ops.my_layernorm(x, gamma, beta, eps)

    max_err = (y_ref - y_ours).abs().max().item()
    print(f"  max abs error: {max_err:.6e}")
    assert torch.allclose(y_ref, y_ours, atol=1e-3, rtol=1e-3), "my_layernorm mismatch vs torch layer_norm!"
    print("  correctness: PASS")

    ms_ref = bench(lambda: torch.nn.functional.layer_norm(x, (cols,), weight=gamma, bias=beta, eps=eps))
    ms_ours = bench(lambda: cucore_ops.my_layernorm(x, gamma, beta, eps))
    print(f"  torch.layer_norm : {ms_ref:.4f} ms")
    print(f"  my_layernorm     : {ms_ours:.4f} ms")
    print(f"  ratio (ours / torch): {ms_ours / ms_ref:.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required to run cuCore's PyTorch extension tests"
    test_softmax()
    test_layernorm()
    print("\nAll PyTorch extension tests passed.")
