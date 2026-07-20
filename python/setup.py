# cuCore/python/setup.py
#
# Build with:  cd python && pip install -e .
# This compiles csrc/ops.cu into a torch extension exposing:
#   - torch.ops.cucore.my_softmax(x)
#   - torch.ops.cucore.my_layernorm(x, gamma, beta, eps)
# and also a plain pybind11 module `cucore_ops` with the same two functions,
# so it can be imported directly without going through torch.ops dispatch.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cucore_ops",
    version="0.1.0",
    description="cuCore: hand-optimized CUDA softmax/layernorm exposed as a PyTorch extension",
    ext_modules=[
        CUDAExtension(
            name="cucore_ops",
            sources=["csrc/ops.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
