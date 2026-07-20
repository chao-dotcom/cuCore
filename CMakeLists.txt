cmake_minimum_required(VERSION 3.18)
project(cuCore LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Override with: cmake -DCMAKE_CUDA_ARCHITECTURES="80;86;89" ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(${CMAKE_SOURCE_DIR}/common)

find_package(CUDAToolkit REQUIRED)

# Helper: define a kernel target "name" from a single .cu file that contains
# its own main(). Each kernel binary IS its own benchmark + correctness test
# (per the project spec: CPU baseline -> naive -> optimized -> benchmark).
function(add_kernel_target name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE CUDA::cudart)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endfunction()

# ---------------------------------------------------------------------------
# Stage 1 — Fundamental kernels
# ---------------------------------------------------------------------------
add_kernel_target(vector_add   kernel/vector_add/vector_add.cu)
add_kernel_target(saxpy        kernel/saxpy/saxpy.cu)
add_kernel_target(dot_product  kernel/dot_product/dot_product.cu)
add_kernel_target(reduction    kernel/reduction/reduction.cu)
add_kernel_target(scan         kernel/scan/scan.cu)
add_kernel_target(histogram    kernel/histogram/histogram.cu)

# ---------------------------------------------------------------------------
# Stage 2 — Memory optimization showcase
# ---------------------------------------------------------------------------
add_kernel_target(transpose    kernel/transpose/transpose.cu)

# ---------------------------------------------------------------------------
# Stage 3 — Matrix computation
# ---------------------------------------------------------------------------
add_kernel_target(matmul       kernel/matmul/matmul.cu)
target_link_libraries(matmul PRIVATE CUDA::cublas)


# ---------------------------------------------------------------------------
# Stage 4 — DL operators
# ---------------------------------------------------------------------------
add_kernel_target(softmax      kernel/softmax/softmax.cu)
add_kernel_target(layernorm    kernel/layernorm/layernorm.cu)
add_kernel_target(gelu         kernel/gelu/gelu.cu)
add_kernel_target(conv2d       kernel/conv2d/conv2d.cu)

# ---------------------------------------------------------------------------
# Stage 5 — Runtime optimization demo (streams / pinned memory / graphs)
# ---------------------------------------------------------------------------
add_kernel_target(streams_demo kernel/streams_demo/streams_demo.cu)

# ---------------------------------------------------------------------------
# Convenience: `make bench` runs every binary back to back
# ---------------------------------------------------------------------------
add_custom_target(bench
  COMMAND ${CMAKE_COMMAND} -E echo "Running all cuCore benchmarks..."
  COMMAND vector_add
  COMMAND saxpy
  COMMAND dot_product
  COMMAND reduction
  COMMAND scan
  COMMAND histogram
  COMMAND transpose
  COMMAND matmul
  COMMAND softmax
  COMMAND layernorm
  COMMAND gelu
  COMMAND conv2d
  COMMAND streams_demo
  DEPENDS vector_add saxpy dot_product reduction scan histogram transpose matmul softmax layernorm gelu conv2d streams_demo
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
