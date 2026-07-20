#!/usr/bin/env bash
# cuCore/scripts/run_tests.sh
#
# Builds the project (if not already built) and runs every kernel binary's
# built-in correctness check + benchmark, aggregating a final PASS/FAIL.
#
# Usage: ./scripts/run_tests.sh [CMAKE_CUDA_ARCHITECTURES]
# Example: ./scripts/run_tests.sh 86

set -u
ARCH="${1:-native}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "== Configuring (arch=${ARCH}) =="
mkdir -p "${BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" -DCMAKE_BUILD_TYPE=Release || exit 1

echo "== Building =="
cmake --build "${BUILD_DIR}" -j || exit 1

BINARIES=(vector_add saxpy dot_product reduction scan histogram transpose matmul softmax layernorm gelu conv2d streams_demo)

FAILED=()
for bin in "${BINARIES[@]}"; do
    echo ""
    echo "===================================================================="
    echo "Running: ${bin}"
    echo "===================================================================="
    "${BUILD_DIR}/bin/${bin}"
    status=$?
    if [ ${status} -ne 0 ]; then
        FAILED+=("${bin}")
    fi
done

echo ""
echo "===================================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL KERNELS PASSED CORRECTNESS CHECKS"
    exit 0
else
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
