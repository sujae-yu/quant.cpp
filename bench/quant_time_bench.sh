#!/bin/bash
# quant_time_bench.sh -- KV cache quantization time microbenchmark
#
# Measures wall-clock time for uniform_4b, turbo_kv_3b, turbo_kv_1b
# quantization and attention operations.
#
# Usage: bash bench/quant_time_bench.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

echo "=== KV Cache Quantization Time Benchmark ==="
echo ""

# Build if needed
if [ ! -f "${BUILD_DIR}/bench_kv_overhead" ]; then
    echo "Building bench_kv_overhead..."
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_BENCH=ON \
        "$PROJECT_DIR" > /dev/null 2>&1
    cmake --build "$BUILD_DIR" --target bench_kv_overhead -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" > /dev/null 2>&1
fi

# Run benchmark
"${BUILD_DIR}/bench_kv_overhead"
