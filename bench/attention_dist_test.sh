#!/bin/bash
# attention_dist_test.sh -- Attention score distribution preservation test
#
# Runs the attention distribution test suite that verifies TurboQuant
# preserves the full attention score distribution (cosine similarity,
# Spearman rank correlation, top-k overlap), not just argmax.
#
# Also proves random keys break attention (non-trivial compression)
# and compares TurboQuant vs uniform at same bit-width.
#
# Usage: bash bench/attention_dist_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

echo "=== Attention Score Distribution Preservation Test ==="
echo ""

# Build if needed
if [ ! -f "${BUILD_DIR}/test_attention_distribution" ]; then
    echo "Building test_attention_distribution..."
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON \
        "$PROJECT_DIR" > /dev/null 2>&1
    cmake --build "$BUILD_DIR" --target test_attention_distribution \
        -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" > /dev/null 2>&1
fi

# Run with verbose output
"${BUILD_DIR}/test_attention_distribution" --gtest_print_time=1
