#!/bin/bash
# TurboQuant.cpp — Address Sanitizer + Undefined Behavior Sanitizer validation
#
# Builds with -fsanitize=address,undefined, runs all tests, and optionally
# runs a short inference to catch memory errors.
#
# Usage:
#   bash scripts/sanitize.sh [model.tqm]
#
# If no model is provided, only tests are run (no inference check).

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="build-sanitize"
MODEL="${1:-}"
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "============================================================"
echo "  TurboQuant.cpp — Sanitizer Validation"
echo "============================================================"
echo ""
echo "  Build dir:  $BUILD_DIR"
echo "  Sanitizers: AddressSanitizer + UndefinedBehaviorSanitizer"
echo "  Model:      ${MODEL:-none (tests only)}"
echo ""

# Step 1: Configure with sanitizers
echo "[1/3] Configuring sanitizer build..."
cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
    -DTQ_BUILD_TESTS=ON \
    -DTQ_BUILD_BENCH=OFF \
    -Wno-dev 2>/dev/null
echo "  Done."

# Step 2: Build
echo ""
echo "[2/3] Building with sanitizers..."
cmake --build "$BUILD_DIR" -j"$NCPU" 2>&1 | tail -5
echo "  Done."

# Step 3: Run tests
echo ""
echo "[3/3] Running tests under sanitizers..."
echo ""

ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:print_stats=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
ctest --test-dir "$BUILD_DIR" --output-on-failure 2>&1
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo ""
    echo "  FAIL: Tests failed under sanitizers!"
    echo "  Review output above for ASan/UBSan errors."
    exit 1
fi

echo ""
echo "  All tests passed under sanitizers."

# Step 4: Optional inference check
if [ -n "$MODEL" ] && [ -f "$MODEL" ]; then
    echo ""
    echo "[bonus] Running short inference under sanitizers..."
    TQ_RUN="$BUILD_DIR/tq_run"
    if [ -f "$TQ_RUN" ]; then
        ASAN_OPTIONS="detect_leaks=1:halt_on_error=1" \
        UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
        "$TQ_RUN" "$MODEL" -p "Hello" -n 10 -T 0.0 2>&1
        echo ""
        echo "  Inference completed without sanitizer errors."
    else
        echo "  Warning: $TQ_RUN not found, skipping inference check."
    fi
elif [ -n "$MODEL" ]; then
    echo ""
    echo "  Warning: Model file '$MODEL' not found, skipping inference check."
fi

echo ""
echo "============================================================"
echo "  Sanitizer validation: PASSED"
echo "============================================================"
