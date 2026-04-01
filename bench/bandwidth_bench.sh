#!/bin/bash
# bandwidth_bench.sh -- Memory bandwidth profiling for TurboQuant KV cache
#
# Measures tok/s at different KV types and context lengths to demonstrate
# that compressed KV cache maintains speed better at longer contexts.
#
# Usage:
#   bash bench/bandwidth_bench.sh <model.tqm> [--threads N]
#
# Output: CSV-formatted results suitable for plotting.
#
# Target: turbo_kv_1b shows < 10% speed degradation at 500 tokens vs 50 tokens.

set -euo pipefail

MODEL="${1:?Usage: $0 <model.tqm> [--threads N]}"
THREADS="${3:-4}"
TQ_RUN="${TQ_RUN:-./build/tq_run}"

if [ ! -f "$TQ_RUN" ]; then
    echo "Error: tq_run not found at $TQ_RUN. Build first." >&2
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: model not found at $MODEL" >&2
    exit 1
fi

# Parse optional --threads
if [ "${2:-}" = "--threads" ] && [ -n "${3:-}" ]; then
    THREADS="$3"
fi

echo "=== TurboQuant Memory Bandwidth Benchmark ==="
echo "Model: $MODEL"
echo "Threads: $THREADS"
echo ""

# KV types to test
KV_TYPES=("fp32" "uniform_4b" "uniform_2b" "turbo_kv_1b" "turbo_kv_3b")
V_TYPES=("fp16" "q4" "q2")

# Header
echo "kv_type,v_quant,context,tok_s,time_s"

for kv in "${KV_TYPES[@]}"; do
    for vq in "${V_TYPES[@]}"; do
        # Skip incompatible combos (fp32 kv doesn't need v quant)
        if [ "$kv" = "fp32" ] && [ "$vq" != "fp16" ]; then
            continue
        fi

        # Run the bench-memory mode
        output=$("$TQ_RUN" "$MODEL" \
            -k "$kv" \
            -v "$vq" \
            -j "$THREADS" \
            -q q4 \
            --bench-memory 2>&1 || true)

        # Extract BENCH_CSV lines
        echo "$output" | grep "^BENCH_CSV:" | while IFS=: read -r _ csv; do
            ctx=$(echo "$csv" | cut -d, -f1)
            tok_s=$(echo "$csv" | cut -d, -f2)
            time_s=$(echo "$csv" | cut -d, -f3)
            echo "${kv},${vq},${ctx},${tok_s},${time_s}"
        done
    done
done

echo ""
echo "=== Benchmark Complete ==="
echo ""
echo "Analysis:"
echo "  - Compare tok/s across context lengths for each KV type"
echo "  - Compressed KV (turbo_kv_1b) should maintain speed better at longer contexts"
echo "  - Target: turbo_kv_1b < 10% degradation at 500 tokens vs 50 tokens"
