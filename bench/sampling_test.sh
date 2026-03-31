#!/bin/bash
# TurboQuant.cpp — Temperature Sampling Comparison
#
# Tests that all KV compression types produce similar quality
# stochastic output at different temperature settings.
#
# Runs same prompt with T=0.3 (low creativity) and T=0.7 (high creativity)
# across KV types, 3 times each, to check variance and coherence.
#
# Usage:
#   bash bench/sampling_test.sh <model.tqm>
#
# Requirements: built tq_run binary in build/

set -e

MODEL="${1:-model.tqm}"
TQ_RUN="./build/tq_run"
THREADS=6
RESULTS_DIR="bench/sampling_results"

if [ ! -f "$TQ_RUN" ]; then
    echo "Error: $TQ_RUN not found. Build first: cmake --build build"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    echo "Usage: bash bench/sampling_test.sh <model.tqm>"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

TEMPS="0.3 0.7"
KV_TYPES="uniform_4b turbo_kv_3b turbo_kv_1b"
RUNS=3
NTOK=100

PROMPT="Write a short poem about the ocean."

echo "============================================================"
echo "  TurboQuant Temperature Sampling Comparison"
echo "============================================================"
echo ""
echo "  Model:        $MODEL"
echo "  Threads:      $THREADS"
echo "  Temperatures: $TEMPS"
echo "  KV types:     $KV_TYPES"
echo "  Runs per config: $RUNS"
echo "  Tokens:       $NTOK"
echo ""
echo "  Purpose: Verify that KV compression does not degrade"
echo "  stochastic sampling quality. All KV types should produce"
echo "  diverse, coherent outputs at each temperature."
echo ""
echo "============================================================"
echo ""

# Phase 1: Generate outputs
echo "[Phase 1] Generating outputs..."
echo ""

for temp in $TEMPS; do
    echo "  Temperature = $temp"
    for kv in $KV_TYPES; do
        for run in $(seq 1 $RUNS); do
            outfile="$RESULTS_DIR/T${temp}_${kv}_run${run}.txt"
            $TQ_RUN "$MODEL" -p "$PROMPT" -j $THREADS -n $NTOK -T "$temp" -k "$kv" 2>&1 \
                | sed -n '/^---$/,/^---$/p' | tail -n +2 | sed '$d' \
                > "$outfile"
        done
        printf "    %-15s %d runs done\n" "$kv" $RUNS
    done
    echo ""
done

# Phase 2: Variance analysis
echo "[Phase 2] Variance analysis..."
echo ""
echo "  For stochastic sampling (T>0), runs should differ from each other"
echo "  (showing the sampler works). We check whether variance is similar"
echo "  across KV types."
echo ""

for temp in $TEMPS; do
    echo "  === Temperature = $temp ==="
    echo ""

    printf "    %-15s %-12s %-12s %-40s\n" "KV Type" "Runs same?" "Avg chars" "Run 1 preview"
    printf "    %-15s %-12s %-12s %-40s\n" "-------" "---------" "---------" "-----------"

    for kv in $KV_TYPES; do
        # Check if all runs are identical (they shouldn't be for T>0)
        all_same="yes"
        for run in $(seq 2 $RUNS); do
            if ! diff -q "$RESULTS_DIR/T${temp}_${kv}_run1.txt" \
                         "$RESULTS_DIR/T${temp}_${kv}_run${run}.txt" > /dev/null 2>&1; then
                all_same="no"
                break
            fi
        done

        # Average character count across runs
        total_chars=0
        for run in $(seq 1 $RUNS); do
            chars=$(wc -c < "$RESULTS_DIR/T${temp}_${kv}_run${run}.txt" | tr -d ' ')
            total_chars=$((total_chars + chars))
        done
        avg_chars=$((total_chars / RUNS))

        # Preview of first run
        preview=$(head -c 40 "$RESULTS_DIR/T${temp}_${kv}_run1.txt" 2>/dev/null | tr '\n' ' ')

        printf "    %-15s %-12s %-12s %-40s\n" "$kv" "$all_same" "$avg_chars" "$preview..."
    done
    echo ""
done

# Phase 3: Coherence check — show first 2 lines of each run
echo "[Phase 3] Output samples (first 80 chars of each run)..."
echo ""

for temp in $TEMPS; do
    echo "  === Temperature = $temp ==="
    echo ""

    for kv in $KV_TYPES; do
        echo "    [$kv]"
        for run in $(seq 1 $RUNS); do
            preview=$(head -c 80 "$RESULTS_DIR/T${temp}_${kv}_run${run}.txt" 2>/dev/null | tr '\n' ' ')
            printf "      Run %d: %s...\n" "$run" "$preview"
        done
        echo ""
    done
done

echo "============================================================"
echo ""
echo "  Summary:"
echo "  - At T>0, runs should differ (sampler working correctly)"
echo "  - Average output length should be similar across KV types"
echo "  - All outputs should be coherent (check Phase 3 samples)"
echo "  - If one KV type produces drastically shorter/longer or"
echo "    incoherent output, that indicates a quality problem."
echo ""
echo "  Results saved to: $RESULTS_DIR/"
echo ""
