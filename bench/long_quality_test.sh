#!/bin/bash
# TurboQuant.cpp — Long Context Quality Test
#
# Runs generation at 200, 500, 1000 tokens and compares:
#   - uniform_4b (baseline)
#   - turbo_kv_1b (K-only, V as FP16)
#   - turbo_kv_1b + Q4 V
#
# Shows first few lines of output to verify coherence,
# and reports speed at each length.
#
# Usage:
#   bash bench/long_quality_test.sh <model.tqm>
#
# Requirements: built tq_run binary in build/

set -e

MODEL="${1:-model.tqm}"
TQ_RUN="./build/tq_run"
THREADS=6
RESULTS_DIR="bench/long_quality_results"

if [ ! -f "$TQ_RUN" ]; then
    echo "Error: $TQ_RUN not found. Build first: cmake --build build"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    echo "Usage: bash bench/long_quality_test.sh <model.tqm>"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

TOKEN_COUNTS="200 500 1000"
PROMPT="Explain the history of computing from the earliest mechanical calculators to modern artificial intelligence. Include key inventions, people, and breakthroughs along the way."

echo "============================================================"
echo "  TurboQuant Long Context Quality Test"
echo "============================================================"
echo ""
echo "  Model:        $MODEL"
echo "  Threads:      $THREADS"
echo "  Token counts: $TOKEN_COUNTS"
echo "  Mode:         greedy (temperature=0)"
echo ""
echo "  Configs tested:"
echo "    1. uniform_4b         — 4-bit K, FP16/FP32 V (baseline)"
echo "    2. turbo_kv_1b        — 1-bit K, FP16/FP32 V"
echo "    3. turbo_kv_1b + Q4 V — 1-bit K, 4-bit V (max compression)"
echo ""
echo "============================================================"
echo ""

# Define configs: name, kv_flag, v_flag
CONFIGS=(
    "uniform_4b:::uniform_4b:::"
    "turbo_kv_1b:::turbo_kv_1b:::"
    "turbo_1b+q4v:::turbo_kv_1b:::-v q4"
)

# Phase 1: Generate and measure
echo "[Phase 1] Generating outputs at various lengths..."
echo ""

for ntok in $TOKEN_COUNTS; do
    echo "  --- $ntok tokens ---"

    for config_str in "${CONFIGS[@]}"; do
        IFS=':::' read -r name kv_type v_flag <<< "$config_str"

        outfile="$RESULTS_DIR/${name}_n${ntok}.txt"
        statsfile="$RESULTS_DIR/${name}_n${ntok}_stats.txt"

        # Build command
        cmd="$TQ_RUN $MODEL -p \"$PROMPT\" -j $THREADS -n $ntok -T 0.0 -k $kv_type"
        if [ -n "$v_flag" ]; then
            cmd="$cmd $v_flag"
        fi
        cmd="$cmd -M"

        # Run and capture both output text and stats
        eval "$cmd" 2>&1 > "$statsfile"

        # Extract generated text
        sed -n '/^---$/,/^---$/p' "$statsfile" | tail -n +2 | sed '$d' > "$outfile"

        # Extract speed
        speed=$(grep "tok/s" "$statsfile" | tail -1 | grep -o '[0-9]*\.[0-9]* tok/s' | head -1)
        per_token=$(grep "Per-token KV" "$statsfile" | head -1 | grep -o '[0-9]*\.[0-9]* KB')
        ratio=$(grep "Compression" "$statsfile" | grep -o '[0-9]*\.[0-9]*x')

        printf "    %-18s %10s   KV/tok: %10s   Compress: %s\n" \
            "$name" "${speed:-N/A}" "${per_token:-N/A}" "${ratio:-N/A}"
    done
    echo ""
done

# Phase 2: Coherence check (first few lines)
echo "[Phase 2] Output coherence check..."
echo ""

for ntok in $TOKEN_COUNTS; do
    echo "  === $ntok tokens ==="
    echo ""

    for config_str in "${CONFIGS[@]}"; do
        IFS=':::' read -r name kv_type v_flag <<< "$config_str"

        outfile="$RESULTS_DIR/${name}_n${ntok}.txt"
        echo "  [$name]:"

        # Show first 3 lines (or fewer if output is short)
        head -3 "$outfile" 2>/dev/null | while IFS= read -r line; do
            echo "    $line"
        done

        # Show word count for reference
        wc=$(wc -w < "$outfile" 2>/dev/null | tr -d ' ')
        chars=$(wc -c < "$outfile" 2>/dev/null | tr -d ' ')
        echo "    ... ($wc words, $chars chars total)"
        echo ""
    done
done

# Phase 3: Divergence analysis
echo "[Phase 3] Divergence analysis (vs uniform_4b baseline)..."
echo ""

printf "%-20s %-12s %-12s %-12s\n" "Config" "200 tok" "500 tok" "1000 tok"
printf "%-20s %-12s %-12s %-12s\n" "------" "-------" "-------" "--------"

for config_str in "${CONFIGS[@]}"; do
    IFS=':::' read -r name kv_type v_flag <<< "$config_str"

    if [ "$name" = "uniform_4b" ]; then
        printf "%-20s %-12s %-12s %-12s\n" "$name" "baseline" "baseline" "baseline"
        continue
    fi

    printf "%-20s " "$name"
    for ntok in $TOKEN_COUNTS; do
        baseline="$RESULTS_DIR/uniform_4b_n${ntok}.txt"
        candidate="$RESULTS_DIR/${name}_n${ntok}.txt"

        if diff -q "$baseline" "$candidate" > /dev/null 2>&1; then
            printf "%-12s " "MATCH"
        else
            first_diff=$(cmp "$baseline" "$candidate" 2>/dev/null | head -1 | grep -o 'byte [0-9]*' | grep -o '[0-9]*')
            if [ -z "$first_diff" ]; then
                printf "%-12s " "PREFIX"
            else
                printf "%-12s " "DIFF@${first_diff}B"
            fi
        fi
    done
    echo ""
done

echo ""

# Phase 4: Speed summary
echo "[Phase 4] Speed summary..."
echo ""

printf "%-20s " "Config"
for ntok in $TOKEN_COUNTS; do
    printf "%-14s " "${ntok} tok"
done
echo ""
printf "%-20s " "------"
for ntok in $TOKEN_COUNTS; do
    printf "%-14s " "--------"
done
echo ""

for config_str in "${CONFIGS[@]}"; do
    IFS=':::' read -r name kv_type v_flag <<< "$config_str"

    printf "%-20s " "$name"
    for ntok in $TOKEN_COUNTS; do
        statsfile="$RESULTS_DIR/${name}_n${ntok}_stats.txt"
        speed=$(grep "tok/s" "$statsfile" 2>/dev/null | tail -1 | grep -o '[0-9]*\.[0-9]* tok/s' | head -1)
        printf "%-14s " "${speed:-N/A}"
    done
    echo ""
done

echo ""
echo "============================================================"
echo ""
echo "  Key observations:"
echo "  - uniform_4b: Baseline quality at moderate compression"
echo "  - turbo_kv_1b: 10.7x key compression, outputs may diverge"
echo "    at longer contexts but should remain coherent"
echo "  - turbo_1b+q4v: Maximum compression (4.9x total K+V),"
echo "    earliest divergence expected due to V quantization"
echo ""
echo "  Results saved to: $RESULTS_DIR/"
echo ""
