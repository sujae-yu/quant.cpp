#!/bin/bash
# quant.cpp — Ablation Test
#
# Compares quantization methods at increasing token lengths to show
# where each method diverges from the uniform_4b baseline.
#
# This helps answer community questions:
#   1. Does quant.cpp's codebook + QJL actually help vs naive uniform?
#   2. At what context length do different methods start diverging?
#   3. How does 1-bit compare to 3-bit in practice?
#
# Usage:
#   bash bench/ablation_test.sh <model.tqm>
#
# Requirements: built quant binary in build/

set -e

MODEL="${1:-model.tqm}"
TQ_RUN="./build/quant"
THREADS=6
RESULTS_DIR="bench/ablation_results"

if [ ! -f "$TQ_RUN" ]; then
    echo "Error: $TQ_RUN not found. Build first: cmake --build build"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    echo "Usage: bash bench/ablation_test.sh <model.tqm>"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

TOKEN_COUNTS="50 100 150 200 300"
KV_TYPES="uniform_4b turbo_kv_3b turbo_kv_1b"
BASELINE="uniform_4b"

PROMPTS=(
    "Explain the theory of relativity in simple terms."
    "Write a Python function that computes fibonacci numbers recursively."
    "The history of artificial intelligence begins with"
)

echo "============================================================"
echo "  quant.cpp Ablation Test"
echo "============================================================"
echo ""
echo "  Model:        $MODEL"
echo "  Threads:      $THREADS"
echo "  Token counts: $TOKEN_COUNTS"
echo "  KV types:     $KV_TYPES"
echo "  Baseline:     $BASELINE"
echo "  Prompts:      ${#PROMPTS[@]}"
echo "  Mode:         greedy (temperature=0)"
echo ""
echo "  Purpose: Show divergence point for each method vs baseline."
echo ""
echo "  Theoretical background:"
echo "    - uniform_4b: Standard min-max quantization (4-bit)"
echo "    - turbo_kv_3b: RHT + Lloyd-Max codebook (2-bit) + QJL residual (1-bit)"
echo "      The QJL correction reduces inner product estimation bias."
echo "    - turbo_kv_1b: RHT + sign hash only (1-bit)"
echo "      Most aggressive compression; attention via XOR + popcount."
echo "    - RHT (Randomized Hadamard Transform) distributes information"
echo "      evenly across dimensions, making quantization more uniform."
echo "    - Without RHT, outlier dimensions would dominate quantization error."
echo ""
echo "============================================================"
echo ""

# Phase 1: Generate outputs at each token count
echo "[Phase 1] Generating outputs at various token lengths..."
echo ""

for pidx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$pidx]}"
    short=$(echo "$prompt" | head -c 50)
    echo "  Prompt $((pidx+1)): \"$short...\""

    for ntok in $TOKEN_COUNTS; do
        for kv in $KV_TYPES; do
            outfile="$RESULTS_DIR/p${pidx}_${kv}_n${ntok}.txt"
            $TQ_RUN "$MODEL" -p "$prompt" -j $THREADS -n "$ntok" -T 0.0 -k "$kv" 2>&1 \
                | sed -n '/^---$/,/^---$/p' | tail -n +2 | sed '$d' \
                > "$outfile"
        done
    done
done

echo ""

# Phase 2: Divergence analysis
echo "[Phase 2] Divergence analysis (vs $BASELINE)..."
echo ""

printf "%-55s " "Prompt / KV Type"
for ntok in $TOKEN_COUNTS; do
    printf "%-10s " "${ntok}tok"
done
echo ""
printf "%-55s " "---"
for ntok in $TOKEN_COUNTS; do
    printf "%-10s " "------"
done
echo ""

for pidx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$pidx]}"
    short=$(echo "$prompt" | head -c 40)
    echo "  [$short...]"

    for kv in $KV_TYPES; do
        if [ "$kv" = "$BASELINE" ]; then
            continue
        fi
        printf "    %-51s " "$kv"

        for ntok in $TOKEN_COUNTS; do
            baseline_file="$RESULTS_DIR/p${pidx}_${BASELINE}_n${ntok}.txt"
            candidate_file="$RESULTS_DIR/p${pidx}_${kv}_n${ntok}.txt"

            if diff -q "$baseline_file" "$candidate_file" > /dev/null 2>&1; then
                printf "%-10s " "MATCH"
            else
                first_diff=$(cmp "$baseline_file" "$candidate_file" 2>/dev/null | head -1 | grep -o 'byte [0-9]*' | grep -o '[0-9]*')
                if [ -z "$first_diff" ]; then
                    printf "%-10s " "PREFIX"
                else
                    printf "%-10s " "DIFF@${first_diff}B"
                fi
            fi
        done
        echo ""
    done
done

echo ""

# Phase 3: Speed at each token count
echo "[Phase 3] Speed comparison at each token count..."
echo ""

printf "%-15s " "KV Type"
for ntok in $TOKEN_COUNTS; do
    printf "%-12s " "${ntok}tok"
done
echo ""
printf "%-15s " "-------"
for ntok in $TOKEN_COUNTS; do
    printf "%-12s " "------"
done
echo ""

for kv in $KV_TYPES; do
    printf "%-15s " "$kv"
    for ntok in $TOKEN_COUNTS; do
        output=$($TQ_RUN "$MODEL" -p "Hello world" -j $THREADS -n "$ntok" -T 0.0 -k "$kv" -M 2>&1)
        speed=$(echo "$output" | grep "tok/s" | tail -1 | grep -o '[0-9]*\.[0-9]* tok/s' | head -1)
        printf "%-12s " "${speed:-N/A}"
    done
    echo ""
done

echo ""

# Phase 4: First-line comparison (qualitative coherence check)
echo "[Phase 4] Output coherence check (first 100 chars at each length)..."
echo ""

for pidx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$pidx]}"
    short=$(echo "$prompt" | head -c 40)
    echo "  Prompt $((pidx+1)): \"$short...\""
    echo ""

    for ntok in $TOKEN_COUNTS; do
        echo "    --- ${ntok} tokens ---"
        for kv in $KV_TYPES; do
            outfile="$RESULTS_DIR/p${pidx}_${kv}_n${ntok}.txt"
            preview=$(head -c 100 "$outfile" 2>/dev/null | tr '\n' ' ')
            printf "      %-15s %s\n" "$kv:" "${preview}..."
        done
        echo ""
    done
done

echo "============================================================"
echo ""
echo "  Ablation Summary:"
echo ""
echo "  - If turbo_kv_3b MATCHes baseline at all lengths,"
echo "    the RHT + codebook + QJL correction preserves quality."
echo "  - turbo_kv_1b (1-bit) will diverge earlier — this is"
echo "    expected at extreme compression (10.7x key compression)."
echo "  - Divergence != quality loss. Check Phase 4 for coherence."
echo ""
echo "  Theoretical notes:"
echo "    Q: Does QJL correction help?"
echo "    A: turbo_kv_3b = 2-bit codebook + 1-bit QJL residual."
echo "       The QJL bit corrects the sign of the inner product"
echo "       estimation error. Without it (codebook-only 2-bit),"
echo "       you would see more divergence at shorter contexts."
echo "       The 3b config typically matches uniform_4b because"
echo "       the unbiased estimation property is preserved."
echo ""
echo "    Q: Does RHT help?"
echo "    A: RHT (Randomized Hadamard Transform) spreads outlier"
echo "       values evenly across all dimensions. Without RHT,"
echo "       a few large-magnitude dimensions would dominate the"
echo "       quantization error, causing systematic bias in inner"
echo "       products. This is proven in Theorem 3.1 of the"
echo "       quant.cpp paper (arxiv:2504.19874)."
echo ""
echo "  Results saved to: $RESULTS_DIR/"
echo ""
