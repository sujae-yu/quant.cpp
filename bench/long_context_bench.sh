#!/usr/bin/env bash
#
# long_context_bench.sh — Long Context KV Cache Memory Benchmark
#
# Measures KV cache memory usage at various context lengths, comparing
# quant.cpp (compressed Q4 KV) vs theoretical FP16 KV (llama.cpp default).
#
# Usage:
#   bash bench/long_context_bench.sh [model.tqm] [kv_type]
#
# Arguments:
#   model.tqm   Path to TQM model file (default: gemma3-4b.tqm)
#   kv_type     KV cache type (default: uniform_4b)
#
# Output:
#   - Table printed to stdout
#   - CSV saved to bench/long_context_results.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
TQ_RUN="$BUILD_DIR/quant"

# Default arguments
MODEL="${1:-gemma3-4b.tqm}"
KV_TYPE="${2:-uniform_4b}"
CSV_OUT="$SCRIPT_DIR/long_context_results.csv"

# Context lengths to test
CONTEXT_LENGTHS=(512 1024 2048 4096)

# --------------------------------------------------------
# Ensure quant is built
# --------------------------------------------------------
if [ ! -f "$TQ_RUN" ]; then
    echo "Building quant..."
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "$PROJECT_DIR" > /dev/null 2>&1
    cmake --build "$BUILD_DIR" --target quant -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" > /dev/null 2>&1
fi

if [ ! -f "$TQ_RUN" ]; then
    echo "ERROR: Failed to build quant" >&2
    exit 1
fi

# --------------------------------------------------------
# Resolve model path
# --------------------------------------------------------
if [ ! -f "$MODEL" ]; then
    # Try common locations
    for candidate in \
        "$PROJECT_DIR/$MODEL" \
        "$PROJECT_DIR/models/$MODEL" \
        "$HOME/.cache/turboquant/$MODEL" \
        "$HOME/$MODEL"; do
        if [ -f "$candidate" ]; then
            MODEL="$candidate"
            break
        fi
    done
fi

if [ ! -f "$MODEL" ]; then
    echo "WARNING: Model file '$MODEL' not found."
    echo "Running in estimation-only mode (no actual inference)."
    echo ""
    ESTIMATION_ONLY=1
else
    ESTIMATION_ONLY=0
    echo "Model: $MODEL"
fi

echo "KV type: $KV_TYPE"
echo ""

# --------------------------------------------------------
# Get model config (if model available)
# --------------------------------------------------------
N_LAYERS=0
N_KV_HEADS=0
HEAD_DIM=0

if [ "$ESTIMATION_ONLY" -eq 0 ]; then
    # Extract model config from --info output
    # Format: "Model: 34 layers, dim=2560, heads=32/4, vocab=262144, inter=6912"
    INFO=$("$TQ_RUN" "$MODEL" --info 2>&1 || true)

    # macOS-compatible parsing using sed (no grep -P)
    # Format: "Model: 34 layers, dim=2560, heads=32/4, head_dim=256, vocab=262144, inter=6912"
    N_LAYERS=$(echo "$INFO" | sed -n 's/^.*Model: \([0-9]*\) layers.*/\1/p' | head -1)
    N_KV_HEADS=$(echo "$INFO" | sed -n 's/^.*heads=[0-9]*\/\([0-9]*\).*/\1/p' | head -1)
    HEAD_DIM=$(echo "$INFO" | sed -n 's/^.*head_dim=\([0-9]*\).*/\1/p' | head -1)
fi

# Fallback to Gemma 3 4B defaults if parsing failed
if [ -z "$N_LAYERS" ] || [ "$N_LAYERS" -eq 0 ]; then
    N_LAYERS=34
fi
if [ -z "$N_KV_HEADS" ] || [ "$N_KV_HEADS" -eq 0 ]; then
    N_KV_HEADS=4
fi
if [ -z "$HEAD_DIM" ] || [ "$HEAD_DIM" -eq 0 ]; then
    HEAD_DIM=256
fi

echo "Model config: ${N_LAYERS} layers, ${N_KV_HEADS} kv_heads, head_dim=${HEAD_DIM}"
echo ""

# --------------------------------------------------------
# Calculate FP16 and compressed KV sizes
# --------------------------------------------------------

# FP16 baseline (llama.cpp default): K_fp16 + V_fp16
# = 2 (K+V) * n_layers * n_kv_heads * head_dim * 2 bytes_per_fp16
FP16_PER_TOKEN=$(( 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * 2 ))

# Quantized block parameters
BLOCK_SIZE=128
case "$KV_TYPE" in
    uniform_4b) TYPE_SIZE=68 ;;   # 4 + 128/2
    uniform_2b) TYPE_SIZE=36 ;;   # 4 + 128/4
    polar_3b)   TYPE_SIZE=72 ;;   # 8 + 128/2
    polar_4b)   TYPE_SIZE=72 ;;   # 8 + 128/2
    turbo_3b)   TYPE_SIZE=112 ;;  # polar(72) + qjl(40)
    turbo_4b)   TYPE_SIZE=112 ;;  # polar(72) + qjl(40)
    mixed_4b8)  TYPE_SIZE=80 ;;   # 4 + 4 + 8 + 128/2
    fp32)       TYPE_SIZE=0 ;;
    *)          TYPE_SIZE=68 ;;   # default to uniform_4b
esac

BLOCKS_PER_HEAD=$(( (HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE ))

# quant.cpp Q4: both keys AND values quantized to same type
# = 2 (K+V) * n_layers * n_kv_heads * blocks_per_head * type_size
if [ "$KV_TYPE" = "fp32" ]; then
    Q4_PER_TOKEN=$(( 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * 4 ))
else
    Q4_PER_TOKEN=$(( 2 * N_LAYERS * N_KV_HEADS * BLOCKS_PER_HEAD * TYPE_SIZE ))
fi

echo "Per-token KV (FP16 / llama.cpp):  $(echo "scale=2; $FP16_PER_TOKEN / 1024" | bc) KB"
echo "Per-token KV (Q4 / quant.cpp):   $(echo "scale=2; $Q4_PER_TOKEN / 1024" | bc) KB"
OVERALL_RATIO=$(echo "scale=2; $FP16_PER_TOKEN / $Q4_PER_TOKEN" | bc)
echo "Compression ratio:                ${OVERALL_RATIO}x"
echo ""

# --------------------------------------------------------
# Generate long prompts and run benchmarks
# --------------------------------------------------------

# Create a repeatable text block (~100 tokens per repetition)
FILLER="The quick brown fox jumps over the lazy dog. In the vast expanse of the universe, countless stars illuminate the darkness of space. Knowledge is the foundation upon which all great achievements are built. Every journey of a thousand miles begins with a single step forward. "

generate_prompt() {
    local target_tokens=$1
    # Rough estimate: ~1.3 tokens per word, ~4 chars per word
    local target_words=$(( target_tokens * 3 / 4 ))
    local result=""
    while [ ${#result} -lt $(( target_words * 5 )) ]; do
        result="${result}${FILLER}"
    done
    # Truncate to approximate target
    echo "${result:0:$(( target_words * 5 ))}"
}

# --------------------------------------------------------
# Write CSV header
# --------------------------------------------------------
echo "context_length,compressed_kv_bytes,fp16_kv_bytes,compressed_kv_mb,fp16_kv_mb,compression_ratio,memory_saved_mb" > "$CSV_OUT"

# Column labels
COL_Q4="Q4 (quant.cpp)"
COL_FP16="FP16 (llama.cpp)"

# --------------------------------------------------------
# Print table header
# --------------------------------------------------------
printf "\n%-15s  %18s  %18s  %8s  %15s\n" \
    "Context Length" "Q4 quant.cpp" "FP16 llama.cpp" "Ratio" "Memory Saved"
printf "%-15s  %18s  %18s  %8s  %15s\n" \
    "---------------" "------------------" "------------------" "--------" "---------------"

# --------------------------------------------------------
# Run benchmark at each context length
# --------------------------------------------------------
for CTX in "${CONTEXT_LENGTHS[@]}"; do
    TOTAL_Q4=$(( Q4_PER_TOKEN * CTX ))
    TOTAL_FP16=$(( FP16_PER_TOKEN * CTX ))
    Q4_MB=$(echo "scale=2; $TOTAL_Q4 / 1048576" | bc)
    FP16_MB=$(echo "scale=2; $TOTAL_FP16 / 1048576" | bc)
    RATIO=$(echo "scale=2; $TOTAL_FP16 / $TOTAL_Q4" | bc)
    SAVED_MB=$(echo "scale=2; ($TOTAL_FP16 - $TOTAL_Q4) / 1048576" | bc)

    # If model is available, also run actual inference to validate
    if [ "$ESTIMATION_ONLY" -eq 0 ] && [ "$CTX" -le 512 ]; then
        # Only run actual inference for smaller contexts (larger ones take too long)
        PROMPT=$(generate_prompt "$CTX")
        GEN_TOKENS=$(( CTX / 4 ))  # Generate 1/4 of context length
        STDERR_OUT=$("$TQ_RUN" "$MODEL" -p "$PROMPT" -n "$GEN_TOKENS" -k "$KV_TYPE" -M -q q4 2>&1 >/dev/null || true)
        # Extract actual MEMORY_CSV line if available
        ACTUAL_LINE=$(echo "$STDERR_OUT" | grep "^MEMORY_CSV:" || true)
        if [ -n "$ACTUAL_LINE" ]; then
            ACTUAL_RATIO=$(echo "$ACTUAL_LINE" | cut -d, -f4)
            printf "%-15s  %18s  %18s  %8s  %15s  (actual: %.2fx)\n" \
                "$CTX tokens" "${Q4_MB} MB" "${FP16_MB} MB" "${RATIO}x" "${SAVED_MB} MB" "$ACTUAL_RATIO"
        else
            printf "%-15s  %15s  %15s  %12s  %15s\n" \
                "$CTX tokens" "${Q4_MB} MB" "${FP16_MB} MB" "${RATIO}x" "${SAVED_MB} MB"
        fi
    else
        printf "%-15s  %18s  %18s  %8s  %15s\n" \
            "$CTX tokens" "${Q4_MB} MB" "${FP16_MB} MB" "${RATIO}x" "${SAVED_MB} MB"
    fi

    # Write CSV row
    echo "$CTX,$TOTAL_Q4,$TOTAL_FP16,$Q4_MB,$FP16_MB,$RATIO,$SAVED_MB" >> "$CSV_OUT"
done

# --------------------------------------------------------
# Extended context lengths (estimation only)
# --------------------------------------------------------
EXTENDED_LENGTHS=(8192 16384 32768 65536 131072)

printf "\n%-15s  %18s  %18s  %8s  %15s\n" \
    "--- Extended ---" "" "" "" ""

for CTX in "${EXTENDED_LENGTHS[@]}"; do
    TOTAL_Q4=$(( Q4_PER_TOKEN * CTX ))
    TOTAL_FP16=$(( FP16_PER_TOKEN * CTX ))
    Q4_MB=$(echo "scale=2; $TOTAL_Q4 / 1048576" | bc)
    FP16_MB=$(echo "scale=2; $TOTAL_FP16 / 1048576" | bc)
    RATIO=$(echo "scale=2; $TOTAL_FP16 / $TOTAL_Q4" | bc)
    SAVED_MB=$(echo "scale=2; ($TOTAL_FP16 - $TOTAL_Q4) / 1048576" | bc)

    printf "%-15s  %15s  %15s  %12s  %15s\n" \
        "$CTX tokens" "${Q4_MB} MB" "${FP16_MB} MB" "${RATIO}x" "${SAVED_MB} MB"

    echo "$CTX,$TOTAL_Q4,$TOTAL_FP16,$Q4_MB,$FP16_MB,$RATIO,$SAVED_MB" >> "$CSV_OUT"
done

echo ""
echo "CSV results saved to: $CSV_OUT"
echo ""
echo "To generate a chart:"
echo "  python3 bench/plot_memory.py bench/long_context_results.csv"
