#!/bin/bash
# auto_profile.sh — Automatic compression profile generator
#
# Runs the full analysis pipeline:
#   1. --profile-kv: measure KV activation distributions
#   2. --recommend: per-layer bit allocation
#   3. --calibrate: Lloyd-Max codebook optimization
#   4. Output a JSON-like profile with recommendations
#
# Usage: bash bench/auto_profile.sh <model_path> [options]
#
# Options:
#   -t <tokenizer>  Tokenizer path
#   -p <prompt>     Prompt for calibration (default: long test prompt)
#   -n <tokens>     Number of tokens (default: 200)
#   -j <threads>    Thread count (default: 4)
#   -q <quant>      Weight quant: q2, q4, q8, none (default: q4)
#   -o <output>     Output JSON file (default: stdout)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
TQ_RUN="${BUILD_DIR}/quant"

# Defaults
MODEL_PATH=""
TOKENIZER_ARGS=""
PROMPT="The quick brown fox jumps over the lazy dog. In natural language processing, quantization of key-value caches enables efficient inference by reducing memory footprint while maintaining quality."
N_TOKENS=200
N_THREADS=4
QUANT="q4"
OUTPUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t) TOKENIZER_ARGS="-t $2"; shift 2 ;;
        -p) PROMPT="$2"; shift 2 ;;
        -n) N_TOKENS="$2"; shift 2 ;;
        -j) N_THREADS="$2"; shift 2 ;;
        -q) QUANT="$2"; shift 2 ;;
        -o) OUTPUT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 <model_path> [options]"
            echo "  -t <tokenizer>  Tokenizer path"
            echo "  -p <prompt>     Prompt for calibration"
            echo "  -n <tokens>     Number of tokens (default: 200)"
            echo "  -j <threads>    Thread count (default: 4)"
            echo "  -q <quant>      Weight quant: q2, q4, q8, none (default: q4)"
            echo "  -o <output>     Output JSON file (default: stdout)"
            exit 0
            ;;
        *)
            if [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: model path required" >&2
    echo "Usage: $0 <model_path> [options]" >&2
    exit 1
fi

if [ ! -f "$TQ_RUN" ]; then
    echo "Building quant..." >&2
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "$PROJECT_DIR" 2>/dev/null
    cmake --build "$BUILD_DIR" --target quant -j$(sysctl -n hw.ncpu 2>/dev/null || nproc) 2>/dev/null
fi

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== quant.cpp Auto Compression Profile ===" >&2
echo "Model: $MODEL_PATH" >&2
echo "Weight quant: $QUANT, Threads: $N_THREADS" >&2
echo "" >&2

# Step 1: Profile KV distributions + recommend
echo "[1/3] Profiling KV distributions and generating recommendations..." >&2
QUANT_ARGS=""
if [ "$QUANT" != "none" ]; then
    QUANT_ARGS="-q $QUANT"
fi

"$TQ_RUN" "$MODEL_PATH" $TOKENIZER_ARGS $QUANT_ARGS \
    -p "$PROMPT" -n "$N_TOKENS" -j "$N_THREADS" \
    --recommend 2>"$TMPDIR/recommend.log" || true

# Step 2: Calibrate codebook
echo "[2/3] Running codebook calibration..." >&2
"$TQ_RUN" "$MODEL_PATH" $TOKENIZER_ARGS $QUANT_ARGS \
    -p "$PROMPT" -n "$N_TOKENS" -j "$N_THREADS" \
    --calibrate 2>"$TMPDIR/calibrate.log" || true

# Step 3: Get model info
echo "[3/3] Extracting model info..." >&2
"$TQ_RUN" "$MODEL_PATH" $TOKENIZER_ARGS --info 2>"$TMPDIR/info.log" || true

# Parse outputs
echo "" >&2
echo "=== Generating Profile ===" >&2

# Extract model config
N_LAYERS=$(grep -oP '\d+(?= layers)' "$TMPDIR/info.log" 2>/dev/null || echo "unknown")
N_HEADS=$(grep -oP 'heads=\K\d+' "$TMPDIR/info.log" 2>/dev/null || echo "unknown")
N_KV_HEADS=$(grep -oP 'heads=\d+/\K\d+' "$TMPDIR/info.log" 2>/dev/null || echo "unknown")
HEAD_DIM=$(grep -oP 'head_dim=\K\d+' "$TMPDIR/info.log" 2>/dev/null || echo "unknown")

# Extract per-layer recommendations
REC_LAYERS=""
AVG_BITS=$(grep -oP 'Average: \K[0-9.]+' "$TMPDIR/recommend.log" 2>/dev/null || echo "unknown")

# Extract calibrated centroids
CAL_2BIT=$(grep -A1 '2-bit codebook' "$TMPDIR/calibrate.log" 2>/dev/null | grep 'Calibrated' | head -1 || echo "")
CAL_3BIT=$(grep -A1 '3-bit codebook' "$TMPDIR/calibrate.log" 2>/dev/null | grep 'Calibrated' | head -1 || echo "")

# Build JSON profile
generate_json() {
    cat <<JSONEOF
{
  "version": "1.0",
  "model": "$MODEL_PATH",
  "config": {
    "n_layers": $N_LAYERS,
    "n_heads": $N_HEADS,
    "n_kv_heads": $N_KV_HEADS,
    "head_dim": $HEAD_DIM,
    "weight_quant": "$QUANT"
  },
  "recommendations": {
    "average_key_bits": $AVG_BITS,
    "value_quant": "Q4",
    "v_highres_window": 32,
    "notes": [
      "Key compression uses mixed 1-bit/3-bit based on per-layer kurtosis",
      "Value compression at Q4 with FP16 highres window for recent 32 tokens",
      "Layers with kurtosis > 5.0 use 1-bit QJL, others use 3-bit turbo_kv"
    ]
  },
  "memory_estimate": {
JSONEOF

    # Compute memory estimates
    if [ "$N_LAYERS" != "unknown" ] && [ "$N_KV_HEADS" != "unknown" ] && [ "$HEAD_DIM" != "unknown" ]; then
        # FP16 baseline: 2 (K+V) * layers * kv_heads * head_dim * 2 bytes per token
        FP16_PER_TOKEN=$((2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * 2))
        # quant.cpp K (3-bit avg) + V (Q4): approximate
        # K: ~3/8 bytes per element * layers * kv_heads * head_dim
        # V: ~0.5 bytes per element (Q4) * layers * kv_heads * head_dim
        K_BYTES=$(echo "scale=0; $N_LAYERS * $N_KV_HEADS * $HEAD_DIM * 3 / 8" | bc)
        V_BYTES=$(echo "scale=0; $N_LAYERS * $N_KV_HEADS * $HEAD_DIM / 2" | bc)
        TQ_PER_TOKEN=$((K_BYTES + V_BYTES))
        RATIO=$(echo "scale=2; $FP16_PER_TOKEN / $TQ_PER_TOKEN" | bc 2>/dev/null || echo "N/A")

        cat <<MEMEOF
    "fp16_bytes_per_token": $FP16_PER_TOKEN,
    "turbo_bytes_per_token": $TQ_PER_TOKEN,
    "compression_ratio": "${RATIO}x",
    "at_1k_tokens_fp16_mb": $(echo "scale=2; $FP16_PER_TOKEN * 1000 / 1048576" | bc),
    "at_1k_tokens_turbo_mb": $(echo "scale=2; $TQ_PER_TOKEN * 1000 / 1048576" | bc)
MEMEOF
    else
        cat <<MEMEOF
    "note": "Could not compute memory estimates (model info not available)"
MEMEOF
    fi

    cat <<JSONEOF
  },
  "calibration": {
    "tokens_used": $N_TOKENS,
    "prompt_sample": "$(echo "$PROMPT" | head -c 80)..."
  }
}
JSONEOF
}

if [ -n "$OUTPUT" ]; then
    generate_json > "$OUTPUT"
    echo "Profile written to: $OUTPUT" >&2
else
    echo "" >&2
    echo "--- Profile JSON ---"
    generate_json
fi

echo "" >&2
echo "=== Auto Profile Complete ===" >&2
echo "" >&2
echo "Usage:" >&2
echo "  quant model.tqm -k turbo_kv_3b -v q4 -V 32   # Apply recommendations" >&2
echo "  quant model.tqm --bench-prefill                 # Verify prefill speed" >&2
echo "  quant model.tqm --bench-memory                  # Verify decode speed" >&2
