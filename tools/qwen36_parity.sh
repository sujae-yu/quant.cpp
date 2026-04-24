#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
QUANT_BIN="${QUANT_BIN:-$ROOT_DIR/build/quant}"
LLAMA_DEBUG_BIN="${LLAMA_DEBUG_BIN:-$ROOT_DIR/refs/llama.cpp/build/bin/llama-debug}"
MODEL="${MODEL:-$ROOT_DIR/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
PROMPT="${PROMPT:-Explain quantum mechanics in simple terms with examples.}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data/qwen36_parity/$(date +%Y%m%d_%H%M%S)}"
N_TOKENS="${N_TOKENS:-8}"
THREADS="${THREADS:-1}"
CTX_SIZE="${CTX_SIZE:-4096}"
ENABLE_DUMPS="${ENABLE_DUMPS:-0}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --model PATH        GGUF model path
  --prompt TEXT       Prompt to compare
  --out-dir PATH      Output directory
  --n N               Generated tokens (default: ${N_TOKENS})
  --threads N         CPU threads (default: ${THREADS})
  --ctx N             Context size (default: ${CTX_SIZE})
  --dump-hidden       Enable TQ_DUMP_HIDDEN/TQ_DUMP_INTERMEDIATE for quant.cpp
  --help              Show this help

Environment overrides:
  QUANT_BIN, LLAMA_DEBUG_BIN, MODEL, PROMPT, OUT_DIR, N_TOKENS, THREADS, CTX_SIZE
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --n) N_TOKENS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --ctx) CTX_SIZE="$2"; shift 2 ;;
        --dump-hidden) ENABLE_DUMPS=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

if [ ! -x "$QUANT_BIN" ]; then
    echo "quant binary not found: $QUANT_BIN" >&2
    exit 1
fi

write_meta() {
    cat > "$OUT_DIR/meta.txt" <<EOF
root_dir=$ROOT_DIR
quant_bin=$QUANT_BIN
llama_debug_bin=$LLAMA_DEBUG_BIN
model=$MODEL
prompt=$PROMPT
n_tokens=$N_TOKENS
threads=$THREADS
ctx_size=$CTX_SIZE
enable_dumps=$ENABLE_DUMPS
EOF
}

run_quant_case() {
    local name="$1"
    shift

    local case_dir="$OUT_DIR/$name"
    mkdir -p "$case_dir"

    local dump_env=()
    if [ "$ENABLE_DUMPS" = "1" ]; then
        mkdir -p "$case_dir/dumps"
        dump_env=(
            TQ_DUMP_HIDDEN="$case_dir/dumps"
            TQ_DUMP_INTERMEDIATE=1
            TQ_DUMP_POS=all
        )
    fi

    if [ "${#dump_env[@]}" -gt 0 ]; then
        env \
            TQ_NO_METAL=1 \
            TQ_NO_MLOCK=1 \
            TQ_QWEN35MOE_NO_PRESET=1 \
            TQ_NO_MOE_TEMP_AUTO=1 \
            TQ_LOGIT_PROBE=every=1 \
            "${dump_env[@]}" \
            "$@" \
            "$QUANT_BIN" "$MODEL" \
            -p "$PROMPT" \
            -n "$N_TOKENS" \
            -T 0 \
            -j "$THREADS" \
            --ctx "$CTX_SIZE" \
            > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt"
    else
        env \
            TQ_NO_METAL=1 \
            TQ_NO_MLOCK=1 \
            TQ_QWEN35MOE_NO_PRESET=1 \
            TQ_NO_MOE_TEMP_AUTO=1 \
            TQ_LOGIT_PROBE=every=1 \
            "$@" \
            "$QUANT_BIN" "$MODEL" \
            -p "$PROMPT" \
            -n "$N_TOKENS" \
            -T 0 \
            -j "$THREADS" \
            --ctx "$CTX_SIZE" \
            > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt"
    fi

    grep '\[logit-probe\]' "$case_dir/stderr.txt" > "$case_dir/logit_probe.txt" || true
}

run_llama_debug() {
    local case_dir="$OUT_DIR/llama_debug"
    mkdir -p "$case_dir"

    if [ ! -x "$LLAMA_DEBUG_BIN" ]; then
        echo "llama-debug binary not found: $LLAMA_DEBUG_BIN" > "$case_dir/error.txt"
        return 0
    fi

    "$LLAMA_DEBUG_BIN" \
        -m "$MODEL" \
        -p "$PROMPT" \
        -n "$N_TOKENS" \
        -c "$CTX_SIZE" \
        --temp 0 \
        --threads "$THREADS" \
        --device none \
        --fit off \
        --no-op-offload \
        --save-logits \
        --logits-output-dir "$case_dir" \
        > "$case_dir/stdout.txt" 2> "$case_dir/stderr.txt" || true
}

write_meta
run_quant_case baseline
run_quant_case llama_kernels TQ_USE_LLAMA_KERNELS=1
run_llama_debug

cat > "$OUT_DIR/README.txt" <<EOF
Produced cases:
  baseline/       quant.cpp CPU-only, no qwen35moe auto-preset, greedy, TQ_LOGIT_PROBE=every=1
  llama_kernels/  same as baseline, plus TQ_USE_LLAMA_KERNELS=1
  llama_debug/    llama.cpp CPU-only debug run with --save-logits

Suggested next checks:
  diff -u "$OUT_DIR/baseline/logit_probe.txt" "$OUT_DIR/llama_kernels/logit_probe.txt"
  sed -n '1,80p' "$OUT_DIR"/baseline/stdout.txt
  sed -n '1,80p' "$OUT_DIR"/llama_kernels/stdout.txt
  sed -n '1,80p' "$OUT_DIR"/llama_debug/stderr.txt
  python3 "$ROOT_DIR/tools/analyze_qwen36_parity.py" "$OUT_DIR"   # best with --dump-hidden
EOF

echo "$OUT_DIR"
