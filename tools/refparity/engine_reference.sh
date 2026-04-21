#!/usr/bin/env bash
# Run quant.cpp engine with TQ_DUMP_HIDDEN to capture per-layer hidden states
# for reference-parity comparison.
#
# Usage:
#   ./engine_reference.sh <model.gguf> "<prompt>" <out_dir>
#
# Output: <out_dir>/emb.bin, h0.bin, h1.bin, ..., post_norm.bin, logits.bin
# Each file is raw FP32 little-endian; shape derived from model config.

set -eu

GGUF="$1"
PROMPT="$2"
OUT_DIR="$3"

BIN="${BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/quant}"

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not executable" >&2
    exit 2
fi
if [[ ! -f "$GGUF" ]]; then
    echo "error: model file not found: $GGUF" >&2
    exit 2
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Force per-token prefill so pos=0 dump captures the first token.
# Use TQ_DUMP_POS=0 (default in tq_dump_hidden).
TQ_NO_METAL=1 TQ_NO_MLOCK=1 TQ_NO_BATCH_PREFILL=1 \
TQ_NO_AUTO_SERIAL=1 \
TQ_DUMP_HIDDEN="$OUT_DIR" \
"$BIN" "$GGUF" -p "$PROMPT" -n 1 -T 0 >"$OUT_DIR/engine.log" 2>&1 || {
    echo "error: engine run failed; see $OUT_DIR/engine.log" >&2
    exit 2
}

# Verify dumps were produced
if ! ls "$OUT_DIR"/h0.bin >/dev/null 2>&1; then
    echo "error: no h0.bin in $OUT_DIR — dump did not fire" >&2
    exit 2
fi

echo "[refparity/engine] dumped to $OUT_DIR/ ($(ls "$OUT_DIR"/*.bin | wc -l | tr -d ' ') files)" >&2
