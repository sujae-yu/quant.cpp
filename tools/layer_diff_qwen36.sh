#!/bin/bash
# tools/layer_diff_qwen36.sh — Per-layer l_out sum diff between ours and llama.cpp
# on Qwen3.6-35B-A3B UD-IQ4_XS.
#
# Runs both engines on the same raw prompt and compares the per-layer
# residual output sum. Finds the first layer where ours diverges from
# llama materially (see R63 memory: divergence localized to L33-L37).
#
# Usage: ./tools/layer_diff_qwen36.sh [PROMPT]  (default: "Hello")

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROMPT="${1:-Hello}"
MODEL="${MODEL:-$ROOT/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
OUT="${OUT:-/tmp/layer_diff}"

mkdir -p "$OUT"
echo "prompt: $PROMPT"
echo "model: $MODEL"
echo "out: $OUT"

echo "=== running ours ==="
TQ_LAYER_TRACE=1 \
  TQ_QWEN35MOE_NO_PRESET=1 TQ_NO_MOE_TEMP_AUTO=1 TQ_MOE_ROUTE_TEMP=1.0 \
  "$ROOT/build/quant" "$MODEL" \
    -p "$PROMPT" -n 1 -T 0 -j 1 \
  2>"$OUT/ours.stderr" >"$OUT/ours.stdout"

echo "=== running llama-debug ==="
# Kill any previous instances
pkill -9 -f "llama-debug" 2>/dev/null || true
sleep 1
"$ROOT/refs/llama.cpp/build/bin/llama-debug" \
  -m "$MODEL" \
  -p "$PROMPT" \
  --verbose --tensor-filter "^l_out-" \
  -n 1 --temp 0 -t 1 --ctx-size 128 \
  --device none -fit off --no-op-offload \
  2>"$OUT/llama.stderr" >"$OUT/llama.stdout"

echo "=== per-layer diff ==="
python3 - <<EOF
import re

ours = {}
for line in open("$OUT/ours.stderr"):
    m = re.match(r'\[trace\] l_out-(\d+) pos=(\d+) sum=([\-\d\.]+)', line)
    if m:
        L, P, S = int(m.group(1)), int(m.group(2)), float(m.group(3))
        ours.setdefault(L, {})[P] = S

positions = sorted({p for v in ours.values() for p in v})
use_pos = positions[0]

llama_de = [None]*40
cur = None
N = None
for line in open("$OUT/llama.stdout"):
    m = re.match(r'common_debug_cb_eval:\s+l_out-(\d+) = \(f32\)\s+ADD\([^{]+\{2048, (\d+)', line)
    if m:
        cur = int(m.group(1)); N = int(m.group(2)); continue
    m = re.match(r'\s+sum\s*=\s*([\-\d\.]+)', line)
    if m and cur is not None:
        if N == 1 and cur < 40:
            llama_de[cur] = float(m.group(1))
        cur = None

print(f"positions seen in ours: {positions}, using pos={use_pos}")
print(f"llama decode layers: {sum(1 for x in llama_de if x is not None)}/40")
print()
print(f"{'L':>3} {'ours':>12} {'llama':>12} {'abs_diff':>10} {'rel_diff':>10}  mark")
for L in range(40):
    ov = ours.get(L, {}).get(use_pos)
    ld = llama_de[L]
    if ov is None or ld is None:
        print(f"{L:>3} {'-':>12} {ld if ld is not None else '-':>12} {'-':>10} {'-':>10}")
        continue
    ad = abs(ov - ld); rd = ad / max(abs(ld), 1e-6)
    mark = "**" if rd > 0.10 else ""
    print(f"{L:>3} {ov:>12.4f} {ld:>12.4f} {ad:>10.4f} {rd:>10.4f}  {mark}")
EOF
