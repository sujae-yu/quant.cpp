#!/bin/bash
# tools/basin_compat.sh — Engine FP32 basin compatibility benchmark
#
# Runs identical prompt through our engine (TQ_LAYER_TRACE) and llama-debug
# (--verbose --tensor-filter ^l_out-) on the same GGUF model. Reports
# per-layer residual-sum divergence and assigns a Tier classification.
#
# Tier 1 (Production): all layers rel_diff < 5%
# Tier 2 (Research grade): late layers 10-40% rel_diff
# Tier 3 (Needs research): early or persistent >50%
#
# See docs/engine_basin_tiers.md for the theory.
#
# CAVEAT: this tool is designed for hybrid DeltaNet/self-attn MoE models
# (like Qwen3.6-A3B) where llama-debug emits per-layer N=1 decode dumps.
# For pure feedforward models (Llama, Phi, Gemma), llama-debug only dumps
# N=1 on the FINAL layer (due to ggml's get_rows optimization), so
# per-layer comparison is limited. Use paired-diff alternative tools for
# those architectures (see docs/custom-quantization.md).
#
# Usage:
#   tools/basin_compat.sh models/<model>.gguf
#   tools/basin_compat.sh models/<model>.gguf "Hello"      # custom prompt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?usage: basin_compat.sh <model.gguf> [prompt]}"
PROMPT="${2:-Hello}"
OUT="${OUT:-/tmp/basin_compat}"

mkdir -p "$OUT"
name=$(basename "$MODEL" .gguf)
echo "== basin compatibility =="
echo "model:  $name"
echo "prompt: $PROMPT"
echo

echo "→ running ours (TQ_LAYER_TRACE)..."
TQ_LAYER_TRACE=1 \
  "$ROOT/build/quant" "$MODEL" \
    -p "$PROMPT" -n 1 -T 0 -j 1 \
  2>"$OUT/$name.ours.stderr" >/dev/null

echo "→ running llama-debug (--tensor-filter ^l_out-)..."
pkill -9 -f "llama-debug" 2>/dev/null || true; sleep 1
"$ROOT/refs/llama.cpp/build/bin/llama-debug" \
  -m "$MODEL" \
  -p "$PROMPT" \
  --verbose --tensor-filter "^l_out-" \
  -n 1 --temp 0 -t 1 --ctx-size 128 \
  --device none -fit off --no-op-offload \
  2>"$OUT/$name.llama.stderr" >"$OUT/$name.llama.stdout"

python3 - <<EOF
import re, sys
ours_lout = {}
for line in open("$OUT/$name.ours.stderr"):
    m = re.match(r'\[trace\] l_out-(\d+) pos=(\d+) sum=([\-\d\.]+)', line)
    if m: ours_lout.setdefault(int(m.group(1)), {})[int(m.group(2))] = float(m.group(3))

if not ours_lout:
    print("error: no layer trace from ours — is TQ_LAYER_TRACE supported on this model?")
    sys.exit(1)

positions = sorted({p for v in ours_lout.values() for p in v})
pos_use = positions[0]

llama_de = {}
cur, N = None, None
for line in open("$OUT/$name.llama.stdout"):
    m = re.match(r'common_debug_cb_eval:\s+l_out-(\d+) = \(f32\)\s+(?:ADD|DUP|VIEW)\([^{]+\{[^,]+, (\d+)', line)
    if m: cur = int(m.group(1)); N = int(m.group(2)); continue
    ms = re.match(r'\s+sum\s*=\s*([\-\d\.]+)', line)
    if ms and cur is not None and N == 1:
        llama_de[cur] = float(ms.group(1)); cur = None

if not llama_de:
    print("error: no layer dump from llama-debug")
    sys.exit(1)

n_layers = max(max(ours_lout.keys()), max(llama_de.keys())) + 1
layer_diffs = []
max_rel = 0.0
tier1_threshold = 0.05  # 5%
tier2_threshold = 0.50  # 50%

for L in range(n_layers):
    ov = ours_lout.get(L, {}).get(pos_use); ld = llama_de.get(L)
    if ov is None or ld is None: continue
    rd = abs(ov - ld) / max(abs(ld), 1e-6)
    layer_diffs.append((L, ov, ld, rd))
    if rd > max_rel: max_rel = rd

print(f"{'Layer':>5}  {'ours':>12}  {'llama':>12}  {'rel_diff':>10}")
for L, ov, ld, rd in layer_diffs:
    mark = "**" if rd > tier1_threshold else ""
    print(f"{L:>5}  {ov:>12.4f}  {ld:>12.4f}  {rd:>10.4f}  {mark}")

# Classification
late_max = max((rd for L, _, _, rd in layer_diffs if L >= n_layers - 5), default=0.0)
early_max = max((rd for L, _, _, rd in layer_diffs if L < n_layers // 2), default=0.0)

print(f"\n== Summary ==")
print(f"layers measured: {len(layer_diffs)} / {n_layers}")
print(f"max rel_diff overall: {max_rel:.4f}")
print(f"max rel_diff early (L < {n_layers//2}): {early_max:.4f}")
print(f"max rel_diff late (last 5 layers): {late_max:.4f}")

if max_rel < tier1_threshold:
    tier = "Tier 1 — Production quality (all layers within 5%)"
elif late_max < tier2_threshold and early_max < 0.20:
    tier = "Tier 2 — Research grade (late-layer drift, early layers stable)"
else:
    tier = "Tier 3 — Needs research (early or persistent divergence)"

print(f"\n=> {tier}")
EOF
