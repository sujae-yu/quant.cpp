#!/bin/bash
# tools/coh_bench.sh — Coherent-length benchmark
#
# For each model, runs 3 standardized prompts at -T 0 and reports:
#   - tokens generated until natural EOS or repetition-stop
#   - quality verdict (first ~500 chars of output for human judgment)
#
# Complements basin_compat.sh which measures numerical divergence.
# This measures actual output quality — the user-facing tier metric.
#
# Usage:
#   tools/coh_bench.sh models/<model>.gguf [models/<model2>.gguf ...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PROMPTS=(
  "Explain quantum mechanics in simple terms with examples."
  "Write a short poem about a solitary lighthouse at dawn."
  "What is the capital of France, and name two famous landmarks there?"
)
PROMPT_LABELS=("quantum" "poem" "trivia")

echo "== coh_bench =="
echo

for MODEL in "$@"; do
  if [ ! -f "$MODEL" ]; then
    echo "skip: $MODEL not found"
    continue
  fi
  name=$(basename "$MODEL" .gguf)
  # Only pass --chat for instruct-tuned models (simple heuristic: has "Instruct" or "chat" or "it" in name)
  chat_flag=""
  if echo "$name" | grep -qiE "instruct|chat|-it-|A3B|Q4_K_M|Q8_0|Q5_K_M"; then
    chat_flag="--chat"
  fi

  echo "### $name"
  for i in 0 1 2; do
    label=${PROMPT_LABELS[$i]}
    prompt=${PROMPTS[$i]}
    # Use -n 300 (reasonable ceiling for "coh length" testing)
    out=$(TQ_ENABLE_THINKING=1 "$ROOT/build/quant" "$MODEL" \
        -p "$prompt" -n 300 -T 0 -j 1 $chat_flag 2>&1 || true)
    tokens=$(echo "$out" | grep -oE "decode [0-9]+ tok" | head -1 | grep -oE "[0-9]+" || echo "?")
    stopped=$(echo "$out" | grep -oE "repetition loop detected|generate.*stopping" | head -1 || echo "natural EOS")
    preview=$(echo "$out" | awk '/^\[tokenizer\]/{flag=1;next} /^---$/{flag=0} flag' | tr '\n' ' ' | head -c 200)
    printf "  [%-10s] tok=%s  stop=%-35s\n    %s\n" "$label" "$tokens" "$stopped" "$preview"
  done
  echo
done
