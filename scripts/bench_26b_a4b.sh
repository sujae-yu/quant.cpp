#!/usr/bin/env bash
# bench_26b_a4b.sh — benchmark gemma-4-26B-A4B-it MoE loading, decode, prefill,
# and memory footprint on quant.cpp.
#
# Usage: bash scripts/bench_26b_a4b.sh

set -u
MODEL="${MODEL:-models/google_gemma-4-26B-A4B-it-IQ2_XXS.gguf}"
Q="${Q:-./build/quant}"

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found at $MODEL" >&2
    echo "Set MODEL=<path> to override." >&2
    exit 1
fi

if [[ ! -x "$Q" ]]; then
    echo "ERROR: $Q not built" >&2
    exit 1
fi

echo "=================================================================="
echo "gemma-4-26B-A4B-it MoE benchmark"
echo "Model: $MODEL"
echo "=================================================================="

echo ""
echo "── Load + model info ──"
"$Q" "$MODEL" --info 2>&1 | grep -E "(architecture|config|layers|vocab|Q4 conv|MoE|Metal|experts|PLE|sliding|KV share|Fused)" | head -20

echo ""
echo "── Short decode: 'Hi' → 32 tokens (Metal + CPU defaults) ──"
/usr/bin/time -l "$Q" "$MODEL" -p "Hi, I am a language model" -n 32 -T 0 2>&1 | \
    grep -E "(tokens in|maximum resident|Metal|Fused MoE)" | tail -5

echo ""
echo "── Memory: FP16 KV vs turbo_kv_4b + V=q4 ──"
"$Q" "$MODEL" -p "test" -n 8 -T 0 -M 2>&1 | grep -E "(Per-token|Compression|Total K)"

echo ""
echo "── Long prompt decode: 100-word prompt, 16 gen tokens ──"
PROMPT=$(python3 -c "import random; random.seed(42); print(' '.join(random.choice(['the','of','and','to','a','in','is','that','for','with']) for _ in range(100)))")
/usr/bin/time -l "$Q" "$MODEL" -p "$PROMPT" -n 16 -T 0 2>&1 | \
    grep -E "(tokens in|maximum resident)" | tail -3

echo ""
echo "── Chat regression: expected greeting ──"
"$Q" "$MODEL" --chat -p "Hello, how are you?" -n 24 -T 0 2>&1 | \
    tail -3
