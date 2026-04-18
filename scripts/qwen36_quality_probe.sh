#!/bin/bash
# Qwen3.6-35B-A3B quality probe — IQ2_XXS vs IQ3_XXS vs any future variant.
#
# Measures:
#   (1) Factual Q&A coherence — 10 questions, greedy T=0, short answer
#   (2) 100-token greedy coherence — single long decode, drift detection
#   (3) Multi-turn chat — 3 turns
#
# Usage: bash scripts/qwen36_quality_probe.sh <path-to-gguf> [jobs]
# Exit codes: 0 = run completed; inspect stdout for per-probe output.
set -u
MODEL="${1:?usage: $0 <gguf> [jobs]}"
J="${2:-8}"
BIN="./build/quant"

if [[ ! -x "$BIN" ]]; then echo "build $BIN first" >&2; exit 2; fi
if [[ ! -f "$MODEL" ]]; then echo "model $MODEL not found" >&2; exit 2; fi

ENV="TQ_NO_METAL=1 TQ_NO_MLOCK=1"
CHAT_ARGS="--chat -n 12 -T 0 -j $J"
LONG_ARGS="-n 100 -T 0 -j $J"

name=$(basename "$MODEL")
echo "=== quality probe — $name ==="
echo

# ----------------------------------------------------------------------
# (1) Factual Q&A — short greedy, 10 prompts
# ----------------------------------------------------------------------
echo "--- factual Q&A (greedy, 12 tokens, expected substring check) ---"
FACT_PASS=0
FACT_TOTAL=0
run_fact() {
    local q="$1"
    local expected="$2"
    FACT_TOTAL=$((FACT_TOTAL+1))
    local out
    out=$(env $ENV "$BIN" "$MODEL" $CHAT_ARGS -p "$q" 2>/dev/null \
        | tr '\n' ' ' | sed 's/  */ /g')
    # Strip everything before the "---" separator CLI prints before gen.
    local ans="${out##*---}"
    if [[ "$ans" == *"$expected"* ]]; then
        FACT_PASS=$((FACT_PASS+1))
        printf "  PASS  %-50s → '%s'\n" "$q" "${ans:0:60}"
    else
        printf "  FAIL  %-50s → '%s'  (wanted '%s')\n" "$q" "${ans:0:60}" "$expected"
    fi
}

run_fact "The capital of France is"                 "Paris"
run_fact "2 plus 2 equals"                           "4"
run_fact "The largest planet in our solar system is" "Jupiter"
run_fact "Water boils at 100 degrees"                "Celsius"
run_fact "The author of Hamlet is"                   "Shakespeare"
run_fact "The chemical symbol for gold is"           "Au"
run_fact "The first president of the United States was" "Washington"
run_fact "The fastest land animal is the"            "cheetah"
run_fact "The speed of light is approximately 300,000" "kilometer"
run_fact "The currency of Japan is the"              "yen"

echo
echo "  FACTUAL: $FACT_PASS / $FACT_TOTAL"
echo

# ----------------------------------------------------------------------
# (2) 100-token coherence — single long decode
# ----------------------------------------------------------------------
echo "--- 100-token coherence (greedy, raw, no chat template) ---"
out=$(env $ENV "$BIN" "$MODEL" $LONG_ARGS -p "Once upon a time" 2>/dev/null \
    | awk '/^---/{flag++;next} flag==1' | tr '\n' ' ' | sed 's/  */ /g')
# Simple coherence heuristic:
#   - non-empty
#   - printable ratio > 90%
#   - contains at least 4 distinct English words
tokens=$(echo "$out" | wc -w | tr -d ' ')
printable=$(echo "$out" | tr -cd '[:print:][:space:]' | wc -c | tr -d ' ')
total=$(echo -n "$out" | wc -c | tr -d ' ')
if [[ "$total" -gt 0 ]]; then
    pct=$((printable * 100 / total))
else
    pct=0
fi
echo "  tokens: $tokens, printable: ${pct}%"
echo "  text: ${out:0:200}..."
echo

# ----------------------------------------------------------------------
# (3) Multi-turn chat — 3 conversation probes
# ----------------------------------------------------------------------
echo "--- multi-turn chat coherence (3 probes) ---"
for prompt in "Hi, what's your name?" "Explain photosynthesis in one sentence." "Write a short poem about the sea."; do
    out=$(env $ENV "$BIN" "$MODEL" --chat -n 40 -T 0 -j $J -p "$prompt" 2>/dev/null \
        | awk '/^---/{flag++;next} flag==1' | tr '\n' ' ' | sed 's/  */ /g')
    printf "  [%s]\n    → %s\n\n" "$prompt" "${out:0:180}"
done

echo "=== done ==="
