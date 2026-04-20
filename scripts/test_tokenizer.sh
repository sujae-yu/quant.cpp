#!/usr/bin/env bash
# Tokenizer-level regression: verifies our BPE output matches HF-known-good
# token IDs for prompts that previously triggered the R3 BPE stale-entry bug.
#
# If this test fails, the heap-based BPE merge loop has likely regressed
# (e.g., someone removed the `tokens[top.pos] < 0` staleness check at
# tq_tokenizer.c:1442). See bench/results/2026-04-20_bpe_fix_proof.md.

set -u
BIN="${BIN:-./build/quant}"
MODELS_DIR="${MODELS_DIR:-models}"
PASS=0
FAIL=0
SKIP=0

check_tokens() {
    local model="$1"
    local prompt="$2"
    local expected="$3"  # space-separated token IDs
    local extra_env="${4:-}"
    if [[ ! -f "$MODELS_DIR/$model" ]]; then
        printf "  %-40s [SKIP] not found\n" "$model"
        SKIP=$((SKIP+1))
        return
    fi
    local got
    got=$(env $extra_env TQ_DEBUG=1 "$BIN" "$MODELS_DIR/$model" -p "$prompt" -n 1 -T 0 2>&1 \
          | grep 'prompt tokens' | sed -E 's/.*prompt tokens \([0-9]+\): //' \
          | awk '{$1=$1; print}')  # normalize whitespace
    if [[ "$got" == "$expected"* ]]; then
        printf "  %-40s [PASS] %-20s → %s\n" "$model" "\"$prompt\"" "$got"
        PASS=$((PASS+1))
    else
        printf "  %-40s [FAIL] \"%s\" expected:%s got:%s\n" "$model" "$prompt" "$expected" "$got"
        FAIL=$((FAIL+1))
    fi
}

# Qwen3.5/3.6 share a 248320-token vocab different from Qwen3-0.6B's 151936.
# We only assert the structural property: "Hello" merges to a SINGLE token
# (not the pre-R3 broken pair [Hel, ll]=two tokens that decoded to "Helll").
check_tokens_single() {
    local model="$1"
    local prompt="$2"
    local extra_env="${3:-}"
    if [[ ! -f "$MODELS_DIR/$model" ]]; then
        printf "  %-40s [SKIP] not found\n" "$model"
        SKIP=$((SKIP+1))
        return
    fi
    local got
    got=$(env $extra_env TQ_DEBUG=1 "$BIN" "$MODELS_DIR/$model" -p "$prompt" -n 1 -T 0 2>&1 \
          | grep 'prompt tokens' | sed -E 's/.*prompt tokens \(([0-9]+)\).*/\1/')
    if [[ "$got" == "1" ]]; then
        printf "  %-40s [PASS] \"%s\" → 1 token (merged)\n" "$model" "$prompt"
        PASS=$((PASS+1))
    else
        printf "  %-40s [FAIL] \"%s\" → %s tokens (expected 1)\n" "$model" "$prompt" "$got"
        FAIL=$((FAIL+1))
    fi
}

echo "=== Tokenizer regression (BPE stale-entry guard — Pillar 1 R3) ==="
echo "Models dir: $MODELS_DIR"
echo ""

# Qwen3 family — the originally broken path.
# Expected token IDs verified against HF AutoTokenizer 2026-04-20.
check_tokens "Qwen3-0.6B-Q4_K_M.gguf"            "Hello"           "9707" \
  "TQ_NO_METAL=1 TQ_NO_MLOCK=1"
check_tokens "Qwen3-0.6B-Q4_K_M.gguf"            "The quick brown fox" "785 3974 13876 38835" \
  "TQ_NO_METAL=1 TQ_NO_MLOCK=1"
check_tokens_single "Qwen3.5-4B-Q4_K_M.gguf"     "Hello" \
  "TQ_NO_METAL=1 TQ_NO_MLOCK=1"
check_tokens_single "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" "Hello" \
  "TQ_NO_METAL=1 TQ_NO_MLOCK=1"

echo ""
echo "--- Summary ---  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
[[ "$FAIL" -eq 0 ]] || exit 1
exit 0
