#!/usr/bin/env bash
# test_long_seq.sh — autoregressive stress test.
#
# Why: PPL (teacher-forced) can be fine while T=0 generation collapses
# after a few hundred tokens — a failure mode KV compression bugs
# typically produce. This test generates 500 tokens at T=0 and rejects
# runs where printable chars fall below 80% of the output (indicating
# repetition-trap garbage, NaN-spew, or token-ID soup).
#
# Complements test_models.sh (which tests first 10 tokens coherence).

set -u
MODELS_DIR="${1:-models}"
QUANT_BIN="${QUANT_BIN:-./build/quant}"
N_TOKENS=500
PASS=0
FAIL=0
SKIP=0
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

if [[ ! -x "$QUANT_BIN" ]]; then
    echo "ERROR: $QUANT_BIN not built." >&2
    exit 1
fi

run_long() {
    local model="$1"
    local prompt="$2"
    local chat_flag="${3:-}"
    local extra_env="${4:-TQ_NO_METAL=1}"

    if [[ ! -f "$MODELS_DIR/$model" ]]; then
        printf "  %-50s [SKIP] not found\n" "$model"
        SKIP=$((SKIP + 1))
        return
    fi

    env $extra_env "$QUANT_BIN" "$MODELS_DIR/$model" $chat_flag \
        -p "$prompt" -n "$N_TOKENS" -T 0 > "$TMP" 2>/dev/null

    local total printable ratio
    total=$(wc -c < "$TMP")
    # Printable = ASCII printable + whitespace + valid UTF-8 multibyte
    # Approximation: chars passing tr -cd '[:print:][:space:]' OR bytes >= 0x80.
    printable=$(tr -cd '[:print:][:space:]\200-\377' < "$TMP" | wc -c)

    if [ "$total" -lt 100 ]; then
        printf "  %-50s [FAIL] too short (%d bytes)\n" "$model" "$total"
        FAIL=$((FAIL + 1))
        return
    fi

    # integer percentage
    ratio=$(( printable * 100 / total ))
    # Preview: first 60 chars, newlines squashed
    preview=$(tr '\n' ' ' < "$TMP" | tr -s ' ' | cut -c1-60)

    if [ "$ratio" -ge 80 ]; then
        printf "  %-50s [PASS] %d%% printable, %d bytes | '%s...'\n" \
            "$model" "$ratio" "$total" "$preview"
        PASS=$((PASS + 1))
    else
        printf "  %-50s [FAIL] %d%% printable, %d bytes | '%s...'\n" \
            "$model" "$ratio" "$total" "$preview"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== quant.cpp Long-Sequence Stress Test (N=$N_TOKENS, T=0) ==="
echo "Models dir: $MODELS_DIR"
echo ""

# Short story continuation prompts — must sustain coherent generation.
run_long "Llama-3.2-1B-Instruct-Q8_0.gguf" \
    "Once upon a time in a small village by the sea, there lived a young woman named Elena who"
run_long "Llama-3.2-3B-Instruct-Q8_0.gguf" \
    "Once upon a time in a small village by the sea, there lived a young woman named Elena who"
run_long "Phi-3.5-mini-instruct-Q8_0.gguf" \
    "Here is a short essay on the importance of clear writing:"
run_long "Phi-3.5-mini-instruct-Q4_K_M.gguf" \
    "Here is a short essay on the importance of clear writing:"
run_long "Qwen3.5-4B-Q4_K_M.gguf" \
    "Write a short story about a robot who learns to paint" "--chat"
run_long "gemma-4-e2b-it-Q8_0.gguf" \
    "Write a short paragraph about the solar system:"

echo ""
echo "--- Summary ---"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"

[ "$FAIL" -gt 0 ] && exit 1
exit 0
