#!/usr/bin/env bash
# Broad validation of the R3 BPE fix across realistic use cases.

BIN="${BIN:-./build/quant}"
MODEL="${MODEL:-models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"

export TQ_NO_METAL=1
export TQ_NO_MLOCK=1
export LC_ALL=C

PASS=0
FAIL=0

check() {
    local name="$1"
    local prompt="$2"
    local expected="$3"
    local n="${4:-40}"
    local chat="${5:---chat}"
    local out
    out=$("$BIN" "$MODEL" $chat -p "$prompt" -n $n -T 0 2>/dev/null | tr '\n' ' ')
    local pretty="${out:0:100}"
    if [[ "$out" == *"$expected"* ]]; then
        printf "  %-20s [PASS] '%s...'\n" "$name" "$pretty"
        PASS=$((PASS+1))
    else
        printf "  %-20s [FAIL] need '%s' | got '%s'\n" "$name" "$expected" "$pretty"
        FAIL=$((FAIL+1))
    fi
}

echo "=== R4 Broad Validation — Qwen3.6-35B IQ4_XS ==="

check "short_story"  "Once upon a time"                "young"        40
check "short_code"   "def fibonacci(n):"               "return"       40  ""
check "short_qa"     "What is the capital of France?"  "Paris"        30

check "mid_recipe"   "Explain how to make a simple pasta dish with tomatoes, garlic, olive oil, salt, and pepper in a few steps." "garlic" 80
check "mid_tech"     "Describe briefly what a hash table is in computer science and why it's useful for fast lookups in programming." "hash" 80

check "long_code"    "Write a Python function that computes the nth Fibonacci number using iterative dynamic programming. It should handle edge cases including negative numbers, zero, and very large inputs. Include proper docstrings and type hints." "def" 100
check "long_story"   "Once upon a time in a small village there lived a clever young programmer named Luna who was known throughout the kingdom for her extraordinary ability to solve the most difficult computer science problems." "Luna" 100 ""
check "long_essay"   "Please explain in a clear and concise manner what the main differences are between supervised learning and unsupervised learning in machine learning, including typical use cases and examples of algorithms used in each approach." "learning" 120

echo ""
echo "--- Summary --- PASS=$PASS FAIL=$FAIL"
exit $FAIL
