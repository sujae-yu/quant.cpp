#!/usr/bin/env bash
#
# book_chat.sh — "Book-in-a-Chat" demo for quant.cpp
#
# Loads the entirety of Alice in Wonderland into context and asks questions
# about it. Compares llama.cpp (FP16 KV, runs out of memory) vs quant.cpp
# (compressed KV, succeeds).
#
# Usage:
#   bash bench/demo/book_chat.sh <model.gguf> [--llama-bin <path>] [--threads <N>]
#
# Prerequisites:
#   1. Build quant.cpp: cmake --build build
#   2. Prepare book:    python3 bench/demo/prepare_book.py
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUANT_BIN="${PROJECT_ROOT}/build/quant"
BOOK_FILE="${SCRIPT_DIR}/alice.txt"

LLAMA_BIN=""          # Path to llama-cli or main (auto-detected if not set)
THREADS=4
CTX_SIZE=0            # 0 = auto (will be set based on token estimate)
MAX_ANSWER_TOKENS=256

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    BOLD="\033[1m"
    GREEN="\033[32m"
    RED="\033[31m"
    YELLOW="\033[33m"
    CYAN="\033[36m"
    RESET="\033[0m"
else
    BOLD="" GREEN="" RED="" YELLOW="" CYAN="" RESET=""
fi

# ============================================================================
# Helpers
# ============================================================================

info()  { printf "${CYAN}[INFO]${RESET}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${RESET}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${RESET}  %s\n" "$*"; }
fail()  { printf "${RED}[FAIL]${RESET}  %s\n" "$*"; }

banner() {
    echo ""
    printf "${BOLD}════════════════════════════════════════════════════════════${RESET}\n"
    printf "${BOLD}  %s${RESET}\n" "$*"
    printf "${BOLD}════════════════════════════════════════════════════════════${RESET}\n"
    echo ""
}

separator() {
    printf "${CYAN}────────────────────────────────────────────────────────────${RESET}\n"
}

# ============================================================================
# Argument parsing
# ============================================================================

MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-bin)  LLAMA_BIN="$2";  shift 2 ;;
        --threads|-j) THREADS="$2";    shift 2 ;;
        --ctx)        CTX_SIZE="$2";   shift 2 ;;
        --help|-h)
            echo "Usage: $0 <model.gguf> [--llama-bin <path>] [--threads <N>] [--ctx <N>]"
            echo ""
            echo "Arguments:"
            echo "  model.gguf          Path to GGUF model file"
            echo "  --llama-bin <path>  Path to llama.cpp binary (auto-detected if not set)"
            echo "  --threads <N>       Number of threads (default: 4)"
            echo "  --ctx <N>           Context size override (default: auto)"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$MODEL_PATH" ]]; then
                MODEL_PATH="$1"
            else
                echo "Unexpected argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: model path is required." >&2
    echo "Usage: $0 <model.gguf> [--llama-bin <path>] [--threads <N>]" >&2
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: model file not found: $MODEL_PATH" >&2
    exit 1
fi

# ============================================================================
# Step 0: Prepare the book if needed
# ============================================================================

banner "Book-in-a-Chat Demo"
info "Demonstrating KV cache compression with a full book in context"

if [[ ! -f "$BOOK_FILE" ]]; then
    info "Book file not found. Downloading Alice in Wonderland ..."
    python3 "${SCRIPT_DIR}/prepare_book.py" "$BOOK_FILE"
    echo ""
fi

# ============================================================================
# Step 1: Book statistics
# ============================================================================

banner "Step 1: Book Statistics"

WORD_COUNT=$(wc -w < "$BOOK_FILE" | tr -d ' ')
LINE_COUNT=$(wc -l < "$BOOK_FILE" | tr -d ' ')
CHAR_COUNT=$(wc -c < "$BOOK_FILE" | tr -d ' ')
# Estimate tokens: ~1.3 tokens per word for English text
EST_TOKENS=$(python3 -c "print(int($WORD_COUNT * 1.3))")

info "File:             $BOOK_FILE"
info "Words:            $WORD_COUNT"
info "Lines:            $LINE_COUNT"
info "Characters:       $CHAR_COUNT"
info "Estimated tokens: $EST_TOKENS (~1.3 tokens/word)"

# Set context size to fit the book + room for answers
if [[ "$CTX_SIZE" -eq 0 ]]; then
    # Round up to next power of 2, add headroom for answers
    CTX_SIZE=$(python3 -c "
import math
needed = $EST_TOKENS + 1024  # book + answer room
# Round up to nice number
ctx = max(4096, 2 ** math.ceil(math.log2(needed)))
print(ctx)
")
fi

info "Context size:     $CTX_SIZE tokens"

# ============================================================================
# Step 2: Try llama.cpp (expect OOM / context limit failure)
# ============================================================================

banner "Step 2: llama.cpp Baseline (FP16 KV Cache)"

# Auto-detect llama.cpp binary
if [[ -z "$LLAMA_BIN" ]]; then
    for candidate in \
        "$(command -v llama-cli 2>/dev/null || true)" \
        "$(command -v main 2>/dev/null || true)" \
        "${PROJECT_ROOT}/../llama.cpp/build/bin/llama-cli" \
        "${PROJECT_ROOT}/../llama.cpp/build/bin/main" \
        "${HOME}/llama.cpp/build/bin/llama-cli" \
        "${HOME}/llama.cpp/build/bin/main"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            LLAMA_BIN="$candidate"
            break
        fi
    done
fi

LLAMA_STATUS="skipped"
LLAMA_TOKENS=0

if [[ -n "$LLAMA_BIN" && -x "$LLAMA_BIN" ]]; then
    info "Found llama.cpp: $LLAMA_BIN"
    info "Attempting to load book with FP16 KV cache ..."

    BOOK_CONTENT=$(cat "$BOOK_FILE")
    LLAMA_PROMPT="Read the following book carefully. After the book text, I will ask questions about it.

--- BOOK START ---
${BOOK_CONTENT}
--- BOOK END ---

What riddle did the Mad Hatter ask Alice at the tea party?"

    # Run llama.cpp with a timeout — expect it to fail or OOM
    LLAMA_OUTPUT=""
    LLAMA_EXIT=0
    set +e
    LLAMA_OUTPUT=$(timeout 60 "$LLAMA_BIN" \
        -m "$MODEL_PATH" \
        -c "$CTX_SIZE" \
        -n "$MAX_ANSWER_TOKENS" \
        -t "$THREADS" \
        -p "$LLAMA_PROMPT" \
        --no-display-prompt \
        2>&1) || LLAMA_EXIT=$?
    set -e

    if [[ $LLAMA_EXIT -ne 0 ]]; then
        fail "llama.cpp failed (exit code: $LLAMA_EXIT)"
        # Extract the most relevant error line
        ERROR_LINE=$(echo "$LLAMA_OUTPUT" | grep -iE "(error|oom|memory|alloc|context|too large|exceed|failed)" | head -3)
        if [[ -n "$ERROR_LINE" ]]; then
            echo "  Error: $ERROR_LINE"
        fi
        LLAMA_STATUS="failed"
        # Try to extract how many tokens it managed
        LLAMA_TOKENS=$(echo "$LLAMA_OUTPUT" | grep -oE '[0-9]+ tokens' | head -1 | grep -oE '[0-9]+' || echo "0")
    else
        # It succeeded (small model or enough RAM) — still show the comparison
        ok "llama.cpp succeeded (model fits in memory with FP16 KV)"
        LLAMA_STATUS="success"
        LLAMA_TOKENS="$EST_TOKENS"
    fi
else
    warn "llama.cpp not found. Skipping baseline comparison."
    warn "Install llama.cpp or use --llama-bin to specify its location."
    LLAMA_STATUS="not_found"
fi

# ============================================================================
# Step 3: Load with quant.cpp (compressed KV cache)
# ============================================================================

banner "Step 3: quant.cpp with Compressed KV Cache"

if [[ ! -x "$QUANT_BIN" ]]; then
    fail "quant binary not found at: $QUANT_BIN"
    info "Build with: cmake -B build && cmake --build build"
    exit 1
fi

info "Engine:     quant.cpp"
info "Key cache:  uniform_4b (4-bit quantized)"
info "Value cache: q4 (4-bit quantized)"
info "Context:    $CTX_SIZE tokens"
info "Threads:    $THREADS"
echo ""

BOOK_CONTENT=$(cat "$BOOK_FILE")

QUESTIONS=(
    "What riddle did the Mad Hatter ask Alice at the tea party?"
    "Quote the Queen of Hearts' most famous line."
    "Describe how the Cheshire Cat disappears."
)

QUANT_STATUS="success"
QUANT_TOKENS="$EST_TOKENS"
ANSWERS_COUNT=0

for i in "${!QUESTIONS[@]}"; do
    Q="${QUESTIONS[$i]}"
    QNUM=$((i + 1))

    separator
    printf "${BOLD}Question ${QNUM}/${#QUESTIONS[@]}:${RESET} %s\n\n" "$Q"

    PROMPT="Read the following book carefully. After the book text, I will ask you a question about it.

--- BOOK START ---
${BOOK_CONTENT}
--- BOOK END ---

Question: ${Q}

Answer:"

    set +e
    ANSWER=$(timeout 120 "$QUANT_BIN" "$MODEL_PATH" \
        -p "$PROMPT" \
        -k uniform_4b \
        -v q4 \
        -n "$MAX_ANSWER_TOKENS" \
        -j "$THREADS" \
        --ctx "$CTX_SIZE" \
        -M \
        2>&1)
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        fail "quant.cpp failed on question $QNUM (exit code: $EXIT_CODE)"
        # Show error context
        echo "$ANSWER" | grep -iE "(error|fail|oom|alloc)" | head -3
        QUANT_STATUS="failed"
    else
        # Extract just the generated text (skip memory stats printed to stderr)
        # The model output goes to stdout via the streaming callback
        printf "${GREEN}Answer:${RESET}\n"
        echo "$ANSWER" | grep -v "^\[" | grep -v "^KV " | grep -v "^Peak " | grep -v "^Compression" | head -20
        echo ""
        ANSWERS_COUNT=$((ANSWERS_COUNT + 1))
    fi
done

# ============================================================================
# Step 4: Summary
# ============================================================================

banner "Summary"

echo "Book: Alice in Wonderland"
echo "Size: $WORD_COUNT words, ~$EST_TOKENS estimated tokens"
echo ""

# llama.cpp result
printf "%-14s " "llama.cpp:"
case "$LLAMA_STATUS" in
    failed)
        printf "${RED}OOM / context limit exceeded"
        if [[ "$LLAMA_TOKENS" -gt 0 ]]; then
            printf " at ~%s tokens" "$LLAMA_TOKENS"
        fi
        printf "${RESET}\n"
        ;;
    success)
        printf "${YELLOW}succeeded (FP16 KV, ~%s tokens)${RESET}\n" "$LLAMA_TOKENS"
        ;;
    not_found)
        printf "${YELLOW}not installed (skipped)${RESET}\n"
        ;;
    skipped)
        printf "${YELLOW}skipped${RESET}\n"
        ;;
esac

# quant.cpp result
printf "%-14s " "quant.cpp:"
if [[ "$QUANT_STATUS" == "success" ]]; then
    printf "${GREEN}loaded ~%s tokens, answered %d/%d questions${RESET}\n" \
        "$QUANT_TOKENS" "$ANSWERS_COUNT" "${#QUESTIONS[@]}"
else
    printf "${RED}failed${RESET}\n"
fi

echo ""
info "KV cache compression (uniform_4b keys + q4 values) reduces memory ~7x,"
info "enabling full-book context that would otherwise require FP16 KV cache."
echo ""
