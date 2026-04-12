#!/usr/bin/env bash
# check_sync.sh — verify critical code sections are in sync between
# quant.h (single header) and src/ (split sources).
#
# This catches the #67-class bug: a feature implemented in quant.h
# but not ported to the split sources (or vice versa).
#
# Usage: bash scripts/check_sync.sh
# Returns 0 if all checks pass, 1 if any drift is detected.

set -euo pipefail

HEADER="quant.h"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

check_marker_list() {
    local label="$1"
    local file1="$2"
    local file2="$3"
    local pattern="$4"

    local list1 list2
    list1=$(grep -o "$pattern" "$file1" 2>/dev/null | sort -u)
    list2=$(grep -o "$pattern" "$file2" 2>/dev/null | sort -u)

    if [ "$list1" = "$list2" ]; then
        echo -e "  ${GREEN}✓${NC} $label"
    else
        echo -e "  ${RED}✗${NC} $label — MISMATCH"
        diff <(echo "$list1") <(echo "$list2") || true
        ERRORS=$((ERRORS + 1))
    fi
}

check_field_exists() {
    local label="$1"
    local field="$2"
    local file="$3"

    if grep -q "$field" "$file" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $label: '$field' found in $(basename $file)"
    else
        echo -e "  ${RED}✗${NC} $label: '$field' MISSING in $(basename $file)"
        ERRORS=$((ERRORS + 1))
    fi
}

check_both_have() {
    local label="$1"
    local pattern="$2"
    local file1="$3"
    local file2="$4"

    local has1 has2
    has1=$(grep -c "$pattern" "$file1" 2>/dev/null || echo 0)
    has2=$(grep -c "$pattern" "$file2" 2>/dev/null || echo 0)

    if [ "$has1" -gt 0 ] && [ "$has2" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} $label: present in both files"
    elif [ "$has1" -eq 0 ] && [ "$has2" -eq 0 ]; then
        echo -e "  ${YELLOW}—${NC} $label: absent in both (OK if not yet needed)"
    else
        local missing
        [ "$has1" -eq 0 ] && missing="$(basename $file1)" || missing="$(basename $file2)"
        echo -e "  ${RED}✗${NC} $label: MISSING in $missing"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "=== quant.h ↔ split-source sync check ==="
echo ""

# --- 1. CHAT_END_MARKERS list ---
echo "[1] CHAT_END_MARKERS (template token filter)"
# Extract only the markers from the CHAT_END_MARKERS array definition
extract_markers() {
    sed -n '/CHAT_END_MARKERS\[\]/,/NULL/p' "$1" | grep -o '"[^"]*"' | sort -u
}
local_m1=$(extract_markers "$HEADER")
local_m2=$(extract_markers "src/engine/tq_generate.c")
if [ "$local_m1" = "$local_m2" ]; then
    echo -e "  ${GREEN}✓${NC} End markers"
else
    echo -e "  ${RED}✗${NC} End markers — MISMATCH"
    diff <(echo "$local_m1") <(echo "$local_m2") || true
    ERRORS=$((ERRORS + 1))
fi

# --- 2. Phi-3 fused tensor support ---
echo ""
echo "[2] Phi-3 fused tensor fields"
check_field_exists "Config: has_fused_qkv" "has_fused_qkv" "include/turboquant/tq_engine.h"
check_field_exists "Config: has_fused_up_gate" "has_fused_up_gate" "include/turboquant/tq_engine.h"
check_field_exists "Layer: gguf_w_qkv" "gguf_w_qkv" "include/turboquant/tq_engine.h"
check_field_exists "Layer: gguf_w_up_gate" "gguf_w_up_gate" "include/turboquant/tq_engine.h"
check_field_exists "Config: rope_factors_short" "rope_factors_short" "include/turboquant/tq_engine.h"

# --- 3. Fused QKV forward path ---
echo ""
echo "[3] Fused QKV forward path"
check_both_have "Fused QKV matmul" "gguf_w_qkv" \
    "$HEADER" "src/engine/tq_transformer.c"
check_both_have "Fused FFN gate||up" "gguf_w_up_gate" \
    "$HEADER" "src/engine/tq_transformer.c"

# --- 4. LongRoPE ---
echo ""
echo "[4] LongRoPE rotation"
check_both_have "rope_factors_short" "rope_factors_short" \
    "$HEADER" "src/engine/tq_transformer.c"
check_both_have "rope_factors_long" "rope_factors_long" \
    "$HEADER" "src/engine/tq_transformer.c"

# --- 5. BOS token handling ---
echo ""
echo "[5] BOS token handling"
check_both_have "BOS <s> lookup in tokenizer" '"<s>"' \
    "$HEADER" "src/engine/tq_tokenizer.c"
check_both_have "BOS <s> auto-detect in generate" '"<s>"' \
    "$HEADER" "src/engine/tq_generate.c"
check_both_have "BOS <|begin_of_text|> lookup" '"<|begin_of_text|>"' \
    "$HEADER" "src/engine/tq_tokenizer.c"

# --- 6. Hybrid attention stride (GQA fix) ---
echo ""
echo "[6] Hybrid attention cache stride"
check_both_have "max_head_dim in quant cache" "max_head_dim" \
    "$HEADER" "src/engine/tq_transformer.c"
check_both_have "max_kv_heads in quant cache" "max_kv_heads" \
    "$HEADER" "src/engine/tq_transformer.c"

# --- 7. Memory free completeness ---
echo ""
echo "[7] GGUF dequant memory free"
check_both_have "free(layer->attn_norm)" "free(layer->attn_norm)" \
    "$HEADER" "src/engine/tq_model.c"

# --- Summary ---
echo ""
echo "========================================="
if [ "$ERRORS" -eq 0 ]; then
    echo -e "  ${GREEN}ALL CHECKS PASSED${NC}"
else
    echo -e "  ${RED}$ERRORS SYNC ISSUES DETECTED${NC}"
fi
echo "========================================="
exit "$ERRORS"
