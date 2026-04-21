#!/usr/bin/env bash
# Execute the full reference-parity matrix: (model × prompt) × diff.
# Reads matrix.json, produces PASS/FAIL per entry, exits non-zero on any failure.
#
# Usage:
#   bash run_matrix.sh                       # full matrix
#   MODELS_DIR=/path bash run_matrix.sh       # override GGUF dir
#   FILTER=qwen3 bash run_matrix.sh           # only entries whose name contains "qwen3"

set -u

cd "$(dirname "$0")"

MATRIX="${MATRIX:-matrix.json}"
MODELS_DIR="${MODELS_DIR:-../../models}"
VENV_DIR="${VENV_DIR:-../pillar1/venv}"   # reuse pillar1's venv by default
FILTER="${FILTER:-}"

if [[ ! -d "$VENV_DIR" ]]; then
    cat >&2 <<EOF
error: Python venv not found at $VENV_DIR

Set up once:
    python3.12 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install torch transformers accelerate huggingface_hub

Or set VENV_DIR=/path/to/existing/venv
EOF
    exit 2
fi

source "$VENV_DIR/bin/activate"

# Parse matrix.json via python one-liner; emit shell-friendly lines
ENTRIES=$(python3 <<PY
import json, sys, os
with open("$MATRIX") as f:
    m = json.load(f)
filt = "$FILTER"
for t in m.get("tests", []):
    if filt and filt not in t["name"]:
        continue
    gguf = t["engine_gguf"]
    for prompt in t["prompts"]:
        # bash-escaped prompt (base64 roundtrip to avoid quoting hell)
        import base64
        p64 = base64.b64encode(prompt.encode()).decode()
        print(f"{t['name']}|{t['hf_model']}|{gguf}|{p64}|{t.get('threshold_l2_rel', 0.05)}|{t.get('threshold_cosine', 0.90)}")
PY
)

TOTAL=0
PASSED=0
FAILED_ENTRIES=()

WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "=== Reference-Parity Matrix ==="
echo "Matrix: $MATRIX"
echo "Models dir: $MODELS_DIR"
echo "Work dir: $WORK_DIR"
[[ -n "$FILTER" ]] && echo "Filter: $FILTER"
echo ""

PREV_NAME=""
IDX=0
while IFS='|' read -r NAME HF_MODEL GGUF P64 TH_L2 TH_COS; do
    [[ -z "$NAME" ]] && continue
    PROMPT=$(echo "$P64" | base64 -d)
    TOTAL=$((TOTAL + 1))

    # Unique slot per (name, prompt) so reports don't overwrite.
    # ENTRIES are grouped by NAME via matrix.json, so reset idx when NAME changes.
    if [[ "$NAME" != "$PREV_NAME" ]]; then
        IDX=0
        PREV_NAME="$NAME"
    else
        IDX=$((IDX + 1))
    fi
    SLOT="${NAME}__p${IDX}"

    PROMPT_SHORT="${PROMPT:0:40}"
    [[ "${#PROMPT}" -gt 40 ]] && PROMPT_SHORT="${PROMPT_SHORT}..."
    echo "── $NAME [p${IDX}] :: \"$PROMPT_SHORT\""

    GGUF_PATH="$MODELS_DIR/$GGUF"
    if [[ ! -f "$GGUF_PATH" ]]; then
        echo "   [SKIP] GGUF not found: $GGUF_PATH"
        continue
    fi

    # HF reference dump (one per slot — different prompts produce different tokens)
    REF_NPZ="$WORK_DIR/$SLOT.npz"
    if ! python hf_reference.py --model "$HF_MODEL" --prompt "$PROMPT" --out "$REF_NPZ" 2>"$WORK_DIR/hf.err"; then
        echo "   [ERROR] HF reference failed:"
        sed 's/^/     /' "$WORK_DIR/hf.err"
        FAILED_ENTRIES+=("$SLOT: hf_reference failed")
        continue
    fi

    # Engine dump
    ENG_DIR="$WORK_DIR/$SLOT.engine"
    if ! bash engine_reference.sh "$GGUF_PATH" "$PROMPT" "$ENG_DIR" 2>"$WORK_DIR/eng.err"; then
        echo "   [ERROR] Engine dump failed:"
        sed 's/^/     /' "$WORK_DIR/eng.err"
        FAILED_ENTRIES+=("$SLOT: engine_reference failed")
        continue
    fi

    # Layer diff
    if python diff_layers.py "$REF_NPZ" "$ENG_DIR" \
           --threshold-l2-rel "$TH_L2" \
           --threshold-cosine "$TH_COS" > "$WORK_DIR/$SLOT.diff" 2>&1; then
        echo "   [PASS]"
        PASSED=$((PASSED + 1))
    else
        echo "   [FAIL]"
        # Show the tail of the diff report (summary + first-fail line)
        tail -20 "$WORK_DIR/$SLOT.diff" | sed 's/^/     /'
        FAILED_ENTRIES+=("$SLOT")
    fi
done <<< "$ENTRIES"

echo ""
echo "── Summary ──"
echo "  PASS:  $PASSED / $TOTAL"
echo "  FAIL:  $((TOTAL - PASSED))"
if [[ ${#FAILED_ENTRIES[@]} -gt 0 ]]; then
    printf "  — %s\n" "${FAILED_ENTRIES[@]}"
fi

# Copy any failure reports out of tmp so user can inspect
if [[ ${#FAILED_ENTRIES[@]} -gt 0 ]]; then
    mkdir -p reports
    cp "$WORK_DIR"/*.diff reports/ 2>/dev/null || true
    echo "  reports: tools/refparity/reports/"
    exit 1
fi
exit 0
