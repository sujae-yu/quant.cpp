#!/usr/bin/env bash
# check_stale.sh — warn when built binaries are older than the core library.
#
# Why: on 2026-04-15 we spent significant time chasing a "CLI vs server
# behavior mismatch" that turned out to be a stale server binary linked
# against an older libturboquant that pre-dated a Qwen3.5 DeltaNet fix.
# Both tools loaded the same GGUF file and produced different results.
#
# Usage: bash scripts/check_stale.sh [build_dir]
# Exit codes: 0 = all fresh, 1 = at least one binary is stale.

set -u
BUILD="${1:-build}"
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

if [ ! -d "$BUILD" ]; then
    echo "build dir '$BUILD' not found"
    exit 0
fi

mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null || echo 0; }

# Compare binaries to the newest SOURCE file — not to build artifacts,
# since CMake touches dylibs even when nothing changes.
#   lib_src_mtime    — newest .c/.h in src/ and include/
#   header_mtime     — quant.h mtime (covers single-header binaries)
lib_src_mtime=0
lib_src_path=""
# Use find with -printf/-f and sort — portable across macOS/Linux.
while IFS= read -r f; do
    m=$(mtime "$f")
    if [ "$m" -gt "$lib_src_mtime" ]; then
        lib_src_mtime="$m"
        lib_src_path="$f"
    fi
done < <(find src include -type f \( -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.metal' \) 2>/dev/null)

header_mtime=0
if [ -f "quant.h" ]; then
    header_mtime=$(mtime "quant.h")
fi

if [ "$lib_src_mtime" -eq 0 ] && [ "$header_mtime" -eq 0 ]; then
    echo "no sources found — are you in the repo root?"
    exit 0
fi

[ "$lib_src_mtime" -gt 0 ] && echo "split-source newest: $lib_src_path ($(date -r "$lib_src_mtime" 2>/dev/null))"
[ "$header_mtime" -gt 0 ] && echo "single-header ref:   quant.h ($(date -r "$header_mtime" 2>/dev/null))"
echo ""

STALE=0
CHECKED=0

check_bin() {
    local bin="$1"
    local ref_mtime="$2"
    local ref_label="$3"
    if [ ! -f "$bin" ]; then return; fi
    CHECKED=$((CHECKED + 1))
    local m age hours
    m=$(mtime "$bin")
    age=$((ref_mtime - m))
    local name
    name=$(basename "$bin")
    if [ "$age" -gt 0 ]; then
        STALE=$((STALE + 1))
        hours=$((age / 3600))
        echo -e "  ${RED}✗${NC} $name is STALE vs $ref_label (older by ${hours}h)"
    else
        echo -e "  ${GREEN}✓${NC} $name is fresh vs $ref_label"
    fi
}

# Split-source binaries — compared against newest src/ file.
if [ "$lib_src_mtime" -gt 0 ]; then
    for bin in \
        "$BUILD/quant" \
        "$BUILD/quant-server" \
        "$BUILD/tq_convert" \
        "$BUILD/standalone"; do
        check_bin "$bin" "$lib_src_mtime" "split sources"
    done
fi

# Single-header binaries — compile quant.h directly, compared against it.
if [ "$header_mtime" -gt 0 ]; then
    for bin in \
        "$BUILD/quant-server-unified" \
        "$BUILD/single_header_example"; do
        check_bin "$bin" "$header_mtime" "quant.h"
    done
fi

echo ""
if [ "$STALE" -gt 0 ]; then
    echo -e "${RED}$STALE/$CHECKED binaries are stale.${NC} Rebuild with:"
    echo "  cmake --build $BUILD --target quant-server-unified single_header_example quant"
    exit 1
fi
echo -e "${GREEN}All $CHECKED binaries are fresh.${NC}"
exit 0
