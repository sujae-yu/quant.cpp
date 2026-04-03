#!/bin/bash
# quant.cpp — Comprehensive Scoring Harness
# Compatible with bash 3.2+ (macOS default)
#
# Usage:
#   ./score.sh              # Full evaluation
#   ./score.sh --quick      # Build + test only
#   ./score.sh --json       # Machine-readable output

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
WBS_FILE="$PROJECT_DIR/docs/wbs_v0.1.md"
SCORE_LOG="$PROJECT_DIR/.score_history"
TMP_SCORES=$(mktemp /tmp/tq_scores.XXXXXX)
MODE="${1:---full}"

trap "rm -f $TMP_SCORES" EXIT

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

calc() { echo "scale=6; $1" | bc 2>/dev/null || echo "0"; }

# Append: category|name|score|max|weight
log_score() {
    echo "$1|$2|$3|$4|$5" >> "$TMP_SCORES"
}

print_item() {
    local name="$1" score="$2" max="$3" weight="$4"
    local color="$RED"
    local ratio=$(calc "$score / $max")
    if [ "$(calc "$ratio >= 1")" = "1" ]; then color="$GREEN"
    elif [ "$(calc "$ratio > 0")" = "1" ]; then color="$YELLOW"; fi
    printf "    ${color}%-28s %6s / %-6s  (w:%s)${NC}\n" "$name" "$score" "$max" "$weight"
}

# ============================================================
# DIMENSION 1: STRUCTURAL COMPLETENESS
# ============================================================
eval_structure() {
    echo -e "\n${BOLD}${CYAN}[1/5] STRUCTURAL COMPLETENESS${NC}"

    local h=0
    [ -f "$PROJECT_DIR/include/turboquant/turboquant.h" ] && h=$((h+1))
    [ -f "$PROJECT_DIR/include/turboquant/tq_types.h" ] && h=$((h+1))
    [ -f "$PROJECT_DIR/include/turboquant/tq_spec.h" ] && h=$((h+1))
    log_score "structure" "public_headers" "$h" 3 1
    print_item "public_headers" "$h" 3 1

    local api=0
    if [ -f "$PROJECT_DIR/include/turboquant/turboquant.h" ]; then
        for fn in tq_init tq_free tq_quantize_keys tq_quantize_values \
                  tq_attention tq_cache_create tq_cache_append tq_cache_free \
                  tq_cache_get_block tq_type_name tq_type_bpe tq_recommend_strategy; do
            grep -q "$fn" "$PROJECT_DIR/include/turboquant/turboquant.h" 2>/dev/null && api=$((api+1))
        done
    fi
    log_score "structure" "api_declarations" "$api" 12 2
    print_item "api_declarations" "$api" 12 2

    local core=0
    for f in tq_polar tq_qjl tq_turbo tq_uniform tq_value_quant tq_traits tq_paged_cache; do
        find "$PROJECT_DIR/src" -name "${f}.*" 2>/dev/null | grep -q . && core=$((core+1))
    done
    log_score "structure" "core_sources" "$core" 7 2
    print_item "core_sources" "$core" 7 2

    local be=0
    find "$PROJECT_DIR/src/backend/cpu" -name "*generic*" -o -name "*cpu_dispatch*" 2>/dev/null | grep -q . && be=$((be+1))
    find "$PROJECT_DIR/src/backend/cpu" -name "*avx2*" 2>/dev/null | grep -q . && be=$((be+1))
    find "$PROJECT_DIR/src/backend/cpu" -name "*neon*" 2>/dev/null | grep -q . && be=$((be+1))
    find "$PROJECT_DIR/src/backend/cuda" -name "*.cu" 2>/dev/null | grep -q . && be=$((be+1))
    find "$PROJECT_DIR/src/backend/metal" -name "*.metal" 2>/dev/null | grep -q . && be=$((be+1))
    log_score "structure" "backend_files" "$be" 5 2
    print_item "backend_files" "$be" 5 2

    local sp=0
    [ -f "$PROJECT_DIR/spec/tq_format_v1.md" ] && sp=$((sp+1))
    [ -f "$PROJECT_DIR/spec/tq_operators_v1.md" ] && sp=$((sp+1))
    [ -d "$PROJECT_DIR/spec/test_vectors" ] && [ "$(ls -A "$PROJECT_DIR/spec/test_vectors" 2>/dev/null)" ] && sp=$((sp+1))
    log_score "structure" "spec_documents" "$sp" 3 1
    print_item "spec_documents" "$sp" 3 1

    local tc=0
    for t in test_polar test_qjl test_turbo test_uniform test_value test_paged_cache; do
        find "$PROJECT_DIR/tests" -name "${t}.*" 2>/dev/null | grep -q . && tc=$((tc+1))
    done
    log_score "structure" "test_files" "$tc" 6 1
    print_item "test_files" "$tc" 6 1

    local wd=0 wt=1
    if [ -f "$WBS_FILE" ]; then
        wt=$(grep -c '^\- \[' "$WBS_FILE" 2>/dev/null || echo "1")
        wd=$(grep -c '^\- \[x\]' "$WBS_FILE" 2>/dev/null || echo "0")
    fi
    log_score "structure" "wbs_progress" "$wd" "$wt" 1
    print_item "wbs_progress" "$wd" "$wt" 1
}

# ============================================================
# DIMENSION 2: BUILD & CORRECTNESS
# ============================================================
eval_correctness() {
    echo -e "\n${BOLD}${CYAN}[2/5] BUILD & CORRECTNESS${NC}"

    local cmake_ok=0
    if [ -f "$PROJECT_DIR/CMakeLists.txt" ]; then
        mkdir -p "$BUILD_DIR"
        if cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
            -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON \
            "$PROJECT_DIR" > "$BUILD_DIR/.cmake_log" 2>&1; then
            cmake_ok=1
        fi
    fi
    log_score "correctness" "cmake_configure" "$cmake_ok" 1 2
    print_item "cmake_configure" "$cmake_ok" 1 2

    local build_ok=0
    if [ "$cmake_ok" -eq 1 ]; then
        local ncpu=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
        # Clean build to capture all warnings
        cmake --build "$BUILD_DIR" --clean-first -j"$ncpu" > "$BUILD_DIR/.build_log" 2>&1 && build_ok=1
    fi
    log_score "correctness" "build_success" "$build_ok" 1 3
    print_item "build_success" "$build_ok" 1 3

    local warn_score=0
    if [ "$build_ok" -eq 1 ]; then
        local wc=0
        if [ -f "$BUILD_DIR/.build_log" ]; then
            wc=$(grep -ci 'warning:' "$BUILD_DIR/.build_log" 2>/dev/null; true)
            wc=$(echo "$wc" | head -1 | tr -d '[:space:]')
        fi
        [ -z "$wc" ] && wc=0
        if [ "$wc" = "0" ]; then
            warn_score=1
        fi
    fi
    log_score "correctness" "zero_warnings" "$warn_score" 1 1
    print_item "zero_warnings" "$warn_score" 1 1

    local tp=0 tt=1
    if [ "$build_ok" -eq 1 ]; then
        local tout=$(cd "$BUILD_DIR" && ctest --output-on-failure --timeout 60 2>&1 || true)
        tt=$(echo "$tout" | grep -oE 'out of [0-9]+' | grep -oE '[0-9]+' | tail -1)
        # "100% tests passed" or "X tests passed"
        local passed_line=$(echo "$tout" | grep 'tests\? passed')
        if echo "$passed_line" | grep -q '100%'; then
            tp="$tt"
        else
            tp=$(echo "$passed_line" | grep -oE '^[0-9]+' || echo "0")
        fi
        [ -z "$tt" ] && tt=1
        [ -z "$tp" ] && tp=0
    fi
    log_score "correctness" "tests_passed" "$tp" "$tt" 4
    print_item "tests_passed" "$tp" "$tt" 4

    local sa=0
    if [ -f "$PROJECT_DIR/include/turboquant/tq_types.h" ]; then
        sa=$(grep -c 'static_assert\|_Static_assert\|TQ_CHECK_SIZE\|TQ_STATIC_ASSERT' "$PROJECT_DIR/include/turboquant/tq_types.h" 2>/dev/null; true)
        sa=$(echo "$sa" | head -1 | tr -d '[:space:]')
        [ -z "$sa" ] && sa=0
    fi
    local sas=0
    [ "$sa" -ge 4 ] 2>/dev/null && sas=1
    log_score "correctness" "static_asserts" "$sas" 1 1
    print_item "static_asserts" "$sas" 1 1
}

# ============================================================
# DIMENSION 3: QUANTIZATION QUALITY
# ============================================================
eval_quality() {
    echo -e "\n${BOLD}${CYAN}[3/5] QUANTIZATION QUALITY${NC}"

    local qt=0
    for q in test_polar test_qjl test_turbo test_uniform; do
        find "$PROJECT_DIR/tests" -name "${q}.*" 2>/dev/null | grep -q . && qt=$((qt+1))
    done
    log_score "quality" "quality_tests_exist" "$qt" 4 3
    print_item "quality_tests_exist" "$qt" 4 3

    # These require benchmark binaries to measure
    local has_bench=0
    [ -f "$BUILD_DIR/tq_bench" ] || [ -f "$BUILD_DIR/tq_quality" ] && has_bench=1

    if [ "$has_bench" -eq 1 ]; then
        local qout=$("$BUILD_DIR/tq_quality" 2>&1 || "$BUILD_DIR/tq_bench" --quality 2>&1 || true)
        local mse=$(echo "$qout" | grep -oE 'roundtrip_mse=[0-9.]+' | grep -oE '[0-9.]+' | head -1)
        local attn=$(echo "$qout" | grep -oE 'attention_cosine=[0-9.]+' | grep -oE '[0-9.]+' | head -1)
        [ -z "$mse" ] && mse=999
        [ -z "$attn" ] && attn=0

        local ms=0
        [ "$(calc "$mse < 0.01")" = "1" ] && ms=1
        [ "$(calc "$mse < 0.05")" = "1" ] && [ "$ms" -eq 0 ] && ms=0.7
        log_score "quality" "roundtrip_mse" "$ms" 1 4
        print_item "roundtrip_mse" "$ms" 1 4

        local as=0
        [ "$(calc "$attn > 0.99")" = "1" ] && as=1
        [ "$(calc "$attn > 0.95")" = "1" ] && [ "$(calc "$as == 0")" = "1" ] && as=0.7
        log_score "quality" "attention_accuracy" "$as" 1 4
        print_item "attention_accuracy" "$as" 1 4
    else
        log_score "quality" "roundtrip_mse" 0 1 4
        print_item "roundtrip_mse" 0 1 4
        log_score "quality" "attention_accuracy" 0 1 4
        print_item "attention_accuracy" 0 1 4
    fi
}

# ============================================================
# DIMENSION 4: PERFORMANCE
# ============================================================
eval_performance() {
    echo -e "\n${BOLD}${CYAN}[4/5] PERFORMANCE${NC}"

    local has=0
    [ -f "$BUILD_DIR/tq_bench" ] && has=1
    log_score "performance" "bench_exists" "$has" 1 2
    print_item "bench_exists" "$has" 1 2

    if [ "$has" -eq 1 ]; then
        local bout=$("$BUILD_DIR/tq_bench" 2>&1 || true)

        # quantize_throughput (target: >100K = 0.3, >1M = 1.0)
        local qt=$(echo "$bout" | grep -oE 'quantize_throughput=[0-9.]+' | grep -oE '[0-9.]+' | head -1)
        [ -z "$qt" ] && qt=0
        local qts=0
        [ "$(calc "$qt > 1000000")" = "1" ] && qts=1
        [ "$(calc "$qt > 100000")" = "1" ] && [ "$(calc "$qts == 0")" = "1" ] && qts=0.7
        [ "$(calc "$qt > 10000")" = "1" ] && [ "$(calc "$qts == 0")" = "1" ] && qts=0.3
        log_score "performance" "quantize_throughput" "$qts" 1 3
        print_item "quantize_throughput" "$qts" 1 3

        # attention_throughput (target: >1K = 0.3, >10K = 1.0)
        local at=$(echo "$bout" | grep -oE 'attention_throughput=[0-9.]+' | grep -oE '[0-9.]+' | head -1)
        [ -z "$at" ] && at=0
        local ats=0
        [ "$(calc "$at > 10000")" = "1" ] && ats=1
        [ "$(calc "$at > 1000")" = "1" ] && [ "$(calc "$ats == 0")" = "1" ] && ats=0.7
        [ "$(calc "$at > 100")" = "1" ] && [ "$(calc "$ats == 0")" = "1" ] && ats=0.3
        log_score "performance" "attention_throughput" "$ats" 1 3
        print_item "attention_throughput" "$ats" 1 3

        # compression_ratio (target: >5x)
        local cr=$(echo "$bout" | grep -oE 'compression_ratio=[0-9.]+' | grep -oE '[0-9.]+' | head -1)
        [ -z "$cr" ] && cr=0
        local crs=0
        [ "$(calc "$cr >= 5.0")" = "1" ] && crs=1
        [ "$(calc "$cr >= 4.0")" = "1" ] && [ "$(calc "$crs == 0")" = "1" ] && crs=0.7
        [ "$(calc "$cr >= 3.0")" = "1" ] && [ "$(calc "$crs == 0")" = "1" ] && crs=0.4
        log_score "performance" "memory_compression" "$crs" 1 3
        print_item "memory_compression" "$crs" 1 3

        # simd_speedup (target: >4x; 1.0 = no speedup yet)
        local ss=$(echo "$bout" | grep -oE 'simd_speedup=[0-9.]+' | grep -oE '[0-9.]+' | head -1)
        [ -z "$ss" ] && ss=0
        local sss=0
        [ "$(calc "$ss >= 4.0")" = "1" ] && sss=1
        [ "$(calc "$ss >= 2.0")" = "1" ] && [ "$(calc "$sss == 0")" = "1" ] && sss=0.7
        [ "$(calc "$ss >= 1.5")" = "1" ] && [ "$(calc "$sss == 0")" = "1" ] && sss=0.3
        log_score "performance" "simd_speedup" "$sss" 1 3
        print_item "simd_speedup" "$sss" 1 3
    else
        for m in quantize_throughput attention_throughput memory_compression simd_speedup; do
            log_score "performance" "$m" 0 1 3
            print_item "$m" 0 1 3
        done
    fi
}

# ============================================================
# DIMENSION 5: INTEGRATION
# ============================================================
eval_integration() {
    echo -e "\n${BOLD}${CYAN}[5/5] INTEGRATION & MATURITY${NC}"

    local ll=0; [ -d "$PROJECT_DIR/integrations/llamacpp" ] && [ "$(ls -A "$PROJECT_DIR/integrations/llamacpp" 2>/dev/null)" ] && ll=1
    log_score "integration" "llamacpp_plugin" "$ll" 1 2
    print_item "llamacpp_plugin" "$ll" 1 2

    local py=0; [ -d "$PROJECT_DIR/bindings/python" ] && [ "$(ls -A "$PROJECT_DIR/bindings/python" 2>/dev/null)" ] && py=1
    log_score "integration" "python_bindings" "$py" 1 1
    print_item "python_bindings" "$py" 1 1

    local vl=0; [ -d "$PROJECT_DIR/integrations/vllm" ] && [ "$(ls -A "$PROJECT_DIR/integrations/vllm" 2>/dev/null)" ] && vl=1
    log_score "integration" "vllm_plugin" "$vl" 1 1
    print_item "vllm_plugin" "$vl" 1 1

    local ex=$(find "$PROJECT_DIR/examples" -name "*.c" -o -name "*.cpp" -o -name "*.py" 2>/dev/null | wc -l | tr -d ' ')
    [ "$ex" -gt 3 ] && ex=3
    log_score "integration" "examples" "$ex" 3 1
    print_item "examples" "$ex" 3 1

    local dc=0
    [ -f "$PROJECT_DIR/README.md" ] && [ "$(wc -l < "$PROJECT_DIR/README.md" 2>/dev/null || echo 0)" -gt 20 ] 2>/dev/null && dc=$((dc+1))
    [ -f "$PROJECT_DIR/docs/architecture.md" ] && dc=$((dc+1))
    [ -f "$PROJECT_DIR/docs/integration_guide.md" ] && dc=$((dc+1))
    log_score "integration" "documentation" "$dc" 3 1
    print_item "documentation" "$dc" 3 1
}

# ============================================================
# FINAL REPORT
# ============================================================
print_final() {
    local grand_weighted=0
    local grand_weight=0

    while IFS='|' read -r cat name score max weight; do
        [ -z "$cat" ] && continue
        local norm=$(calc "$score / $max")
        [ "$(calc "$norm > 1")" = "1" ] && norm=1
        local w=$(calc "$norm * $weight")
        grand_weighted=$(calc "$grand_weighted + $w")
        grand_weight=$(calc "$grand_weight + $weight")
    done < "$TMP_SCORES"

    local final=0
    [ "$(calc "$grand_weight > 0")" = "1" ] && final=$(calc "$grand_weighted / $grand_weight")

    echo ""
    echo "========================================"
    printf "  ${BOLD}${GREEN}TOTAL SCORE:  %.4f / 1.0000  (%.1f%%)${NC}\n" "$final" "$(calc "$final * 100")"
    echo "========================================"

    # Dimension breakdown
    echo ""
    echo "  Dimension Breakdown:"
    for dim in structure correctness quality performance integration; do
        local ds=0 dw=0
        while IFS='|' read -r cat name score max weight; do
            if [ "$cat" = "$dim" ]; then
                local n=$(calc "$score / $max")
                [ "$(calc "$n > 1")" = "1" ] && n=1
                ds=$(calc "$ds + $n * $weight")
                dw=$(calc "$dw + $weight")
            fi
        done < "$TMP_SCORES"
        if [ "$(calc "$dw > 0")" = "1" ]; then
            printf "    %-20s %5.1f%%\n" "$dim" "$(calc "$ds / $dw * 100")"
        else
            printf "    %-20s %5.1f%%\n" "$dim" "0.0"
        fi
    done

    # Save history
    echo "$(date '+%Y-%m-%d %H:%M:%S') $final" >> "$SCORE_LOG"

    # Trend
    if [ -f "$SCORE_LOG" ] && [ "$(wc -l < "$SCORE_LOG" | tr -d ' ')" -gt 1 ]; then
        echo ""
        echo "  Score History (last 5):"
        tail -5 "$SCORE_LOG" | while IFS=' ' read -r d t s; do
            printf "    %s %s  %s\n" "$d" "$t" "$s"
        done
    fi

    printf "%.4f" "$final" > "$PROJECT_DIR/.score"
    echo ""
}

# ============================================================
# MAIN
# ============================================================
echo ""
echo "========================================"
echo "  quant.cpp — Scoring Report"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

case "$MODE" in
    --quick)
        eval_structure
        eval_correctness
        ;;
    --bench)
        eval_performance
        ;;
    --quality)
        eval_quality
        ;;
    --full|*)
        eval_structure
        eval_correctness
        eval_quality
        eval_performance
        eval_integration
        ;;
esac

print_final
