#!/bin/bash
# quant.cpp — Hierarchical Harness Runner
#
# Combines two methodologies:
#   - Karpathy AutoResearch: score → improve → score → revert-if-worse
#   - ClawTeam Multi-Agent: parallel workers in isolated worktrees
#
# Architecture:
#   Outer Loop (Leader): score full project → identify bottleneck → delegate
#   Inner Loop (Workers): each runs own score→improve→score loop on one module
#   Merge Gate: leader merges worker results one-by-one, reverts if score drops
#
# Usage:
#   ./harness/run.sh                     # Hybrid mode (recommended)
#   ./harness/run.sh --single            # Single agent Karpathy loop
#   ./harness/run.sh --parallel-only     # ClawTeam workers only (no leader loop)
#   ./harness/run.sh --rounds 10         # Limit outer loop rounds
#   ./harness/run.sh --target 0.9        # Stop at target score

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="hybrid"
MAX_ROUNDS=999999
TARGET_SCORE="1.0"
TEAM_NAME="tq-$(date +%H%M)"

while [ $# -gt 0 ]; do
    case $1 in
        --single)         MODE="single"; shift ;;
        --parallel-only)  MODE="parallel"; shift ;;
        --rounds)         MAX_ROUNDS="$2"; shift 2 ;;
        --target)         TARGET_SCORE="$2"; shift 2 ;;
        --team)           TEAM_NAME="$2"; shift 2 ;;
        *)                echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  quant.cpp — Hierarchical Harness"
echo "  Mode: $MODE"
echo "  Team: $TEAM_NAME"
echo "  Target: $TARGET_SCORE"
echo "============================================"

# Ensure git is initialized
if [ ! -d "$PROJECT_DIR/.git" ]; then
    git init "$PROJECT_DIR"
    git -C "$PROJECT_DIR" add -A
    git -C "$PROJECT_DIR" commit -m "Initial commit"
fi

get_score() {
    bash "$PROJECT_DIR/score.sh" --quick > /dev/null 2>&1
    cat "$PROJECT_DIR/.score" 2>/dev/null || echo "0.0000"
}

score_reached() {
    local current="$1"
    echo "$current >= $TARGET_SCORE" | bc -l 2>/dev/null | grep -q "1"
}

# ============================================================
# MODULE DEFINITIONS
# Each module can be worked on independently
# ============================================================

# Module → files it owns (workers only touch their own files)
# This prevents merge conflicts
MODULE_NAMES="foundation polar qjl turbo uniform cache simd-neon bench"

module_task() {
    local mod="$1"
    case "$mod" in
        foundation)
            echo "Create project foundation: CMakeLists.txt, include/quant/ headers (quant.h, tq_types.h, tq_spec.h), src/ directory structure. Read CLAUDE.md and docs/prd_v0.1.md for specifications. Run ./score.sh --quick after each change. Only modify: CMakeLists.txt, include/**, src/core/tq_traits.c"
            ;;
        polar)
            echo "Implement PolarQuant in src/core/tq_polar.c and tests/test_polar.cpp. Read refs/PolarQuant/models/modeling_llama_polar.py for the algorithm. Key: atan2 for angle, norm for radius, group min-max quantize, pack rho<<tbits|theta. Write Google Test unit tests. Run ./score.sh --quick after each change. Only modify: src/core/tq_polar.*, tests/test_polar.*"
            ;;
        qjl)
            echo "Implement QJL in src/core/tq_qjl.c and tests/test_qjl.cpp. Read refs/QJL/models/llama2_utils_qjl.py for the algorithm. Key: random projection matrix, sign quantization, 8-bit packing, outlier detection via L2 norm top-k, Hamming distance attention. Write Google Test unit tests. Run ./score.sh --quick after each change. Only modify: src/core/tq_qjl.*, tests/test_qjl.*"
            ;;
        turbo)
            echo "Implement quant.cpp composite in src/core/tq_turbo.c and tests/test_turbo.cpp. This combines PolarQuant (stage 1) + QJL residual correction (stage 2). Read docs/prd_v0.1.md section 6.1 FR-3. Write Google Test unit tests. Run ./score.sh --quick after each change. Only modify: src/core/tq_turbo.*, tests/test_turbo.*"
            ;;
        uniform)
            echo "Implement uniform baseline in src/core/tq_uniform.c, src/core/tq_value_quant.c, and their tests. Simple min-max quantization for 2/4-bit. Also implement value cache quantization. Run ./score.sh --quick after each change. Only modify: src/core/tq_uniform.*, src/core/tq_value_quant.*, tests/test_uniform.*, tests/test_value.*"
            ;;
        cache)
            echo "Implement paged cache in src/cache/tq_paged_cache.c and progressive compression in src/cache/tq_progressive.c with tests. Read vLLM refs/vllm/csrc/cache_kernels.cu for paged cache patterns. Run ./score.sh --quick after each change. Only modify: src/cache/**, tests/test_paged_cache.*, tests/test_progressive.*"
            ;;
        simd-neon)
            echo "Implement ARM NEON optimized kernels in src/backend/cpu/tq_neon.c. Also write generic fallback in src/backend/cpu/tq_generic.c and dispatch in src/backend/cpu/tq_cpu_dispatch.c. Read refs/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c for NEON patterns. Write AVX2 stubs in src/backend/cpu/tq_avx2.c (compile-guarded). Run ./score.sh --quick after each change. Only modify: src/backend/cpu/**"
            ;;
        bench)
            echo "Create benchmarks: bench/tq_bench.cpp and bench/tq_quality.cpp. tq_bench must output: quantize_throughput=N, attention_throughput=N, compression_ratio=N, simd_speedup=N. tq_quality must output: roundtrip_mse=N, attention_cosine=N, cross_platform=pass/fail. Also create spec/ documents and test vectors. Run ./score.sh after each change. Only modify: bench/**, spec/**, tests/reference/**"
            ;;
    esac
}

# Modules that can run in parallel (no shared files)
PARALLEL_GROUP_1="foundation"
PARALLEL_GROUP_2="polar qjl uniform"          # after foundation
PARALLEL_GROUP_3="turbo cache simd-neon bench" # after group 2

# ============================================================
# SINGLE AGENT LOOP (Pure Karpathy)
# ============================================================
run_single() {
    local round=0
    local prev_score=$(get_score)

    while [ "$round" -lt "$MAX_ROUNDS" ]; do
        round=$((round + 1))
        echo ""
        echo "===== ROUND $round | score: $prev_score | target: $TARGET_SCORE ====="

        claude --print \
            --allowedTools "Edit,Write,Read,Bash,Glob,Grep" \
            --max-turns 30 \
            -p "Read program.md. Run ./score.sh --quick. Find the next unchecked WBS item in docs/wbs_v0.1.md. Implement it. Run ./score.sh --quick to verify. Mark [x] in WBS. Commit. Say ROUND_COMPLETE when done." \
            2>&1 | tee "$PROJECT_DIR/.logs/round_${round}.log"

        local new_score=$(get_score)
        if echo "$new_score < $prev_score" | bc -l 2>/dev/null | grep -q "1"; then
            echo "SCORE DROPPED $prev_score → $new_score — reverting"
            git checkout -- . 2>/dev/null || true
        else
            echo "OK: $prev_score → $new_score"
            prev_score="$new_score"
        fi

        score_reached "$new_score" && echo "TARGET REACHED: $new_score" && break
    done

    bash "$PROJECT_DIR/score.sh"
}

# ============================================================
# SPAWN WORKERS FOR A MODULE GROUP (ClawTeam)
# ============================================================
spawn_group() {
    local group_name="$1"
    shift
    local modules="$@"

    echo ""
    echo "--- Spawning group: $group_name ($modules) ---"

    for mod in $modules; do
        local task_desc=$(module_task "$mod")

        echo "  Spawning worker: $mod"
        clawteam spawn \
            --team "$TEAM_NAME" \
            --agent-name "$mod" \
            --workspace \
            --repo "$PROJECT_DIR" \
            --task "$task_desc" \
            2>&1 || echo "  Warning: spawn $mod may have failed"
    done
}

# ============================================================
# MERGE GATE: merge worker results one-by-one with score check
# ============================================================
merge_with_gate() {
    local modules="$@"
    local base_score=$(get_score)
    echo ""
    echo "--- Merge Gate (base score: $base_score) ---"

    for mod in $modules; do
        local worktree_branch="clawteam/${TEAM_NAME}/${mod}"

        # Check if branch exists
        if ! git rev-parse --verify "$worktree_branch" > /dev/null 2>&1; then
            echo "  [$mod] No branch found — skipping"
            continue
        fi

        # Save state before merge
        local pre_merge=$(git rev-parse HEAD)

        echo "  [$mod] Merging..."
        if git merge "$worktree_branch" --no-edit -m "Merge $mod worker results" 2>/dev/null; then
            local new_score=$(get_score)

            if echo "$new_score < $base_score" | bc -l 2>/dev/null | grep -q "1"; then
                echo "  [$mod] SCORE DROPPED $base_score → $new_score — REVERTING merge"
                git reset --hard "$pre_merge" 2>/dev/null
            else
                echo "  [$mod] OK: $base_score → $new_score"
                base_score="$new_score"
            fi
        else
            echo "  [$mod] Merge conflict — skipping (needs manual resolution)"
            git merge --abort 2>/dev/null || true
        fi
    done

    echo "  Final score after merge gate: $base_score"
}

# ============================================================
# HYBRID MODE: Leader Karpathy loop + Worker ClawTeam parallelism
# ============================================================
run_hybrid() {
    local round=0
    local prev_score=$(get_score)

    mkdir -p "$PROJECT_DIR/.logs"

    echo ""
    echo "Phase 0: Initialize team"
    clawteam team spawn-team "$TEAM_NAME" \
        -d "quant.cpp development" \
        2>/dev/null || true

    while [ "$round" -lt "$MAX_ROUNDS" ]; do
        round=$((round + 1))
        prev_score=$(get_score)

        echo ""
        echo "============================================"
        echo "  OUTER LOOP ROUND $round"
        echo "  Current score: $prev_score"
        echo "  Target: $TARGET_SCORE"
        echo "============================================"

        # Decide which group to work on based on current state
        local group_modules=""

        if echo "$prev_score < 0.05" | bc -l 2>/dev/null | grep -q "1"; then
            # Nothing exists yet — do foundation first (single agent, sequential)
            echo "Phase: Foundation (sequential)"
            claude --print \
                --allowedTools "Edit,Write,Read,Bash,Glob,Grep" \
                --max-turns 50 \
                -p "$(module_task foundation). Also read program.md and CLAUDE.md first." \
                2>&1 | tee "$PROJECT_DIR/.logs/round_${round}_foundation.log"

        elif echo "$prev_score < 0.30" | bc -l 2>/dev/null | grep -q "1"; then
            # Foundation done — spawn parallel algorithm workers
            echo "Phase: Core Algorithms (parallel: polar, qjl, uniform)"
            spawn_group "algorithms" $PARALLEL_GROUP_2
            echo "Waiting for workers to complete (timeout: 30min)..."
            clawteam task wait "$TEAM_NAME" --timeout 1800 2>/dev/null || true
            merge_with_gate $PARALLEL_GROUP_2

        elif echo "$prev_score < 0.60" | bc -l 2>/dev/null | grep -q "1"; then
            # Core algorithms done — spawn next parallel group
            echo "Phase: Advanced (parallel: turbo, cache, simd, bench)"
            spawn_group "advanced" $PARALLEL_GROUP_3
            echo "Waiting for workers to complete (timeout: 30min)..."
            clawteam task wait "$TEAM_NAME" --timeout 1800 2>/dev/null || true
            merge_with_gate $PARALLEL_GROUP_3

        else
            # High score — fine-tuning with single agent (precision over speed)
            echo "Phase: Fine-tuning (single agent, careful iteration)"
            claude --print \
                --allowedTools "Edit,Write,Read,Bash,Glob,Grep" \
                --max-turns 30 \
                -p "Read program.md. Run ./score.sh. Focus on the LOWEST scoring dimension. Improve it. Run ./score.sh to verify. Commit. Say ROUND_COMPLETE when done." \
                2>&1 | tee "$PROJECT_DIR/.logs/round_${round}_finetune.log"
        fi

        # Score check
        local new_score=$(get_score)
        echo ""
        echo "Round $round result: $prev_score → $new_score"

        score_reached "$new_score" && echo "TARGET $TARGET_SCORE REACHED!" && break
    done

    echo ""
    echo "=== Final Report ==="
    bash "$PROJECT_DIR/score.sh"
}

# ============================================================
# PARALLEL ONLY MODE
# ============================================================
run_parallel() {
    clawteam team spawn-team "$TEAM_NAME" -d "quant.cpp" 2>/dev/null || true
    spawn_group "all" $MODULE_NAMES
    echo ""
    echo "All workers spawned. Monitor with:"
    echo "  clawteam board attach $TEAM_NAME"
    echo "  watch -n 30 bash ./score.sh --quick"
    clawteam board attach "$TEAM_NAME" 2>/dev/null || true
}

# ============================================================
# MAIN
# ============================================================
case "$MODE" in
    single)   run_single ;;
    parallel) run_parallel ;;
    hybrid)   run_hybrid ;;
    *)        echo "Unknown mode: $MODE"; exit 1 ;;
esac
