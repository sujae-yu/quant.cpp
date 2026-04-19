#!/usr/bin/env bash
#
# bench_my_mac.sh — 30-second daily-driver readiness check
#
# Runs a warm TTFT + decode bench on whichever supported GGUF files
# you already have in ./models/ and prints a small report. No network
# access, no downloads — uses what's there.
#
# Usage:  bash scripts/bench_my_mac.sh
# Prereq: cmake build complete (./build/quant must exist)

set -euo pipefail

BIN="./build/quant"
MODELS_DIR="${MODELS_DIR:-./models}"
PROMPT="Once upon a time"
N_TOK=30

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN not found — run 'cmake --build build -j' first." >&2
  exit 1
fi

# Candidate daily-driver models, in order of preference.
# Absent files are skipped silently.
CANDIDATES=(
  "Phi-3.5-mini-instruct-Q4_K_M.gguf"
  "Llama-3.2-3B-Instruct-Q8_0.gguf"
  "Qwen3.5-4B-Q4_K_M.gguf"
  "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
  "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
  "Llama-3.2-1B-Instruct-Q8_0.gguf"
  "gemma-4-e2b-it-Q8_0.gguf"
)

found=()
for m in "${CANDIDATES[@]}"; do
  if [[ -f "$MODELS_DIR/$m" ]]; then
    found+=("$m")
  fi
done

if [[ ${#found[@]} -eq 0 ]]; then
  cat <<EOF >&2
error: no candidate GGUF found in $MODELS_DIR/

Put any of the following there to run a bench:
$(printf '  - %s\n' "${CANDIDATES[@]}")

Recommended first download for 16 GB Mac daily driver:
  Phi-3.5-mini-instruct-Q4_K_M.gguf (~2.4 GB, fast & coherent)
EOF
  exit 2
fi

echo "== bench_my_mac — quant.cpp daily-driver readiness =="
echo "Hardware: $(uname -sm), $(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)"GB"}') RAM, $(sysctl -n hw.ncpu) cores"
echo "Prompt:   \"$PROMPT\" (-n $N_TOK -T 0)"
echo "KV:       turbo_kv_4b (default, 7x compression)"
echo ""
printf "%-50s %-14s %-18s\n" "Model" "Warm TTFT" "Warm decode t/s"
printf "%-50s %-14s %-18s\n" "-----" "---------" "---------------"

for m in "${found[@]}"; do
  short="${m%.gguf}"
  short="${short:0:48}"

  # Cold run (warmup) — discard
  TQ_NO_METAL=1 TQ_NO_MLOCK=1 "$BIN" "$MODELS_DIR/$m" \
    -p "$PROMPT" -n $N_TOK -T 0 >/dev/null 2>/tmp/bench_my_mac_cold.$$ || true

  # Warm run — capture
  TQ_NO_METAL=1 TQ_NO_MLOCK=1 "$BIN" "$MODELS_DIR/$m" \
    -p "$PROMPT" -n $N_TOK -T 0 >/dev/null 2>/tmp/bench_my_mac_warm.$$ || true

  line=$(grep -E "TTFT.*decode" /tmp/bench_my_mac_warm.$$ 2>/dev/null | tail -1 || echo "")
  if [[ -z "$line" ]]; then
    printf "%-50s %-14s %-18s\n" "$short" "n/a" "n/a"
    continue
  fi

  ttft=$(echo "$line" | sed -nE 's/.*TTFT ([0-9.]+s).*/\1/p')
  decode=$(echo "$line" | sed -nE 's/.*decode [0-9]+ tok in [0-9.]+s \(([0-9.]+) tok\/s\).*/\1/p')
  printf "%-50s %-14s %-18s\n" "$short" "$ttft" "${decode} t/s"
done

rm -f /tmp/bench_my_mac_cold.$$ /tmp/bench_my_mac_warm.$$

cat <<'EOF'

Interpretation:
  TTFT   = time-to-first-token after cold load was already paid (warm run).
           Includes prefill compute + any cold-page paging for routed experts.
  Decode = sustained generation rate once streaming starts.

See bench/results/2026-04-20_ttft_daily_driver.md for the published
reference matrix on 16 GB M1 Pro CPU-only.
EOF
