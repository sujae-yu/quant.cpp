# TTFT Daily Driver Baseline — 16 GB M1 Pro (2026-04-20)

Round 51 introduced `TTFT | decode` split output in the CLI, separating
prefill latency from sustained decode rate. This page establishes the
honest daily-driver numbers an individual developer should expect.

**Hardware**: M1 Pro, 16 GB unified memory, macOS (CPU-only, `TQ_NO_METAL=1 TQ_NO_MLOCK=1`)
**Build**: Release (default flags), 8 worker threads
**Prompt**: `"Once upon a time"` with `-n 30 -T 0` (greedy, 30-token cap)
**KV cache**: `turbo_kv_4b` (default, 7× compression)

## Matrix

| Model | File | First run (cold) | Steady warm |
|---|---:|---|---|
| **Phi-3.5-mini Q4_K_M** (3.8B) | 2.4 GB | TTFT **4.14s**, decode **14.3 t/s** | TTFT **2.3s**, decode **14.5 t/s** |
| **Llama-3.2-3B Q8→Q4** (3B, load-time Q4) | 3.2 GB | TTFT ~1.5s, decode ~22 t/s† | TTFT **0.97s**, decode **29.0 t/s** |
| **Qwen3.6-35B-A3B-UD-IQ4_XS** (35B MoE K=8) | 16.5 GB | TTFT **9.6s**, decode **3.0 t/s** | TTFT **1.83s**, decode **10.5 t/s** |

† First run on Llama-3.2-3B had early EOS (5 tokens) — numbers
extrapolated from partial sample + prior-session matrix.

## Reading the numbers

**TTFT (prefill)**: time from `tq_generate` call to first token emit.
Dominated by model load on cold runs (mmap + MADV traversal +
transformer pass #1), drops to real prefill cost once page cache is
warm.

**Decode (steady)**: sustained token-per-token rate. This is the
engine's actual compute speed — unchanged between cold and warm
(modulo hot/cold expert paging on Qwen3.6 MoE).

**Overall**: n/total_time. Includes TTFT. For short queries (n=30)
this conflates cold start with decode speed — **misleading for
individual-dev UX assessment**. Previous bench reports collapsed to
this single metric; Round 51 splits it out.

## Why this matters for daily-driver selection

| Use case | Best pick | Why |
|---|---|---|
| **Snappy short chat** (1–3 turns, <50 tok answers) | Phi-3.5 Q4_K_M | Warm TTFT 2.3s + decode 14.5 t/s = 4-5s to complete 30-token answer after second call |
| **Quick code/math** (one-shot, <15 tok) | Llama-3.2-3B Q8→Q4 | Fastest pure decode 29 t/s, <1s TTFT warm |
| **Quality long-form** (60+ tok, complex reasoning) | Qwen3.6 IQ4_XS | 35B MoE quality, warm TTFT <2s, decode 10.5 t/s = 6s for 60 tokens |
| **Only when quality must win** | Qwen3.6 Q5_K_M | Decode ~7.9 t/s warm, file 24.6 GB, best quality per existing Round 20 bench |

## Warm-up advice for individual devs

1. **First run is always slower** — mmap of a 16 GB GGUF touches
   cold SSD pages; Qwen3.6 cold TTFT is 9.6s vs warm 1.8s (5.3×).
2. **Running the same command twice** is the simplest warmup — after
   the first call, macOS keeps the MoE hot-subset in page cache and
   subsequent calls see the warm numbers.
3. **Switching models** evicts the previous model's pages — if you
   alternate, each switch pays the cold penalty.
4. **Use `./build/quant-server`** if you want a persistent warm
   state — HTTP mode keeps the model resident across requests.

## Methodology reproducibility

```bash
# 1. Clean the page cache (optional, for cold measurement):
sudo purge    # macOS — requires admin

# 2. Run the benchmark:
for run in 1 2 3; do
  echo "=== $run ==="
  TQ_NO_METAL=1 TQ_NO_MLOCK=1 \
    ./build/quant MODEL.gguf -p "Once upon a time" -n 30 -T 0 \
    2>&1 | grep TTFT
done

# Example output:
# TTFT 4.14s | decode 29 tok in 2.03s (14.3 tok/s) | total 6.2s (4.9 tok/s overall)
# TTFT 2.31s | decode 29 tok in 1.97s (14.7 tok/s) | total 4.3s (7.0 tok/s overall)
# TTFT 2.28s | decode 29 tok in 2.01s (14.4 tok/s) | total 4.3s (7.0 tok/s overall)
#       ^^^^^^^                                ^^^^^^^^
#       cold→warm                              decode stable across all 3
```

The key insight: **decode rate is a model property, TTFT is a warmup
property**. Reporting them separately prevents the common "engine is
slow" misdiagnosis that actually means "first run is cold."

## Related

- `bench/results/2026-04-19_qwen36_quant_matrix_16gb.md` — 5-tier
  Qwen3.6 quant comparison (IQ2 → Q5_K_M)
- `bench/results/2026-04-20_q4kxs_vs_q5km_iq4xs.md` — IQ4_XS wins
  4/5 formats vs Q5_K_M daily-driver
- `docs/getting-started.md` — CLI usage walkthrough
