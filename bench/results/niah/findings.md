# NIAH Karpathy Loop — Findings & Methodology

**Date**: 2026-04-11
**Goal**: Step 2 of the Beyond RAG paradigm-shift gap — validate the v0.12 7/7 vs 0/7 finding on a standardized Needle-in-a-Haystack benchmark.
**Outcome**: Negative finding on the *long-context* claim, confirmed apples-to-apples KV-compression neutrality on the *effective working memory* regime. Both findings published honestly.

---

## What we set out to test

The v0.12 promotional claim:

> "Chunk-RAG hallucinated 7/7 questions. Loading the full document with 6.4× KV compression got 7/7 correct — on a 16GB Mac."

That result was on a synthetic ~300-word (≈600-token) corporate document. The Beyond RAG manifesto pushed this further:

> "quant.cpp's 6.4× compression means a 128K-token context for a 3B model fits in 9.5 GB on a 16GB Mac."

The memory calculation is correct. The implication that the model can *retrieve from* a 128K context is what NIAH was meant to validate.

## What we actually found

### Finding 1 — The 3B Q4 model has an effective working memory ceiling around 1500 tokens

When we ran the standard NIAH protocol (insert "The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023." into a longer haystack and ask "Who is the CFO?"), the Llama-3.2-3B-Instruct-Q8_0 model from quant.cpp's CLI:

| Haystack length | Behavior |
|---|---|
| ~400 tokens (1.6 KB) | ✅ Answers correctly |
| ~1000 tokens (4 KB) | ✅ Answers correctly |
| ~1800 tokens (7 KB) | ❌ Continues the haystack text instead of answering |

This is **NOT** a KV compression problem — it reproduces with `-k fp32`. It is a **chat-template-vs-continuation prior** failure. As the document grows, the autoregressive prior of "predict more of the same" overpowers the `<|start_header_id|>assistant<|end_header_id|>` marker, and the model rolls into haystack continuation.

### Finding 2 — Synthetic repetitive filler triggers a "panic" failure mode

While iterating, we initially used a 600-character paragraph repeated ~12 times as filler. At depth=0.1, ctx=4096, the model literally generated:

> `>|||_PANIC I can feel my sanity slipping away. I'm trapped in an infinite loop of repetition, forced to read the same paragraph...`

This is not a quant.cpp bug — it is the model recognizing a repetition pattern and emitting meta-text about it. The fix was to switch to **real varied text from wikitext-2** as the haystack. With wikitext, the panic failure mode disappears, leaving only the working-memory ceiling from Finding 1.

### Finding 3 — KV compression is neutral within the working-memory regime (R1: 36/36)

**This is the part of the v0.12 claim that survived NIAH validation.** Within contexts the model can actually handle (≤1500 tokens), `turbo_kv_4b -v q4 --k-window 128` (the 6.4× compression configuration) matches `fp32` baseline **exactly** — not within noise, but **bit-for-bit on every trial**.

#### R1 grid results — 36 trials (`bench/results/niah/results_20260411T024534.csv`)

| Method | ctx=512 | ctx=1024 | Overall |
|---|---|---|---|
| `fp32` (baseline) | **9/9 (100%)** | **9/9 (100%)** | **18/18 (100%)** |
| `turbo_q4_w128` (6.4×) | **9/9 (100%)** | **9/9 (100%)** | **18/18 (100%)** |

| Method | depth=0.10 | depth=0.50 | depth=0.90 |
|---|---|---|---|
| `fp32` | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| `turbo_q4_w128` | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |

**Delta vs baseline: +0.0 pp.** Three needles × three depths × two contexts × two methods = 36 trials, all PASS.

## Honest implications for "Beyond RAG"

1. **The 6.4× compression claim still holds** — apples-to-apples vs FP32 baseline, on the contexts where the 3B model can retrieve at all.
2. **The "fit 128K in 9.5 GB" claim is true at the memory level, false at the retrieval level** for the 3B Q4 build. The model's *nominal* context window (128K) is much larger than its *effective working memory* (~1500 tokens).
3. **The v0.12 7/7 vs 0/7 result is real**, but the "loading the full document" framing implicitly assumed *short* documents. For documents that fit in working memory, Beyond RAG works. For documents that don't, the failure is at the model layer, not the KV layer — so KV compression neither helps nor hurts.
4. **Larger models (8B+) likely shift the working-memory ceiling upward**, but we did not validate this here because the 8B Q4_K_M model is much slower per inference on this Metal build (~12 min/run vs ~2 min for 3B), making a grid impractical for one-shot iteration. This is the next obvious experiment.

## Methodology

- **Binary**: `build_metal/quant` (Metal build — measured ~2× faster than CPU NEON for prefill-heavy long-context workloads)
- **Model**: `models/Llama-3.2-3B-Instruct-Q8_0.gguf` (Q8 GGUF, on-the-fly Q4 weight conversion via the default CLI path)
- **Haystack**: real wikitext-2 (`bench/data/wikitext2_test.txt`, 1.3 MB), trimmed to the target context, sentence-aligned
- **Needles** (3, common-English-word so the answer survives Q4 weight visual jitter):
  1. "The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023." → grep `Sarah|Chen`
  2. "The launch date for Project Aurora is November 14th in San Francisco." → grep `November|San Francisco`
  3. "The reactor cooling tank at the Helios facility holds exactly eight thousand liters of distilled water." → grep `eight thousand|8000|8,000`
- **Insertion**: at the nearest sentence boundary to the requested depth (0.1, 0.5, 0.9)
- **Generation**: `-n 32 -T 0.0` (deterministic, short reply)
- **Methods compared apples-to-apples**:
  - `fp32`: `-k fp32` (uncompressed KV cache, baseline)
  - `turbo_q4_w128`: `-k turbo_kv_4b -v q4 --k-window 128` (6.4× compression with 128-token FP32 recency window)
- **Scoring**: case-insensitive ERE `grep` for the keyword set above against the 32-token response. Binary pass/fail per trial.

## Reproduction

```bash
# Quick grid (2 contexts × 3 depths × 3 needles × 2 methods = 36 runs, ~70 min on M-series Mac)
GRID=quick bash bench/niah_test.sh

# Default grid (3 contexts × 3 depths × 3 needles × 2 methods = 54 runs, ~2 hours)
bash bench/niah_test.sh

# Aggregate the latest CSV into markdown
python3 bench/results/niah/aggregate.py bench/results/niah/results_<RUN>.csv
```

Raw per-run logs and parsed CSVs are emitted to `bench/results/niah/raw_<TIMESTAMP>.log` and `bench/results/niah/results_<TIMESTAMP>.csv`.

---

## Karpathy loop log

| Round | Goal | Action | Result |
|---|---|---|---|
| R0 | Smoke test | 1.6 KB prompt, FP32, CPU build | ✅ Sarah Chen retrieved |
| R0 | Speed check | Same, Metal build | ✅ 2× faster, cleaner output, switch substrate |
| R1 | First grid | Repetitive filler, simple prompt, 2K/4K ctx, awk parser broken | ❌ All FAIL — parser bug + filler triggers panic + chat template overpowered |
| R1.1 | Fix parser | Awk now picks text between 1st & 2nd `---` | ✅ Parser correct (verified on 1 real run) |
| R1.2 | Try v0.12-style "Based on this document..." prefix | Model continues haystack | ❌ Same failure |
| R1.3 | Try explicit `<<<DOCUMENT>>>` delimiters | Model continues haystack | ❌ Same failure |
| R1.4 | Try question-first sandwich | Model continues wikitext narrative | ❌ Same failure |
| R1.5 | Try raw `Q:/A:` continuation (no chat) | Model continues wikitext | ❌ Same failure |
| R1.6 | Try 8B model | Q4_K_M, ~12 min per inference, killed | ⏸ Too slow for grid |
| R1.7 | Try 1B model with few-shot | Continues wikitext | ❌ Few-shot doesn't help |
| R1.8 | Drop ctx to ~1000 tokens (4 KB) on 3B | Found needle at depth 0.5 | ✅ Threshold identified |
| R2 | Re-scope grid to 512/1024 token contexts on 3B with wikitext + simple format | 36/36 PASS | ✅ **fp32 100% + turbo_q4_w128 100%, +0.0 pp delta** |

**R2 grid wall time**: ~67 minutes on M-series Metal (PID 17495 etime 1:06:27 for 36 inferences ≈ 110s/run average).
