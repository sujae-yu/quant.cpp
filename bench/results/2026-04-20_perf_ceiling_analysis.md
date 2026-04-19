# Performance Ceiling Analysis — R56-R59 Findings (2026-04-20)

## Context

User directed a 20-round perf optimization loop. After 4 rounds of
kernel-level attempts on Qwen3.6-35B-A3B decode, none delivered
measurable speed gains. This document records what was tried, why
each failed, and where the real ceiling is.

## The 4 Attempts

| R | Target | Result | Root cause |
|---|---|---|---|
| 56 | IQ4_XS 2-row inner unroll (activation reuse) | **-29%** reverted | Register pressure + memory-bound kernel; mirror of R18 Q5_K failure |
| 57 | `tq_matmul_q8` vdotq_s32 int8×int8 path | Dormant, reverted | Qwen3.6/Phi/Llama all go through `tq_matmul_gguf` (GGUF path); internal `.q8` format unused |
| 58 | IQ4_XS prefetch distance 1 → 2 blocks | Noisy, reverted | HW prefetch already covers 1-block lookahead; no benefit |
| 59 | MoE accumulation scalar → NEON vmlaq_f32 | Kept (tiny, safe) | 655K scalar FMAs/tok → NEON. Measurement dominated by noise but code is strictly cleaner. |

## Profile evidence (Qwen3.6-35B-A3B IQ4_XS, warm decode)

```
[Profile 20 tok]  matmul=20% recurrent=2% moe=88% conv=0.2% attn=0.1%
  matmul sub:  lm_head=5.6ms  attn_QKVO=5.0ms  delta_QKVO=22.9ms  rest=0ms
```

MoE kernel takes **~64 ms of the 73 ms/token budget**. Per token, the
MoE router selects 8 experts out of 256; each expert has 3 matmuls
(gate, up, down) × 40 layers = 960 matmuls/token. Expert weights are
mmap'd from a 16.5 GB file on a 16 GB Mac — most expert pages are
**cold** on each token because the routing picks a different subset.

Measured weight bandwidth consumed per token: ~150 MB × 13.7 t/s ≈
**~2 GB/s effective**, which is consistent with M1 Pro's random-access
DRAM bandwidth when coming through mmap + page cache.

## Why kernel micro-opts don't help

The NEON kernels (IQ4_XS, Q5_K, Q4_K, Q8_0) are all compute-capable
of ~8-12 GB/s of weight throughput at 3 GHz. Memory is the choke
point at ~2 GB/s effective. Extra ILP, tighter prefetch, and register-
budget games all target compute — which isn't the bottleneck.

**Empirical precedent**: every 2-row k-quant attempt on this codebase
has regressed (R18 Q5_K -14%, R56 IQ4_XS -29%). Saved as
`feedback_2row_kquant_moe_fails.md`.

## Where the real headroom is

### 1. Batched prefill for Phi-3.5 (not-yet-enabled)

Phi-3.5-mini-Q4_K_M TTFT on a 40-word prompt: **6.1 s**.
Llama-3.2-3B-Q8→Q4 on the same prompt: **0.52 s** (12× faster).

Cause: `tq_forward_batch` bails for Phi-3.5 because:
- `c->has_fused_qkv` is true (Phi-3's fused weight layout)
- `!model->use_q4_weights` — Phi-3.5 Q4_K_M loads as GGUF Q4_K, not internal Q4

Fix: Phi-3-specific batched driver similar to `tq_forward_batch_moe_hybrid`
for Qwen3.6. Estimated **400-600 LOC**, one focused multi-session project.

### 2. Speculative decoding / n-gram lookup

For N=1 autoregressive decode, the 150 MB/token weight fetch is
unavoidable. Speculative decoding lets a small draft model propose
several tokens, verified by the big model in a batch — amortizes the
fetch 3-5× on common continuations.

### 3. Aggressive quantization on 16 GB Mac

IQ3_XXS 12.3 GB (working set ~6.5 GB) decoded at **14.6 t/s warm** in
earlier session (`project_q3_breakthrough.md`). Quality is worse but
speed is already higher. The "quality-speed" frontier is well-mapped
in `bench/results/2026-04-19_qwen36_quant_matrix_16gb.md`.

## Honest takeaway for the user

**Qwen3.6-35B-A3B IQ4_XS at 13.7 t/s warm on 16 GB M1 Pro CPU-only
is at the memory ceiling for this hardware.** Further decode speedup
is not available through kernel work on this workload.

Remaining perf opportunities are architectural (Phi-3.5 batched
prefill, speculative decoding) and belong to multi-session missions,
not a Karpathy-style tight loop.

## Ledger

- Committed: R59 (MoE accum NEON vectorization, safe, no regression)
- Reverted: R56, R58 (negative + noisy)
- Meta-committed: R57 (dormant, for learning)
- Regression: 15/15 PASS throughout

## Recommendation

Stop the perf loop at R59. Open a follow-up multi-session project for
Phi-3.5 batched prefill (biggest visible opportunity: 6.1 s → ~1.5 s
TTFT on 40-word prompts, 4× improvement).
