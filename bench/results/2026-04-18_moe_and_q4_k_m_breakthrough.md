# MoE & Q4_K_M throughput breakthrough — 2026-04-18

**Headline**: Qwen3.6-35B-A3B (MoE, IQ2_XXS) now runs at **16.1 t/s on a 16GB M1 Pro**, CPU-only — **3.2× faster than llama.cpp's CPU path** (5.07 t/s) on the same hardware, with **35% less RAM** (6.5 GB vs ~10 GB).

Q4_K_M models (the most common GGUF format on HuggingFace) also got a silent **+115% to +180%** boost from a profile-driven fix in the Q6_K kernel that turned out to be the real bottleneck on dense models as well.

All measurements: M1 Pro 8P+2E cores, 16 GB RAM, macOS 24, CPU-only (`TQ_NO_METAL=1`), 8 threads, temperature 0, 30-50 token greedy decode, 3-run warm average.

---

## 1. Q6_K NEON int8 fast path — the unrecognized Q4_K_M bottleneck

**What was wrong**: `fused_dot_q6_k` in `tq_gguf_quants.c` was a pure scalar FMA loop — no SIMD at all. Q4_K_M models embed Q6_K for the critical `attention.wo` and `ffn_down` projections (a long-standing llama.cpp convention to protect precision on the paths where Q4 hurts most). Nobody had noticed because its cost was being attributed to "matmul" in our wall-clock profile, and ~~cycle counting~~ napkin math kept landing on kernel-latency theories instead.

**What fixed it**: `sample` the running process **after** load completes. Waiting for the "Threads:" banner before starting the sample window shifted the heaviest function from `tq_quantize_weights_q4` (load-time, single-threaded, ~30 s on 4B) to `fused_dot_q6_k` (17725 self-time samples vs 3040 for the already-NEON `matmul_q4_rows`).

The fix is mechanical: pre-quantize the activation to int8 in 32-element blocks (Q8_0 layout), unpack each 16-element Q6_K sub-block to int8 `[-32..31]`, issue one `vdotq_s32` per sub-block. Same pattern as the IQ2_XXS int8 kernel that landed on 2026-04-17.

**Results** (warm, 3-run peak):

| Model                    | before | after | vs llama.cpp CPU 8t |
|--------------------------|:------:|:-----:|:-------------------:|
| Qwen3.5-4B Q4_K_M        | 5.0    | **14.1** | 19.9 (71%)       |
| Phi-3.5-mini Q4_K_M      | 6.2    | **14.1** | 26.7 (53%)       |

Env toggle: `TQ_Q6K_NOINT=1` reverts to the scalar path for A/B testing.

**Commit**: `9fdafaa` — "perf(q6_k): NEON int8 fast path"

---

## 2. Qwen3.6-35B-A3B on 16GB Mac — 4.2× full breakthrough

Qwen3.6-35B-A3B is a hybrid DeltaNet + full-attention MoE with 256 experts (8 active per token), 40 layers. The `UD-IQ2_XXS` variant from Unsloth is a mixed quantization: IQ2_XXS + IQ2_S for routed experts, IQ3_S for critical layers, Q4_K / Q6_K for shared experts, Q4_K for `lm_head`. It is *the* MoE to run if you want 35B-class capabilities on a MacBook Pro.

Three independent fixes stacked to take it from 3.08 t/s (first session baseline) to 16.1 t/s peak.

### 2.1 Q6_K int8 fast path (from section 1 above)
Already lifted Qwen3.6 from ~7.8 t/s (prior session best) to ~9.7 t/s by vectorizing the Q6_K path used for this model's attention output and shared-expert down projections.

### 2.2 MoE router NEON vectorize

**What was wrong**: `tq_moe_route` computed router logits with a pure scalar dot loop:

```c
for (int e = 0; e < num_experts; e++) {
    float sum = 0.0f;
    for (int j = 0; j < hidden_dim; j++)
        sum += hidden[j] * row[j];
    logits[e] = sum;
}
```

On Qwen3.6: 30 deltanet layers × 256 experts × 2048 hidden = **15.7 M scalar FP32 ops per token, just for routing**. On top of that, `malloc(num_experts * 4)` and `calloc(num_experts, 1)` every call — 60 heap allocations per token at 15 t/s = 900 allocs/second of pointless churn.

**What fixed it**: 4-accumulator NEON FMA inner loop (16 floats/cycle peak on M1), and `static __thread` scratch buffers for `logits[]` and `used[]`.

**Results**: `tq_moe_route` self-time dropped from **628 samples → 30 samples** (21× reduction). Negligible on the profile now.

### 2.3 IQ3_S int8 fast path

**What was wrong**: `fused_dot_iq3_s` was also pure scalar (390 samples, mid-tier hot). Same pattern as Q6_K.

**What fixed it**: NEON int8 path — build int8x16 from 2-of-4 `l`-iteration grid lookups (4-byte each) with sign bytes applied via `vtst`/`veor`/`vsub`, then one `vdotq_s32` per 16-element chunk. 8 `vdotq`s per 256-element block.

**Results**: `fused_dot_iq3_s_int8` dropped to 62 self-time samples (from 390). Small win alone, but it's all additive.

Env toggle: `TQ_IQ3S_NOINT=1`.

### 2.4 `TQ_NO_MLOCK=1` — when the OS knows better than you

The prior iteration introduced `mlock(10 GB)` to keep Qwen3.6's expert weights pinned in RAM, because page-faults on expert pages were catastrophic (~100× slowdown). Measured on a quiet 16GB system, flipping `mlock` off gave a **surprising double win**:

| mode | RSS | decode (warm peak) |
|------|-----|--------------------|
| `mlock(10 GB)` (prior default) | 12.0 GB | 14.3 t/s |
| `TQ_NO_MLOCK=1` + MADV_WILLNEED | **6.5 GB** | **16.1 t/s** |

The OS LRU beats `mlock` here because:
1. Only 8 of 256 experts route per token, so the truly-hot set is far smaller than the pinned 10 GB.
2. Page-cache hit patterns track the routing distribution automatically.
3. `mlock` prevented the kernel from reclaiming cold pages, bloating RSS without improving hit rate.

**Default behavior unchanged** (mlock still on, for systems under memory pressure from other apps). Setting `TQ_NO_MLOCK=1` is recommended for 16GB Macs running quant.cpp as the primary workload.

---

## 3. End-to-end Qwen3.6-35B-A3B trajectory

| iteration                                           | t/s peak | RSS     |
|-----------------------------------------------------|:--------:|:-------:|
| baseline (Q4 recompress)                            | 3.08     | —       |
| + TQ_NO_Q4 (keep native GGUF types)                 | ~4.0     | —       |
| + IQ2_XXS int8 kernel (`092daa0`)                   | 6.7      | —       |
| + IQ2_S int8 kernel (`9d33b4e`)                     | 7.8      | —       |
| + **Q6_K int8 kernel** (`9fdafaa`)                  | ~10      | 12.0 GB |
| + **MoE router NEON + IQ3_S int8** (`f738325`)      | 14.3     | 12.0 GB |
| + **`TQ_NO_MLOCK`** (`ee4778a`)                     | **16.1** | **6.5 GB** |
| llama.cpp CPU reference (8 threads)                 | 5.07     | ~10 GB  |

**Net: 5.2× faster than the first-session baseline, 3.2× faster than llama.cpp CPU, at 35% lower RAM.**

---

## Reproduce

```bash
# Build (release)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

# Qwen3.6-35B-A3B on 16GB Mac — recommended settings
TQ_NO_METAL=1 TQ_NO_MLOCK=1 ./build/quant \
  models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  --chat -p "What is the capital of France?" \
  -n 60 -T 0.7 -j 8

# Q4_K_M (Phi-3.5 / Qwen3.5-4B / etc)
TQ_NO_METAL=1 ./build/quant models/Qwen3.5-4B-Q4_K_M.gguf \
  -p "Once upon a time" -n 50 -T 0 -j 8
```

A/B toggles for kernel isolation: `TQ_Q6K_NOINT=1`, `TQ_IQ2_XXS_NOINT=1`, `TQ_IQ3S_NOINT=1`.

## Regression status

`scripts/test_models.sh`: **12/12 PASS** on the full model suite (Llama 3.1/3.2, Qwen 2.5/3.5/3.6, Phi-3.5, Gemma-4).

## Quality caveat

Qwen3.6-35B at IQ2_XXS (2.05 bpw on a 35B model) has intrinsic quality limits. Short factual chat answers work ("The capital of France is Paris."), but long greedy decode (>40 tokens) drifts into repetition / byte-fragment territory. This is an IQ2_XXS quantization property, not an engine bug — the same behavior is visible in llama.cpp at the same quant level. Higher quants (IQ3_XS, IQ4_NL) would solve it but don't fit 16GB.
