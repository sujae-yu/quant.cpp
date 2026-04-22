## Qwen3.6-A3B 35B Long-Generation: SE + DN_NORM_FP64 preset (R53 P3)

### R53 P3 Update (2026-04-23): DeltaNet RMSNorm FP64 stacks on SE

Added per-head RMSNorm FP64 accumulation (TQ_DN_NORM_FP64=1) to the R52 SE
preset. Unlike state/output-path precision fixes (DN_FP64, SSM_OUT_FP32)
which REGRESS when stacked on SE, normalization-path precision stacks
additively. Auto-enabled alongside SE on filename match.

### R52 Foundation (SE-aware override)

40 routed experts (top-1 per layer by max activation from a dense-thinking
calibration prompt) are kept at FP32, preserving outlier-channel weights
that uniform IQ4_XS would clip. Memory cost ~480 MB extra. Per arxiv
2507.23279 "Unveiling Super Experts in MoE LLMs", these specific experts
hold load-bearing model state.

### Results (deterministic -j 1, "Once upon a time in a faraway land")

| Configuration | Total tok | Coherent tok |
|---|---:|---:|
| No preset | 54 | ~30 |
| R51 preset (ssm_out + LM + lcppport) | 234 | ~30 (extension only) |
| R52 SE preset (40 SEs FP32) | 187 | ~165 |
| **R53 P3 SE + DN_NORM_FP64 (current)** | **258** | **~195** |
| llama.cpp same model | ~1500 | ~1500 |

R53 P3 is +71 tok / +30 coh over R52 alone.

### Stacking table (what helps vs regresses on top of SE)

| Stacked with SE | Total tok | Verdict |
|---|---:|---|
| TQ_DN_NORM_FP64 (group norm) | **258** | ✓ stacks |
| TQ_DN_FP64 (recurrent state) | 182 | ✗ mild regression |
| TQ_SSM_OUT_FP32 | 195 | ✗ factual drift earlier |
| TQ_DN_FP64 + TQ_SSM_OUT_FP32 | 89 | ✗✗ severe regression |
| -k fp32 (FP32 KV cache) | 131 | ✗ worse than turbo_kv_4b |
| L2-norm FP64 on Q/K | 191 | ✗ regression |

Observation: **normalization-path precision** (RMSNorm accumulators)
stacks with SE; **state-path precision** (recurrent FP64, output FP32)
does not. RMSNorm over dv=128 values × 16 heads × 30 DeltaNet layers × N
tokens accumulates per-element rounding that compounds; FP64 mantissa
eliminates it. L2 normalize surprisingly regresses — the Q/K pre-norm
is sensitive to exact FP32 rounding pattern.

### Calibration recipe

```bash
# 1. Run with TQ_MOE_EXPERT_PROBE_ALL=1 on a representative prompt
TQ_MOE_EXPERT_PROBE=1 TQ_MOE_EXPERT_PROBE_ALL=1 TQ_QWEN35MOE_NO_PRESET=1 \
  build/quant <model> -p "<calibration prompt>" -n 50 -T 0 -j 1 --chat \
  > calib.log 2>&1

# 2. Aggregate per (layer, expert) max maxabs
grep "^\[moe-expert\]" calib.log | awk '...' > agg.txt

# 3. Top-1 routed per layer = SE candidate list
# 4. Pass via TQ_SE_LIST="L:E,L:E,..."
```

For Qwen3.6-35B-A3B UD-IQ4_XS, the calibrated list is hardcoded into the auto-preset.

## (R51 documentation below — superseded by R52 above)

## Qwen3.6-A3B 35B Long-Generation: known limit and preset

### Status

quant.cpp supports Qwen3.6-A3B 35B (qwen35moe arch, hybrid DeltaNet + 256-expert MoE) for inference. **Long-form generation has a documented coherence ceiling** of approximately 30 coherent tokens before entering attractor patterns (number sequences, quoted-word loops). The engine continues generating up to ~234 tokens with the auto-preset enabled, but the portion past ~30 tokens is typically degraded.

llama.cpp on the same model + weights produces 1500+ coherent tokens. The gap is in our forward-pass numerical realization (early-position logit ordering), not in the algorithm structure.

### Auto-preset (default ON)

When the model filename contains `Qwen3.6`, `qwen35moe`, `Qwen3.5-30B`, or `A3B`, the CLI automatically enables three precision flags that triple total generated length:

```
TQ_SSM_OUT_FP32=1       # FP32 dequant of ssm_out projection
TQ_OUTPUT_FP32=1        # FP32 dequant of LM head (~2 GB extra RAM)
TQ_DN_LLAMACPP_PORT=1   # Use verbatim llama.cpp delta_net inner loop
```

Stderr shows: `tq_main: qwen35moe preset auto-enabled (...)`.

**Opt out**: `TQ_QWEN35MOE_NO_PRESET=1`.

### Empirical baseline (deterministic, -j 1)

| Configuration | Tokens generated | Coherent portion |
|---|---:|---:|
| No preset (default Q4 ssm_out, Q6_K LM head) | 54 | ~30 |
| **With auto-preset (3 fixes stacked)** | **234** | **~30** |
| llama.cpp same model, same prompt | ~1500 | ~1500 |

Memory cost of the preset on 35B: ~2 GB extra RAM (FP32 LM head dequant).

### Why this works on 35B but hurts smaller models

The 3 fixes only help when the model uses Q8_0 attention weights with separate ssm_out (Qwen3.6-A3B specific). On Qwen3.5-4B (Q4_K throughout), enabling them regresses 4B from 191 → 122 tokens. The auto-preset is filename-gated to avoid this regression.

### Recommended use

For 35B Qwen3.6-A3B:
- **Short responses (under 30 coherent tokens)**: Works well with default preset.
- **Long-form / story generation**: Use llama.cpp until our forward-pass divergence is fixed.
- **Production**: We recommend Phi-3.5-mini, Qwen3.5-4B, or Llama-3.x for stable long-gen on 16 GB Macs.

### Investigation history (R48-R51)

R48-R51 (50+ engine modifications) extensively investigated the gap. Key findings:
- DeltaNet algorithm verified equivalent to llama.cpp (verbatim port test)
- Attractor entry happens by coherent token ~30, before any state drift
- Multi-thread mode introduces ±20-40 tok variance (use `-j 1` for measurement)
- 3 precision-up fixes (preset above) are the only single-line interventions that help

The remaining gap to llama.cpp's 1500 coherent tokens requires bit-exact match of forward-pass top-1 logits at early narrative-shaping positions, which has not yet been achieved despite line-by-line algorithmic equivalence with reference.

### Files

- `src/engine/tq_model.c`: TQ_SSM_OUT_FP32, TQ_OUTPUT_FP32 dequant paths
- `src/engine/tq_transformer.c`: TQ_DN_LLAMACPP_PORT (lines ~960-1030)
- `tools/quant.c`: auto-preset filename detection
- `.claude/state.md`: full R48-R51 investigation log
