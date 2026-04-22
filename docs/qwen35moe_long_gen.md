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
