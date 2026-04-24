# Engine Basin Tiers — What to Expect Per Model

> **tl;dr** — quant.cpp's coherent-generation quality depends on per-model **FP32 basin compatibility**. Not all engines are equal at long-generation on all models, even with identical weights and math. We classify supported models into three tiers so you know what to expect.

## Why tiers

After measuring 13+ rounds of attempted FP32 parity with llama.cpp on Qwen3.6-35B-A3B (a hybrid DeltaNet/self-attn MoE), we confirmed what long-time LLM inference practitioners suspect: **inference engines exist in FP32 stability basins**. Two engines can implement the same mathematical model with different floating-point operation orderings and end up in different attractor landscapes during autoregressive decode. Model weights — trained under a specific numerical profile — adapt implicitly to one basin and not another.

This is not a bug. It's a measurable property of floating-point non-associativity compounded over 40+ layers, softmax `exp()` amplification, MoE hard decision boundaries, and recurrent state feedback. See [FP32 Basin Theory](./fp32_basin_theory.md).

## The tiers

### Tier 1 — Production quality

Our engine's FP32 basin is compatible with this model family. Long-generation quality matches llama.cpp within 20%. Suitable for user-facing applications.

- **Llama 3.1 8B** (and variants)
- **Phi-3.5-mini** — our fastest quality-coherent model on Apple Silicon
- **Gemma 4** (all sizes)
- **Qwen3.5-4B dense**

### Tier 2 — Research grade

Functional but our basin differs from reference implementation. Short-context correctness verified; long-generation may hit our-basin-specific attractors earlier than llama.cpp's.

- **Qwen3.6-35B-A3B** (UD-IQ4_XS, Q5_K_M)
  - Short reasoning (<200 tokens): fine
  - Long thinking-mode generation: ~150 coherent tokens vs llama.cpp's 1090
  - Root cause understood (hybrid DeltaNet + MoE cascade amplification), fix is system-wide not piecemeal
  - Opt into with eyes open; not recommended for production chat UI

### Tier 3 — Needs engine research

Models where basin incompatibility is severe. We currently skip or require explicit acknowledgement. Future calibration research may promote.

*Currently empty — we add models here when our basin compatibility tool measures >50% per-layer cumulative divergence.*

## Measurement methodology

We ship [`tools/layer_diff_qwen36.sh`](../tools/layer_diff_qwen36.sh) as a reference basin-compatibility tool. It runs the same prompt through our engine with `TQ_LAYER_TRACE=1` and `llama-debug --tensor-filter "^l_out-"`, producing a per-layer residual-sum diff.

Rule of thumb:
- All 40 layers within 5% rel_diff → Tier 1
- 10-40% rel_diff at late layers → Tier 2
- 50%+ cumulative, early jumps → Tier 3

## Why we don't just match llama.cpp

Because **we measured** (R63, 2026-04-24/25) that single-operator alignment with llama.cpp REGRESSES coherent output. Example: matching llama's NEON dot-product accumulation order in our DeltaNet port improved layer-33 raw divergence from 0.46 → 0.22 but dropped coherent output from 149 tokens → 75 tokens. Local metric improved, global stability broke.

This is the "delicate equilibrium" phenomenon: our engine's compensating auto-presets (temperature 2.0, FP64 normalization, etc.) were co-tuned with original operator ordering. Changing one op alone breaks the compensation chain. Changing ALL ops simultaneously = becoming a llama.cpp fork, which defeats our project identity (`"LLM의 SQLite"` — smallest, most readable, most embeddable engine).

The right path forward — which no one else is pursuing — is **engine-specific calibration**: lightweight weight fine-tuning that adapts a model to a specific engine's FP32 profile. Analog of post-training quantization calibration, but for numerical basin. Research in progress.

## If you need 1000+ coherent on Qwen3.6

Use llama.cpp. They earned that quality through years of ggml graph-compiler ordering. We respect that.

Use us when you need:
- **Long context on constrained hardware** — our 6.4-7× KV compression (killer feature)
- **Smallest binary** — 192 KB WASM, 17.6K LOC single header
- **Tier 1 models on Apple Silicon** — often faster than llama.cpp
- **Embedding into games/mobile/browsers** — where a 6+ MB binary is unacceptable
