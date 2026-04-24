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

We ship [`tools/basin_compat.sh`](../tools/basin_compat.sh) as a reference basin-compatibility tool. It runs the same prompt through our engine with `TQ_LAYER_TRACE=1` and `llama-debug --tensor-filter "^l_out-"`, producing a per-layer residual-sum diff.

### Basin compatibility ≠ output quality (important caveat!)

Measured 2026-04-25 on `Qwen3.5-4B Q4_K_M`:
- `basin_compat.sh` classification: **Tier 3** (max rel_diff 2.30, L0 diverges heavily)
- Actual generation on "Explain quantum mechanics in simple terms with examples.":
  > "Imagine the world around you is a giant, chaotic puzzle where everything is constantly moving and changing. **Quantum mechanics** (often called 'quantum physics') is the set of rules that describes how this tiny, chaotic universe actually works at the smallest possible scale. While we usually see a world where things move smoothly and predictably (like a car driving down a highway), the universe at its smallest level (atoms) behaves very differently. It behaves more like a game of chance than a chess game..."
  >
  > 147 coherent tokens, user-validated quality.

**So basin_compat is a DIAGNOSTIC of numerical divergence, not a quality verdict.** A different basin can still produce quality output if the weights generalize across numerical profiles (most models do on dense architectures).

### When basin mismatch actually bites

Basin mismatch predicts quality degradation specifically when:
1. Model architecture is deep (30+ layers)
2. Hidden state magnitudes grow through the network (late-layer values >> early layers)
3. Hard decision boundaries exist (MoE top-K, hybrid layer switching)
4. Autoregressive decode amplifies per-token drift via KV cache

Qwen3.6-35B-A3B hits all four: 40 layers, hybrid DeltaNet × MoE, top-8 expert gating, recurrent state. That's why `basin_compat` Tier 2 matches actual long-generation ceiling (~100 coh vs llama's 1090).

Qwen3.5-4B hits conditions 1 and 3 partially (32 layers, DeltaNet hybrid but dense FFN, no MoE gating). Basin diverges numerically but output quality holds.

Dense Llama/Phi/Gemma: conditions mostly absent; basin divergence rarely bites.

### Current tier assignments (2026-04-25)

Based on user-validated output quality, not basin_compat alone:

Rule of thumb:
- Long-generation quality matches llama.cpp within ~20%: **Tier 1**
- Long-generation hits attractor substantially earlier than llama on hard prompts: **Tier 2**
- Model is essentially unusable for long generation: **Tier 3** (currently empty)

Use `basin_compat.sh` as a diagnostic when a model unexpectedly degrades.

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
