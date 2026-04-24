# FP32 Basin Theory: Why Two Correct Inference Engines Produce Different Long-Generation Output

> **tl;dr** — Given identical model weights and identical math, two LLM inference engines can produce radically different long-generation quality. Not because either is wrong, but because floating-point addition is non-associative and 40-layer autoregressive decoders compound that non-associativity into distinct attractor basins. Chasing bit-exact parity with a reference engine piecemeal REGRESSES coherent output, because compensating operator orderings were co-tuned with the prior state. This is a measurable property of the engine, not a bug. It has a name now: the **FP32 stability basin**.

---

## The setup that made us see it

Qwen3.6-35B-A3B is a hybrid DeltaNet + MoE model. 40 layers. 256 experts, 8 active per token. `full_attention_interval = 4` (every 4th layer is regular self-attention; the rest are gated delta-net recurrent). We quantize to IQ4_XS (4.25 bits per weight), a fine quantization with almost no quality drop on the benchmarks that ship with the model card.

On `"Explain quantum mechanics in simple terms with examples."` in thinking mode, `-T 0`, `-n 1500`:

- **llama.cpp**: 1090 coherent words. Full 7-step reasoning scaffold. Real physics concepts. Self-correction step.
- **Our engine** (pre-fix): 5 coherent words, then `"killing jargon.but keeping it accurate.k\n.l\n.l2.l3.l4.l5.l6..."` alphabet-walk attractor.

Same weights. Same math. Same `-T 0`. Same prompt. Same quantization. Different universe of output.

We spent ~13 rounds of investigation assuming "we have a bug somewhere". We had, in fact, one bug. Fixing it delivered a 20× quality jump. But everything we tried *after* that regressed. And we learned why.

## The one bug we found: DeltaNet delta-rule FP32 ordering

In llama.cpp's `ggml_compute_forward_gated_delta_net_one_chunk`:

```c
// state[j*S_v + i] = S[i][j]  (TRANSPOSED layout, j-outer-i-inner)
for (int j = 0; j < S_v; j++) s_out[j] += delta[j] * k_d;   // rank-1 outer update, all j
for (int j = 0; j < S_v; j++) o[j] = dot(s_out[j], q_d) * scale;  // then output
```

In our engine's default:

```c
// state[i*dv + j] = S[i][j]  (row-major, i-outer-j-inner)
for (int i = 0; i < dk; i++) {
    sp[j] += ki * d_vec[j];    // update state row i
    oh[j] += sp[j] * qi;        // AND use it immediately for output accumulation
}
```

**Mathematically identical.** Element-wise `o[j] = Σ_i S[i][j] * q[i]` where S has been updated. But:

- We interleave state update and output accumulation (per-row).
- llama separates them (two passes over j).

FP32 addition is non-associative. The sum `S[0][j]*q[0] + S[1][j]*q[1] + ... + S[127][j]*q[127]` produces a slightly different 32-bit float depending on the order the 128 additions happen. Per layer: ~1e-5 relative error. Innocuous.

Except:

- DeltaNet layers number 30 out of 40. State carries forward through the decode loop.
- MoE top-K selection at each layer is a hard decision boundary. ~1e-3 logit perturbation → different top-8 experts on a close call.
- Softmax `exp()` amplifies ULP differences into visible probability shifts.
- Autoregressive decode: one token's different pick → different KV cache → different trajectory from that point forward.

So the 1e-5 per layer compounds into a token-level divergence by position ~30, which compounds into an attractor choice by position ~100.

We ported llama's exact delta-rule ordering behind `TQ_DN_LLAMACPP_PORT=1`, auto-enabled for qwen35moe models (commit [`f6a65bb`](https://github.com/quantumaikr/quant.cpp/commit/f6a65bb)). Layer-33 residual-sum rel_diff: **14.5% → 4.3%**. Long-generation coherent output: **5 words → 100 words**, with real physics concepts:

> "Superposition (Schrödinger's cat) → Entanglement (quantum entanglement) → Wave-particle duality (Double-slit experiment) → Quantum tunneling → Quantum computing (quantum supremacy)"

**20× improvement from one operator-level FP32-ordering port.** This is the shape of a real root-cause fix.

## What happened when we kept going

Hungry for more, we tried six additional fixes. All regressed.

| Fix attempt | Hypothesis | Tokens generated | Coherent words | Verdict |
|---|---|---:|---:|---|
| `DN_PORT` alone (baseline) | — | 149 | ~100 | ✓ our current peak |
| + Kahan summation in MoE aggregation | compensate FP32 error | 75 | fewer | regress |
| + llama-style router (softmax→top-K→renorm) | bit-exact routing | 69 | fewer | regress |
| + Bit-exact IQ4_XS/IQ3_S/Q6_K kernels | fully match llama quant kernels | 66 | fewer | regress |
| + FP32 KV cache | higher attention precision | 59 | fewer | regress (rep stop) |
| + NEON-matched delta-rule dot product | match llama's 4-parallel accumulator | 75 | fewer | regress — but **L33 rel_diff 0.46 → 0.22** |
| + DN_NORM_FP64 stacked | more precision | 59 | fewer | regress |

The NEON-matched dot product case is the smoking gun: it brought us **measurably closer** to llama on the per-layer residual metric (L33 raw divergence halved) while making the final output **measurably worse**.

Closer to the reference, by the obvious metric, means worse output. How?

## The theory: inference engines live in FP32 stability basins

Imagine the space of all consistent floating-point implementations of "apply these weights to this input using this architecture". Infinitely many. Same math, infinitely many `(order of ops, parallel width, reduction tree)` choices. Each produces a slightly different FP32 output.

For autoregressive decode, each implementation induces its own attractor landscape on sequences. Some implementations have a wide coherent basin (model "stays on track" for 1000+ tokens before hitting repetition). Some have a narrow one (degenerate after 30 tokens). Most implementations are somewhere in between.

**Model weights are trained under one specific numerical profile.** PyTorch + CUDA + FP16 mixed precision, typically. The weights "know" that profile — they encode not only "what to compute" but implicitly "what numerical regime this computation lives in". A basin that's "close to training" generalizes well to long generation. A basin that's "far from training" hits attractors.

llama.cpp has earned a good basin through years of ggml-graph-compiler tuning. They enforce consistent reduction order across operations via their graph IR. Small numerical accidents get baked into the engine and then stay.

Our engine, hand-coded C with per-op NEON micro-optimizations, lives in a **different basin**. For some models (Llama, Phi, Gemma, Qwen3.5-4B dense) our basin happens to be compatible — output quality matches llama within 20% across long generation. For hybrid DeltaNet × MoE on a specific mid-sized architecture (Qwen3.6-A3B), our basin is narrower — we hit a different attractor earlier.

**This is a property of the engine, not a bug to fix.**

### Why partial alignment regresses

Before the DN fix, our engine had compensating auto-presets (temperature 2.0 for MoE routing, FP64 RMSNorm, a list of 40 specific experts to keep in FP32). These were co-tuned against the buggy DN FP32 order. Each was a local minimum that offset a specific piece of drift.

When we fixed DN, the offsets became miscompensations pushing the same direction the fix pushed. Basin moved. Output got worse until we also removed the old compensations.

When we tried to port MORE of llama's FP32 order (NEON dot product, Kahan sums), we pushed ONE more operator's offset toward llama's basin while leaving the rest of our operators in our basin. That's not "halfway to llama". That's "off of our basin and not yet on llama's". Worst of both worlds.

The only stable configurations are:
1. Our engine's native basin (hand-coded, consistent with itself)
2. Exact llama bit-level copy (all operators, all orderings, all reductions)

There's no stable middle.

## What this means for the project

We are quant.cpp. We are not trying to be llama.cpp in C single-header form. That would be a silly goal. Our tagline is "the SQLite of LLMs" — smallest, most readable, most embeddable. Our killer feature is KV cache compression, 7× at PPL +0.0%. Our binary is 192 KB as WASM. llama.cpp's is 6+ MB. We run bit-exact from a 17.6K-line single header.

So we pivot:

**Axis 1 — strengthen strengths.** Push KV compression further. MLA attention × our 4-bit KV = potentially 64× compression, 128K context on a 16 GB MacBook. Ship.

**Axis 2 — be honest about basins.** Introduce a model Tier system in the README. Qwen3.6-A3B is Tier 2 on our engine. Users who need 1000+ coherent thinking traces on that specific model should use llama.cpp. Users who need 128K context on a constrained device should use us. This is not a concession; it's good positioning.

**Axis 3 — the research angle no one owns.** If engines live in different FP32 basins and model weights implicitly depend on their training profile, then there should exist a small fine-tune that **adapts model weights to a target engine's basin**. Analog of post-training quantization calibration, but for numerical ordering. We have the measurement tool (`tools/basin_compat.sh`). We have the target (Qwen3.6-A3B on our basin). If a 100-sample LoRA can close the Tier 2 gap, that's novel research, arxiv-worthy, and converts "we're weaker here" into "we have a solution nobody else has thought of".

## What you can do with the tool

```bash
./tools/basin_compat.sh models/Your-Model.gguf
```

Pair-runs our engine and `llama-debug` on the same prompt, dumps per-layer residual-sum divergence, assigns Tier. Works best on DeltaNet-hybrid models (Qwen3.5/3.6 family). Pure-feedforward models only get final-layer decode dumps from llama-debug, limiting comparison granularity.

If your model's per-layer rel_diff stays below 5% across all 40 layers, you're in Tier 1 — our engine delivers reference-close quality. Between 10-40% at late layers: Tier 2, functional but watch long-generation on that specific family. Above 50% or early-layer jumps: Tier 3, needs per-model work.

## The deeper question

LLM inference isn't one function. It's a family of consistent-with-the-math implementations that disagree on floating-point rounding. Every engine picks one member of that family. The model's weights picked one too, at training time. When the choices align, long generation works. When they don't, you hit attractors.

We don't normally talk about LLM inference this way. Papers describe algorithms. Engines implement algorithms. Tests verify numerical closeness on short sequences. Long-generation stability gets lumped in with "quality" and blamed on quantization, sampler temperature, or the vague category of "model artifacts."

Our measurement (and the hundreds of hours that led up to seeing it clearly) says: the single biggest thing you can do to improve long-generation quality on a model is to make sure your engine's FP32 profile sits in a basin compatible with how the model was trained. That's more important than kernel speed, quantization tightness, or sampler sophistication.

This is maybe the most interesting thing we've learned in 18 months of building this engine. It took a specific model's specific failure mode to make it visible. It took reverting six "improvements" to be sure.

Now we have a language for it, a measurement tool, a Tier system, and a research direction. If you've been trying to reconcile "my engine passes unit tests but the output is weird on long prompts," this may be why.

---

*Tools and commits referenced above:*
- `tools/layer_diff_qwen36.sh` ([`a07069f`](https://github.com/quantumaikr/quant.cpp/commit/a07069f)) — paired-diff against llama-debug, Qwen3.6 specific
- `tools/basin_compat.sh` ([`24cadbb`](https://github.com/quantumaikr/quant.cpp/commit/24cadbb)) — generalized tier classifier
- `TQ_DN_LLAMACPP_PORT=1` ([`f6a65bb`](https://github.com/quantumaikr/quant.cpp/commit/f6a65bb)) — auto-enabled DeltaNet delta-rule port
- Strategic-pivot auto-preset cleanup ([`9fbe82e`](https://github.com/quantumaikr/quant.cpp/commit/9fbe82e))
- `docs/engine_basin_tiers.md` — tier system + measurement methodology
