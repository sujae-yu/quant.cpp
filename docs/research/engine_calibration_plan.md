# Engine-Specific Calibration — Research Plan

> **Goal**: given a model M trained for numerical profile P_train, and an inference engine E with numerical profile P_E ≠ P_train, produce a small weight-delta Δ such that E(M+Δ, x) coherent-length ≈ reference-engine(M, x) across autoregressive decode.
>
> **Novel framing**: post-training quantization calibration adapts weights for a *bit-width* change. Engine calibration adapts weights for a *numerical basin* change. Same mathematical model, different basin, same weights → attractor divergence. Add small delta, realign basin.

## Motivation

From R63 measurements:
- Qwen3.6-35B-A3B on our engine: ~100 coherent words, then attractor (Tier 2)
- Same model on llama.cpp: ~1090 coherent words
- Gap is structural (FP32 non-associativity × 40 layers × recurrent state × MoE gating)
- Piecemeal operator-level alignment REGRESSES (measured 6× this session)
- Full alignment = becoming a llama clone

If a small weight adjustment can close most of the gap, it's:
- Novel research direction (no prior art on "engine basin calibration")
- Project-identity-preserving (we stay "LLM의 SQLite", just with engine-aware model loading)
- Arxiv-publishable (framing alone is a contribution)

## Hypothesis space

### H1: Additive bias calibration
Add layer-specific bias vectors `b_L` such that our layer output matches reference.

```
our_output_L = our_layer_L(x) + b_L
```

Requires: ~2048 × 40 = 80K float parameters per model. Stored alongside GGUF.

Estimation method: run calibration corpus through both engines, record (our_L, reference_L) pairs, solve least-squares for b_L per layer.

**Pros**: tiny parameter count, easy to ship, interpretable.
**Cons**: linear correction — may not capture nonlinear basin drift.

### H2: Low-rank LoRA per layer
Add per-layer `A_L @ B_L` (rank 8-16) applied to hidden state.

```
our_output_L = our_layer_L(x) + α * (A_L @ B_L @ x)
```

Requires: 2048 × 16 × 2 × 40 = ~2.6M params per model. ~10 MB storage.

Estimation: standard LoRA training with (reference_trajectory, our_trajectory) pairs as target.

**Pros**: nonlinear correction capacity, proven infrastructure (PEFT library).
**Cons**: training loop needed, risk of overfitting to calibration data.

### H3: Operator-injection points
Add learned scalar scales at specific operators (RMSNorm weight, attention scale, MoE temperature).

```
our_attn_scale = learned_α_L
our_moe_temp = learned_β_L
our_rmsnorm = γ_L * default
```

Requires: ~200 scalar params per model.

**Pros**: ultra-lightweight, targeted at known divergence points.
**Cons**: may not be expressive enough for cascading errors.

### Recommended order
H3 (cheapest, quickest test) → H1 → H2.

## Experimental protocol

### Phase 1: Measurement (1 week)
1. Dump hidden states at each layer from both engines on a calibration corpus (100 diverse prompts).
2. Compute per-layer per-position divergence (cosine + L2).
3. Identify layers with highest cumulative contribution to final-logit divergence.

### Phase 2: H3 prototype (1 week)
1. For top-10 divergent layers, add scalar scale parameter.
2. Grid search (or gradient descent if differentiable) on calibration corpus.
3. Measure: does coh-bench improve on Qwen3.6-A3B?

Success criterion: Qwen3.6-A3B 3/3 prompts natural EOS (Tier 1 by coh_bench).

### Phase 3: H1 if H3 insufficient (2 weeks)
1. Collect (our_L_output, ref_L_output) pairs on calibration corpus.
2. Solve `min_b ||our_L + b - ref_L||_2` per layer.
3. Ship b_L as sidecar file alongside GGUF.

### Phase 4: H2 if H1 insufficient (4+ weeks)
1. Standard LoRA training setup.
2. Need: PEFT-compatible forward pass in our engine OR export intermediate activations to standard framework for training.
3. Much heavier infrastructure.

## Technical questions to resolve

1. **Reference target**: should calibration target llama.cpp's output, or the training-time output (PyTorch FP32)? The latter is "the truth" but unavailable; the former is a strong proxy.

2. **Corpus selection**: random wikitext? Instruction-tuned prompts? Task-specific?

3. **Generalization**: does calibration on prompt set A transfer to prompt set B, or does it overfit?

4. **Cost to ship**: can calibration delta be made prompt-independent, or does it need to be computed per-prompt? (Prompt-independent is the only practical option.)

5. **Unlearning**: does calibrated model lose zero-shot capability on truly out-of-distribution inputs?

## Prior work to review

Before investing weeks, we need to check:
- PTQ calibration literature (GPTQ, AWQ, HQQ) — what can we borrow?
- Knowledge distillation literature — engine-specific loss?
- "Model surgery" / test-time adaptation — closest analogs?
- Inference fingerprinting (papers on detecting which engine produced output) — inverse of our problem.

## Success definitions

**Minimum viable**: Qwen3.6-35B-A3B moves from Tier 2 to Tier 1 on our coh_bench (3/3 prompts natural EOS). No regression on Tier 1 models.

**Full success**: Method generalizes — we can apply it to any new Tier 2 model and reach Tier 1. Documented workflow. Arxiv paper draft.

**Moon-shot**: Open-source calibration deltas for top-20 Hugging Face models as sidecar files. Community adoption.

## Non-goals (to stay honest)

- NOT trying to match llama.cpp bit-exact. That's a lost war (see `memory/project_fp32_basin_theory.md`).
- NOT modifying ggml or llama.cpp. We adapt OUR side.
- NOT retraining the base model. We add a small delta.
- NOT making the engine slower. Delta application is O(dim) per layer — trivial.

## What this session has (2026-04-25)

✓ Hypothesis well-defined (this doc)
✓ Measurement infrastructure (`tools/basin_compat.sh`, `tools/coh_bench.sh`)
✓ Target model confirmed Tier 2 with concrete symptoms (docs/tier_benchmark_2026_04_25.md)
✓ FP32 basin theory as framing (docs/blog/fp32-basin-theory.md)
☐ H3 prototype
☐ Calibration corpus prep
☐ Training loop scaffold

## Next session kick-off

```bash
# 1. Dump hidden state pairs for small corpus
for prompt in $(cat calibration_prompts.txt); do
  TQ_DUMP_HIDDEN=/tmp/cal_ours ./build/quant <qwen3.6> -p "$prompt" -n 1
  ./refs/llama.cpp/build/bin/llama-debug -m <qwen3.6> -p "$prompt" --verbose \
    --tensor-filter "^l_out-" -n 1 > /tmp/cal_llama.$hash
done

# 2. Fit H3 (per-layer scalar) via offline optimization
python3 tools/calibration/fit_h3.py /tmp/cal_ours /tmp/cal_llama

# 3. Re-run coh_bench with calibrated model
./tools/coh_bench.sh qwen3.6 --calibration fitted_h3.json
```

Code/doc owner: future session. This plan is the handoff.
