# Tier Benchmark — 2026-04-25

Standardized coherent-length measurement across 5 models, 3 prompts each. Run via [`tools/coh_bench.sh`](../tools/coh_bench.sh), `-T 0`, `-n 300`, single-thread deterministic.

## Prompts

1. **quantum**: "Explain quantum mechanics in simple terms with examples."
2. **poem**: "Write a short poem about a solitary lighthouse at dawn."
3. **trivia**: "What is the capital of France, and name two famous landmarks there?"

## Raw results — 14 models across family

| Model                           | quantum            | poem              | trivia             | Tier |
|:--------------------------------|:------------------:|:-----------------:|:------------------:|:----:|
| SmolLM2-135M Q8_0               | 299 (-n cap)       | 108 (EOS)         | 22 (EOS)           | 1    |
| SmolLM2-360M Q8_0               | 299 (-n cap)       | 108 (EOS)         | 22 (EOS)           | 1    |
| **Qwen2.5-0.5B Q4_K_M**         | **64, rep loop**   | **49, rep loop**  | **55, rep loop**   | **3** |
| Qwen3-0.6B Q4_K_M               | 299 (-n cap)       | 285 (EOS)         | 299 (-n cap)       | 1    |
| Qwen3.5-4B Q4_K_M               | 147 (EOS)          | 106 (EOS)         | 66 (EOS)           | 1    |
| llama-3.2-1B Q4_K_M             | 299 (-n cap)       | 133 (EOS)         | 110 (EOS)          | 1    |
| Llama-3.2-1B Q8_0               | 261 (EOS)          | 107 (EOS)         | 137 (EOS)          | 1    |
| Llama-3.2-3B Q8_0               | 299 (-n cap)       | 105 (EOS)         | 120 (EOS)          | 1    |
| Gemma-4-e2b-it Q8_0             | 299 (-n cap)       | 92 (EOS)          | 31 (EOS)           | 1    |
| Gemma-4-e4b-it Q4_0             | 299 (-n cap)       | 82 (EOS)          | 19 (EOS)           | 1    |
| Phi-3.5-mini-instruct Q4_K_M    | 299 (-n cap)       | 299 (-n cap)      | 299 (-n cap)       | 1    |
| Phi-3.5-mini-instruct Q8_0      | 299 (-n cap)       | 299 (-n cap)      | natural EOS        | 1    |
| **Qwen3.6-35B-A3B IQ4_XS**      | 149 (EOS, thinking) | **73, rep loop**  | **51, attractor**  | **2** |
| **Qwen3.6-35B-A3B Q5_K_M**      | 169 (EOS, thinking) | **68, rep loop**  | **69, rep loop**   | **2** |
| **Qwen3.6-27B Q4_K_M** (NEW)    | not measured        | not measured       | not measured       | **3** |

**Key observations:**

- Qwen2.5-0.5B (Tier 3): model is genuinely too small — attractor under any engine. Keep in table as lower-bound reference.
- Qwen3.6-35B-A3B (both quants Tier 2): **quantization change doesn't help**. Both IQ4_XS and Q5_K_M show the same basin mismatch (attractor at ~70 tok on non-thinking prompts). Confirms basin issue is structural, not bit-width.
- **Qwen3.6-27B Q4_K_M (Tier 3, 2026-04-25)**: newly released dense DeltaNet hybrid (arch `qwen35`, 64 layers, dim 5120, time_step_rank=48 / ssm_inner=6144 — different config from 35B-A3B). basin_compat shows max rel_diff **3.87 at L3** with sign flips throughout. This is NOT basin mismatch (which preserves sign) — it's a fundamental forward-pass bug (likely in DeltaNet config variants we haven't validated, or MoE-code-path leak into dense models). 16.8 GB Q4_K_M file also exceeds 16 GB RAM, causing constant disk swap. Marked Tier 3 pending engineering investigation.
- 10/14 models are clean Tier 1 across 3 prompts.
- No prior Tier 1 model was demoted by R63 cleanup. The DN_PORT fix + auto-preset cleanup was a net improvement.

## Qwen3.6-27B Q4_K_M — full diagnostic (2026-04-25)

basin_compat measurement (sums over 5120 elements per layer):

| Layer | ours | llama | rel_diff | Notes |
|-------|------|-------|----------|-------|
| 0 | 23.5 | 6.8 | 2.47 | already off, same sign |
| 1 | 28.1 | 6.5 | 3.35 | growing |
| 3 | **-52.2** | **+18.2** | **3.87** | **sign flip — fundamental** |
| 7 | 117 | 33.5 | 2.49 | |
| 33 | 14 | 268 | 0.95 | huge magnitude diff |
| 51 | -196 | 84 | 3.33 | sign flip |
| 57 | -339 | 138 | 3.46 | sign flip |
| 59 | -481 | 219 | 3.20 | sign flip |
| 63 | 160 | -121 | 2.32 | sign flip on FINAL layer |

Max rel_diff: **3.87** (L3). For comparison:
- Qwen3.6-35B-A3B (Tier 2): max 0.41, sign preserved everywhere
- Qwen3.5-4B (Tier 1 quality, Tier 3 by basin): max 2.30, sign mostly preserved
- Qwen3.6-27B (Tier 3): max 3.87, sign flips at L3 onwards

**Sign flips at early layers are the diagnostic signature of fundamental forward-pass bug** (not FP32 basin drift). Recommended action: skip 27B until bug is investigated.

**Suspected causes** (not validated this session):
1. DeltaNet config variant (time_step_rank=48 vs A3B's 32, ssm_inner=6144 vs 4096) handling.
2. MoE-only code paths firing on dense model (load message "Fused MoE kernels ready" appeared on dense model — likely harmless but worth confirming).
3. Tensor layout interpretation differences in 27B GGUF.

**Validated this session (2026-04-25)**:
- Tensor names match A3B exactly (`attn_qkv`, `attn_gate`, `ssm_a/alpha/beta/dt/norm/out/conv1d`)
- All tensor shapes consistent with config (qkv split = 16×128×2 + 48×128 = 10240 ✓)
- GGUF metadata matches expected (rope sections [11,11,10,0], eps 1e-7, ssm.conv_kernel=4)
- Layer pattern (16 attn at L3,7,11,...,63) matches llama-debug
- ssm_a values in sensible range (-0.34 to -0.004 at L0, similar A3B pattern)
- `is_moe = (num_experts > 0) = false` ✓ (MoE code paths gated correctly)
- TQ_FORCE_QK_NORM=1 does NOT help (max rel_diff stays at 3.87 — not a QK-norm issue)
- TQ_DN_LLAMACPP_PORT auto-enabled (DeltaNet detect via delta_n_heads > 0) ✓

**Remaining investigation** (next session, multi-hour):
- Element-level sub-op trace at L0 pos=0: qkv-proj output, conv1d output, Q/K/V split, L2-norm, decay, delta, state update, output, ssm_norm
- Compare each sub-op against llama-debug's `cb(...)` named tensors at L0
- First materially divergent sub-op identifies the bug

**Quick paths to try first** (not done this session):
- Q4_K dequant validation — 27B has hidden_dim 5120; Q4_K block size is 256 (5120/256=20 blocks). For previous models hidden 2048/3072/4096 (8/12/16 blocks). 5120 is unusual — verify dequant produces correct values.
- TQ_DELTANET_FP32=1 to bypass DN quant entirely and compare ✓ TESTED — same Tier 3, max 3.87. Quant is NOT the cause.
- Run on smaller quant (UD-IQ2_M 10.1 GB) to see if Tier 3 persists across bit-widths

**Element-level L0 comparison** (2026-04-25, after BOS alignment confirmed):
```
position 0 (BOS), L0 first 3 elements:
  ours:   [-0.0548,  0.3547, -0.7901]
  llama:  [-0.1097, -0.0390,  0.0355]
  diff:   [ 2× off, SIGN FLIP, SIGN FLIP + 22× magnitude ]
```

Sum-level diff was 247% (23.5 vs 6.77). Element-level shows OUTLIER CHANNELS pattern — specific dimensions blown up while sum stays manageable.

**Outlier-channel pattern suggests**:
- Specific norm weight reading mis-aligned (e.g., attn_norm dim 5120 — boundary issue?)
- Embedding lookup scaling factor missing (some Qwen variants have embed_scale = sqrt(hidden_dim))
- Specific projection (qkv split offsets, conv1d channel split) shifting dim assignments

**Concrete next investigation step**: dump first 20 elements of each named tensor at L0 (post_embed, attn_norm_out, qkv_proj_out, conv1d_out, q_split, k_split, v_split, q_l2norm, k_l2norm, gate_silu, delta_state, delta_out, ssm_norm_out, residual). First materially-divergent step localizes the bug.

**Memory**: at 16.8 GB Q4_K_M model size on 16 GB RAM Mac, evaluation is impractical (constant swap, ~0.3 tok/s, -n 30 test took 15+ min). For users wanting to test 27B, smaller quants are available:
- UD-IQ2_M: 10.1 GB (recommended for 16 GB RAM)
- UD-Q2_K_XL: 11.0 GB
- Q3_K_S: 11.5 GB
But the same Tier 3 basin issue would apply regardless of bit-width.

## Quality verdicts (first ~200 chars)

### Qwen3-0.6B — Tier 1

quantum: *"Okay, the user wants me to explain quantum mechanics in simple terms with examples. Let me start by recalling what I know about quantum mechanics. First, I need to define it clearly. Quantum mechanics..."*

poem: *"Okay, the user wants me to write a short poem about a solitary lighthouse at dawn. Let me start by visualizing the scene. A lighthouse at dawn—so it's peaceful, maybe with some elements like sea..."*

trivia: *"Okay, so I need to figure out what the capital of France is and also name two famous landmarks there. Let me start by recalling some basic facts about France. First, the capital of France is Paris..."*

### Qwen3.5-4B — Tier 1

quantum: *"Imagine the world around you is a giant, chaotic puzzle where everything is constantly moving and changing. **Quantum mechanics** (often called 'quantum physics') is the set of rules that describes..."*

poem: *"The morning breaths its first soft sigh, As mist clings heavy to the shore; A single star above the hilltop sky, Watches over a silent shore. The sea is still, a vast expanse of gray..."*

trivia: *"The capital of France is **Paris**. Two famous landmarks located in Paris include: 1. The **Eiffel Tower** (La Tour Eiffel), 2. The **Louvre Museum**..."*

### Llama-3.2-3B — Tier 1

quantum: *"Quantum mechanics! It's a fascinating and mind-bending branch of physics that can be tricky to wrap your head around, but I'll try to explain it in simple terms with examples. **What is quantum mechanics?**..."*

poem: *"Amidst the misty morning's hush, A sentinel stands, steadfast and still. The solitary lighthouse, a beacon's rush, Cuts through the dawn's gray veil with gentle will..."*

trivia: *"The capital of France is Paris. Two famous landmarks in Paris are: 1. The Eiffel Tower (La Tour Eiffel) - a iconic iron lattice tower built for the 1889 World's Fair..."*

### Phi-3.5-mini — Tier 1

quantum: *"Quantum Mechanics is a branch of physics that deals with the behavior and interactions of particles at very small scales, such as atoms or subatomic particles..."*

poem: *"In the hush of early morn, A sentinel stands alone. Its beacon pierces through night's embrace, a steadfast guide for the lost at sea. The lighthouse watches over the waves with silent vigil..."*

trivia: *"The capital city of France is Paris. Two notable landmarks in this vibrant metropolis are the Eiffel Tower and Notre-Dame Cathedral, both iconic symbols of French culture..."*

### Qwen3.6-35B-A3B — Tier 2

quantum (thinking mode): *"Here's a thinking process: 1. **Deconstruct the Request:** - 'Explain quantum mechanics simply.' -> 'Use analogies, avoid jargon where possible, keep it engaging...'"* → 149 tok, quality reasoning trace.

poem: *"Thinking Process: . ) )56789101012345678910. Here is a poem about the lighthouse: - A solitary figure stands against the dawn's first light, casting long shadows of three45678901234567..."* → **73 tok, repetition loop**. Number-walk attractor after ~15 coherent words.

trivia: *"Thinking Process: .1. The user is asking about the location ofof Paris, France's capital city3.'4.'56789012345678901234567890"* → **51 tok, attractor**. Answer never delivered.

## Tier classification (data-driven)

Rule: model is **Tier 1** if all 3 prompts complete without attractor (natural EOS or -n cap, no repetition loop on -T 0). Otherwise **Tier 2**.

| Model | Tier | Evidence |
|-------|:----:|----------|
| Qwen3-0.6B Q4_K_M | 1 | 3/3 prompts natural |
| Qwen3.5-4B Q4_K_M | 1 | 3/3 prompts natural |
| Llama-3.2-3B Q8_0 | 1 | 3/3 prompts natural |
| Phi-3.5-mini Q4_K_M | 1 | 3/3 prompts natural |
| Qwen3.6-35B-A3B IQ4_XS | 2 | 1/3 prompts natural (thinking only); 2/3 hit attractor < 75 tok |

## Reproducibility

```bash
# Requires build/quant and relevant models under models/
./tools/coh_bench.sh \
  models/Qwen3-0.6B-Q4_K_M.gguf \
  models/Qwen3.5-4B-Q4_K_M.gguf \
  models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
```

## Pairs well with

- [`tools/basin_compat.sh`](../tools/basin_compat.sh) — numerical divergence diagnostic (per-layer rel_diff vs llama-debug)
- [`docs/engine_basin_tiers.md`](engine_basin_tiers.md) — theory of when numerical divergence translates to quality loss
- [`docs/blog/fp32-basin-theory.md`](blog/fp32-basin-theory.md) — public-facing writeup of the finding

As noted in the theory doc, **basin compat (numerical) does not always predict coh quality**. Qwen3.5-4B has Tier 3 basin_compat score (2.30 max rel_diff) yet delivers Tier 1 coherent quality. Qwen3.6-A3B has Tier 2 basin_compat (0.41 max rel_diff) AND Tier 2 quality. The two metrics should be read together.
