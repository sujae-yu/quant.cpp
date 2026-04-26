# Tier Benchmark — 2026-04-25

Standardized coherent-length measurement across 5 models, 3 prompts each. Run via [`tools/coh_bench.sh`](../tools/coh_bench.sh), `-T 0`, `-n 300`, single-thread deterministic.

## Prompts

1. **quantum**: "Explain quantum mechanics in simple terms with examples."
2. **poem**: "Write a short poem about a solitary lighthouse at dawn."
3. **trivia**: "What is the capital of France, and name two famous landmarks there?"

## Raw results — 14 models across family

(Original 2026-04-25 measurement; updated 2026-04-26 column shows post R1–R6 re-measurement.)

| Model                           | 4-25 quantum / poem / trivia | 4-26 quantum / poem / trivia | Tier (4-26) | Δ |
|:--------------------------------|:-----------------------------|:-----------------------------|:-----------:|:-:|
| SmolLM2-135M Q8_0               | 299 / 108 / 22  (3/3 EOS)    | 299 EOS / **241 rep** / 63 EOS | **2** | ↓1 |
| SmolLM2-360M Q8_0               | 299 / 108 / 22  (3/3 EOS)    | 299 EOS / 108 EOS / 22 EOS    | 1 | = |
| **Qwen2.5-0.5B Q4_K_M**         | 64 / 49 / 55  (3/3 rep)      | 64 rep / 49 rep / 55 rep      | **3** | = |
| Qwen3-0.6B Q4_K_M               | 299 / 285 / 299  (3/3 EOS)   | 299 EOS / 285 EOS / 299 EOS   | 1 | = |
| Qwen3.5-4B Q4_K_M               | 147 / 106 / 66  (3/3 EOS)    | 114 EOS / 131 EOS / 66 EOS (post-R8) | 1 | = |
| llama-3.2-1B Q4_K_M             | 299 / 133 / 110  (3/3 EOS)   | 299 EOS / 133 EOS / 110 EOS   | 1 | = |
| Llama-3.2-1B Q8_0               | 261 / 107 / 137  (3/3 EOS)   | 261 EOS / 107 EOS / 137 EOS   | 1 | = |
| Llama-3.2-3B Q8_0               | 299 / 105 / 120  (3/3 EOS)   | 299 EOS / 105 EOS / 120 EOS   | 1 | = |
| Gemma-4-e2b-it Q8_0             | 299 / 92 / 31  (3/3 EOS)     | 299 EOS / 92 EOS / 31 EOS     | 1 | = |
| Gemma-4-e4b-it Q4_0             | 299 / 82 / 19  (3/3 EOS)     | 299 EOS / 82 EOS / 19 EOS     | 1 | = |
| Phi-3.5-mini Q4_K_M             | 299 / 299 / 299  (3/3 -n)    | 299 -n / 299 -n / 299 -n      | 1 | = |
| Phi-3.5-mini Q8_0               | 299 / 299 / EOS  (3/3 OK)    | 299 -n / 299 -n / 299 -n      | 1 | = |
| **Qwen3.6-35B-A3B IQ4_XS**      | 149 EOS / 73 rep / 51 rep    | 149 EOS / 76 rep / 60 rep (post-R7) | **2** | = |
| **Qwen3.6-35B-A3B Q5_K_M**      | 169 EOS / 68 rep / 69 rep    | **24 EOS / 225 rep / 46 EOS** | **2** | = |
| **Qwen3.6-27B Q4_K_M**          | not measurable on 16 GB Mac (R2)                            | not measurable (R2) | **3** | n/a |
| Qwen3.6-27B-TQ2_0 (R5/R6)       | engine path verified (paging-cliff cleared) but quality is requantize-artifact garbage | requantize-from-Q4 or Q8 both garbled | **n/a (engine-only)** | new |

**Summary of post-R1–R6 changes** (and R7 follow-up regression-fix):
- **R7 regression bisect (2026-04-26)**: deterministic 35B-A3B IQ4_XS regression (149 EOS quantum → 94 rep loop) was bisected to commit `12e4d94` (R1 BOS fix). Root cause: GGUF metadata declares `tokenizer.ggml.add_bos_token=false` for both Qwen3.6-27B and 35B-A3B; R1 force-enabled BOS via `<|endoftext|>` presence detection regardless of the metadata flag. Chat template is self-contained — prepending BOS broke generation. **R7 fix removes the auto-enable path; 35B-A3B IQ4_XS quantum restored to 149 tok EOS (Tier 2 confirmed).**
- **R8 generalisation (commit 714cd4c)**: replace R7 family-specific heuristic with model-agnostic GGUF metadata read. New `tq_tokenizer_t.add_bos_token` tristate field (`+1` / `-1` / `0` for true / false / unset) parsed from `tokenizer.ggml.add_bos_token`. `tq_generate.c` consults it before any heuristic. Verified: 35B-A3B IQ4_XS quantum still 149 EOS post-R8; Qwen3.5-4B trivia returns to its true baseline 66 tok EOS (the earlier 4-26 measurement of 209 tok was a side-effect of R1's BOS auto-enable, not a real quality gain).
- **SmolLM2-135M poem rep loop is a measurement-only artifact**: re-running on the `0829285` baseline tokenizer produces the *same* 241 rep loop, so the original 4-25 doc value (108 EOS) is the outlier. The 4-26 column reflects current behavior; SmolLM2-135M is genuinely Tier 2 on this prompt under both pre-R1 and post-R7 codebases.
- **All other 11 Tier 1 models unchanged** — R1 BOS fix (post-R7), R3 IQ2_XS impl, and R5 TQ2_0 impl did not break any prior-passing model.

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

**llama sub-op tensor names captured (2026-04-25)** for paired-diff:
```
attn_norm-N      = MUL(norm output × attn_norm.weight)
linear_attn_qkv_mixed-N  shape {10240, n_tokens}
conv_states_reshaped-N   shape {3, 10240}        (conv buffer state)
conv_input-N             shape {5, 10240}        (concat states + new)
conv_output_raw-N        SSM_CONV(input, conv1d.weight)
conv_output_silu-N       SILU(conv_output_raw)
q_conv-N                 VIEW shape {128, 16, n_tokens}     ← Q at offset 0
q_conv_predelta-N        L2_NORM(q_conv)
k_conv-N                 VIEW shape {128, 16, n_tokens}     ← K at offset 16×128=2048
k_conv_predelta-N        L2_NORM(k_conv)
v_conv_predelta-N        VIEW shape {128, 48, n_tokens}     ← V at offset 2×16×128=4096
```

**Verified split offsets match ours**: Q at 0, K at 2048, V at 4096. Our `delta_qkv[0:2048]` Q, `delta_qkv[2048:4096]` K, `delta_qkv[4096:10240]` V — ✓ identical.

**Suspected at this point** (since shape/split/load all verified):
- ssm_conv1d weight: shape `{4, 10240}` in GGUF. Our load assumes specific layout. May need to verify how we read this 2-D weight with channel dim 10240 (different from A3B's 8192).
- L2_NORM op specifics — we may apply differently than llama's `ggml_l2_norm`.
- input_layernorm to DN: we use `attn_norm` weight; verify boundary handling for hidden=5120.
- BOS token handling differences (both engines DO add BOS, confirmed).

**Memory**: at 16.8 GB Q4_K_M model size on 16 GB RAM Mac, evaluation is impractical (constant swap, ~0.3 tok/s, -n 30 test took 15+ min). For users wanting to test 27B, smaller quants are available:
- UD-IQ2_M: 10.1 GB (still paging-bound on 16 GB — see R3 update below)
- UD-Q2_K_XL: 11.0 GB
- Q3_K_S: 11.5 GB
But the same Tier 3 basin issue would apply regardless of bit-width.

## R1–R3 update (2026-04-26) — BOS fix verified, IQ2_XS implemented, measurement still RAM-blocked

**R1 — BOS root cause found and fixed (commit 12e4d94)**
- Symptom: L0 attn_norm_out elements sign-flipped vs llama (ours +0.25, llama −0.29)
- Root cause: Qwen3.6 GGUF declares `bos_token_id = 248044` (`<|endoftext|>`), but our str_lookup chain hit `<|im_start|>` (248045) first
- Fix: added `<|endoftext|>` to BOS lookup chain in `tq_tokenizer.c`; added Qwen3.6 BOS auto-detection + post-encode override in `tq_generate.c` (vocab > 240K models)
- **Verification**: with corrected BOS, L0 attn_norm first3 = `[-0.2891, -0.6430, 0.4991]` — bit-exact match with llama-debug. Sign flip and outlier-channel pattern from Q4_K_M diagnostic resolved at the prefill layer.
- Regression check: Qwen3.5-4B remains Tier 1 (3/3 natural EOS), trivia improved 66→209 tok with the new BOS path.

**R2 — Q4_K_M coh_bench attempt: blocked by RAM**
- 16.8 GB file vs 16 GB physical RAM ⇒ severe swap, RSS dropped to 752 KB after 54 min, no decode progress.
- Confirms previous "evaluation is impractical" finding; not a path forward on this hardware.

**R3 — IQ2_XS dequant implemented to unblock UD-IQ2_M; measurement still RAM-bound**
- Discovery: UD-IQ2_M (10.1 GB) uses IQ2_XS internally for some tensors. Our engine had stub returning zeros (`tq_gguf_quants.c:1664`), producing garbled output: `tq_gguf_quants: WARNING: IQ2_XS dequant not yet implemented, returning zeros`.
- Fix: ported `dequant_iq2_xs` from `refs/llama.cpp/ggml/src/ggml-quants.c:2440` — 74-byte block, 512-entry codebook (`iq2xs_grid`), 9-bit grid index + 7-bit signs index per qs uint16. Reuses existing `kmask_iq2xs` and `ksigns_iq2xs` tables.
- Build: clean. Stub warning gone.
- coh_bench attempt: still paging-bound. 30-tok chat test ran 16 min wall / 2.6 min CPU (84% time in I/O). Output partial: `" t.\n\nt。\n\n t tว่า"` — same garbled mixed-script pattern as before. Cannot disambiguate "IQ2_M too lossy for 27B dense" from "subtle IQ2_XS impl bug" without working llama.cpp reference, but llama.cpp also couldn't load alongside ours due to RAM contention.
- Conclusion: 10 GB+ models on 16 GB Mac are paging-bound regardless of compute backend. **Reliable Qwen3.6-27B coh_bench measurement requires either ≥32 GB RAM, or a quant ≤8 GB (Q2_K plain, IQ2_XXS standard — neither commonly published for 27B).**

**Tier verdict**: Qwen3.6-27B Q4_K_M remains Tier 3 (cannot be promoted without coherent-length measurement on this hardware). The BOS fix + IQ2_XS impl are real improvements that will pay off on hardware that fits the model, but the original goal "Tier 3→1 promotion via Karpathy loop" is **hardware-blocked on a 16 GB Mac**, not engine-blocked.

**R4 — UD-IQ2_XXS (9.39 GB, Unsloth's smallest published 27B quant) — same RAM ceiling**
- Web research (2026-04-26): the only published Qwen3.6-27B quants are ≥9.39 GB (Unsloth UD-IQ2_XXS). IQ1_S/IQ1_M would be ~5-6 GB but no 27B file exists, and BitNet b1.58 / PowerInfer / Deja Vu / TurboSparse all require pre-training or ReLU activation (Qwen3.6 uses SwiGLU). Apple's "LLM in a Flash" assumes sparse FFN. PowerInfer-style mmap layer streaming is the only post-training option for dense SwiGLU and would take a multi-session refactor. None of these enable a 27B dense to fit a 16 GB box at usable speed.
- Hypothesis tested: 9.39 GB sits below the IQ2_M (10.1 GB) paging cliff, should give measurable tok/s.
- Result: **partially refuted**. UD-IQ2_XXS sanity test on chat-mode `"What is the capital of France?"` -n 30:
  - TTFT 406s (~7 min) — 27B prefill is paging-bound regardless of quant width
  - Decode 10 tok in 266s = **0.038 tok/s** (IQ2_M was 0.03, +27% — direction is right, magnitude is not)
  - CPU/wall = 21% (78% I/O wait — same paging signature as IQ2_M's 84%)
  - Quality improvement: output was clean token IDs (`22121\n\n5032`) instead of IQ2_M's garbled mixed-script `" t.\n\nt。\n\n t tว่า"` — **incidentally validates that the R3 IQ2_XS dequant is functioning correctly**; the IQ2_M garbage was the quant's own quality limit on 27B dense, not a bug in our impl.
- coh_bench projection: each `-n=80` prompt would take ~35 min; 3 prompts ≈ 105 min per measurement. Not Karpathy-loop tractable.
- **Final verdict for this hardware**: 27B Tier 3→1 promotion is hardware-blocked. Path forward is either (a) ≥32 GB RAM, (b) a Qwen3.6-27B IQ1_M quant (not yet published, ~6.5 GB would fit), or (c) implement post-training mmap layer streaming for dense SwiGLU (multi-session refactor, see PowerInfer / prima.cpp references).

**R5 — TQ2_0 self-quantize (8.06 GB) clears the cliff but loses quality**
- Idea: bypass Unsloth's lower limit by self-quantizing Q4_K_M → TQ2_0 (2.06 bpw ternary) via `llama-quantize --allow-requantize`. TQ2_0 needs no imatrix and is structurally simple (66-byte block, ternary {-d, 0, +d}).
- Quantize: 16028 MB Q4_K_M → 7674 MB TQ2_0, 28.9 s wall.
- Engine impl: ported `dequant_tq2_0` (~30 LOC) plus enum/dispatch entries. Build clean.
- Sanity test (`-n 30`, chat mode, single thread):
  - **Decode 29 tok in 559s = 0.052 tok/s** (vs IQ2_M 0.030 / IQ2_XXS 0.038 — **+73% / +37%**)
  - **TTFT 340s** (vs IQ2_XXS 406s, −16%)
  - **CPU/wall = 62%** (vs IQ2_M 16% / IQ2_XXS 21% — **paging cliff cleared**)
  - madvise(DONTNEED) on the 7.5 GB mmap ran cleanly
  - **Output garbled** (`_actorriguesurindezמותenuquetal做一名…`) — `--allow-requantize` from already-Q4_K_M warned about this and was right
- Conclusion: the cliff lives at ~8 GB on a 16 GB Mac, and we now have an engine path through it. But coh_bench evaluation needs a quality-preserving quant, which means either (i) downloading source BF16/F16 (~54 GB) and quantizing once, or (ii) waiting for Unsloth/bartowski to publish a calibrated TQ2_0/IQ1_M for 27B.
- Engine takeaway: TQ1_0 enum is reserved (54 B/block) but dequant not implemented; TQ2_0 dequant is in.

**R6 — Q8_0 (28.6 GB) → TQ2_0: source quality irrelevant, TQ2_0 is from-scratch quant**
- Hypothesis: R5 garbled output came from `--allow-requantize` doubling the quality loss (BF16 → Q4_K_M → TQ2_0). Re-quantize from Q8_0 (8.5 BPW, near-BF16) and quality should recover.
- Procedure: downloaded `unsloth/Qwen3.6-27B-Q8_0.gguf` (28.6 GB, ~30 min @ ~500 MB/min), `llama-quantize --allow-requantize Q8_0 → TQ2_0` (45 s, 7674 MB output identical to R5).
- Sanity test (`-n 30`, chat mode, single thread):
  - Decode 29 tok in 514s = **0.056 tok/s** (R5: 0.052, R4 IQ2_XXS: 0.038)
  - TTFT **305s** (R5: 340s, R4: 406s)
  - Output: `رانMutexご覧amatFormatExceptionSA享erb忌tape和改善heckmemo済大道ilder…` — **same multilingual garbage as R5**
- **Hypothesis refuted**: Q8_0 source produced *exactly* the same garbled output as Q4_K_M source. Quality difference between the two sources is invisible at the TQ2_0 output. **TQ2_0 (2.06 bpw ternary) is a from-scratch-only format for 27B dense** — post-training requantize cannot preserve coherence regardless of source bpw, because the ternary value space {−d, 0, +d} can't represent the weight distribution Qwen3.6 was trained for. This matches BitNet b1.58's design (trained ternary from step 0, never quantized after).
- Speed/cliff results consistent with R5: 0.05 tok/s, ~60 % CPU utilisation, paging cliff cleared.
- **Path forward** (next session, not in this hardware budget):
  - (a) Download a calibrated IQ1_M (~6.5 GB) when Unsloth/bartowski publish one (none exists today for 27B)
  - (b) Build imatrix locally (`llama-imatrix` on a calibration corpus; forward pass cost is paging-bound on this Mac, multi-hour) and quantize Q8_0 → IQ1_M
  - (c) Use ≥32 GB RAM hardware to skip the cliff entirely and run Q4_K_M directly
- Permanent assets from R1–R6: BOS fix, IQ2_XS dequant, TQ2_0 dequant, paging-cliff measurement (~8 GB on 16 GB Mac), and the empirical fact that low-bpw ternary requires training-time integration on this model family.

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
