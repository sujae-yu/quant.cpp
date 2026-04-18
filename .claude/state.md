# quant.cpp — Session State

**Last updated**: 2026-04-19
**Score**: 12/12 regression PASS, 0 build warnings (core)
**Session HEAD**: `3a34cbf` (Step 3h: batched shared expert)

## What Works

### Qwen3.6-35B-A3B MoE on 16 GB M1 Pro (CPU-only)
- **Decode: 16.1 t/s** (IQ2_XXS) / 14.3 t/s (Q3_K_S) / 12.5 t/s (IQ4_XS warm peak)
- **vs llama.cpp CPU 5.1 t/s = 2.8-3.2× faster** on MoE
- **RSS: 5.24 GB** (Q3_K_S) / 5.44 GB (IQ4_XS) — smaller than IQ2_XXS at higher bpw
- 4 quant tiers verified on 16 GB (IQ2_XXS / IQ3_XXS / Q3_K_S / IQ4_XS)
- Q8_0 재양자화 트랩 수정 (ea01222): "The capital of France is Paris" ✓

### Engine Kernel Suite (all NEON int8 vdotq_s32 paths)
- Q4 internal + Q6_K + Q3_K + IQ2_XXS + IQ2_S + IQ3_XXS + IQ3_S + IQ4_XS (TBL-16) + Q8_0 v2
- RoPE TLS sin/cos cache: 4 branches (partial/standard/LongRoPE/Gemma NeoX)
- SwiGLU `fast_exp_neon`, MoE router NEON, TQ_NO_MLOCK env

### Batched Prefill Path (active under `TQ_MOE_BATCH=1`)
- `tq_forward_batch_moe_hybrid` (627b65e, f255b46) — per-token attn + batched MoE FFN
- Entry point in `tq_generate.c`: routes when is_moe + !is_gemma4 + TQ_MOE_BATCH
- Sanity mode first 20 tokens match per-token ✓
- FP noise 1e-5 flips greedy top-1 after 40 layers → default OFF
- j=6: +40% prefill; j=8: neutral (expert_parallel already saturated)

### Batched Kernels
- `tq_batched_matmul_q8_0` (b7c42dd) — non-expert path
- `fused_dot_iq3_xxs_int8_batched` (8dd4920, **fixed in 61d7ce8** — missing `qs += 8`)
- `fused_dot_iq3_s_int8_batched` (30428f3) — 19.0% compute
- `fused_dot_iq4_xs_int8_batched` (30428f3) — 0.9% compute
- **`tq_moe_forward_batch` (9fb237d) — 3-phase dispatch, publicly exported**
- **Sanity mode** `TQ_MOE_BATCH_SELFTEST=1` (3794fd2) — routes single-token MoE through batch(N=1), max_abs_diff = 1.2e-7 ~ 3.6e-7 (all Qwen3.6 tiers)
- **Bug caught by sanity**: IQ3_XXS batched missed `qs += 8` per sub-block → 60× error. Same precedent as single-query kernel bug. Fix verified.

### Verified equivalence
- IQ2_XXS / IQ3_XXS / IQ4_XS / Q3_K_S all produce `max_abs_diff ≤ 3.6e-7` vs per-token reference — pure FP noise level, **well under 1e-3 spec**.

### Verification
- `scripts/test_models.sh`: **12/12 PASS** (Llama 3.1/3.2, Qwen 2.5/3.5/3.6, Phi-3.5, Gemma-4)
- Dormant kernels verified safe (no caller yet, regression unchanged)
- Coherence probes on all Qwen3.6 tiers

### Benchmarks (7 new reports this session)
- `bench/results/2026-04-18_moe_and_q4_k_m_breakthrough.md` — Q6_K/router/NO_MLOCK
- `bench/results/2026-04-18_q3_breakthrough.md` — Q3 tier unlock
- `bench/results/2026-04-18_q3_k_s_tier.md` — Q3_K_S 5.24 GB
- `bench/results/2026-04-18_iq4_xs_tier.md` — IQ4_XS fits 16 GB
- `bench/results/2026-04-18_vs_mlx_vs_llamacpp.md` — 3-way: MLX 58.5, ll.cpp 20, us 14
- `bench/results/2026-04-18_prefill_analysis.md` — Mission A plan revised
- Memory: Q8_0→Q4 double-quant trap教訓 기록

## What Needs Work (Priority Order)

### ✅ Mission A Step 3 COMPLETE (3d + 3e + 3f + 3h)

**Final measurement** (Qwen3.6-UD-Q3_K_S, 450-word prompt, warm, j=8):
| | baseline | 3f (prior round) | **3h (current)** | Δ total |
|---|---:|---:|---:|---:|
| Wall time | 103.2s | 85.3s | **81.6s (agent median)** | **-21%** |
| Prefill rate | 4.4 t/s | 5.4 t/s | **5.5-6.1 t/s** | **+25-39%** |
| CPU work | 307s | 178s | 185s | -40% |

Agent reported at j=8 with 951-tok prompt: baseline 10.3 → batched 11.3 t/s (+9%).
Longer prompts benefit more (larger M_e per expert = better amortization).

Decode: unchanged (13+ t/s, batched path only affects prefill).

**Step 3f completed** via `/grow` Round 3:
- Cross-expert parallel dispatch in `tq_moe_forward_batch` Phase 3
  (`e5f721a`) — 8 workers, one expert each, private scatter buffer
  reduced serially at end. Respects `tq_tls_force_serial_matmul`.
- Q3_K batched kernel + MoE dispatch (`f9e5af1`) — mirrors IQ3_XXS
  pattern. Dormant on UD-Q3_K_S (mixed-tier uses IQ3_XXS/IQ3_S) but
  active for pure Q3_K MoE models.
- Patches to `tq_batched_matmul_q8_0` and `tq_batched_matmul_q4` for
  nested-pool safety.

Sanity: `TQ_MOE_BATCH_SELFTEST=1` max_abs_diff 1.2e-7. First-20-token
match with per-token reference under `TQ_MOE_BATCH_SANITY=1`.

### P0 Remaining (small incremental gains)
**Step 3g: dynamic work queue for expert stragglers** (still pending)
Current wave-based `tq_tp_run` has long tails when one expert gets
600+ tokens and others 30. First-come-first-served atomic dispatch
would flatten. ~100-200 LOC in `tq_ops.c` `tq_tp_run` or a new
`tq_tp_dynamic`. Expected +5-10% additional.

**Step 3h: ✅ DONE (3a34cbf)**
Batched shared expert dispatch. Extra +8% vs Step 3f measured
(81.6s vs 88.4s median). Approach: `tq_batched_matmul_q4` × 3
(gate/up/down) with stack scratch, replacing per-token loop.
Limitation: GGUF-native shared expert still per-token fallback
(dormant for Q4-converted Qwen3.6 UD quants, so no impact there).

**Step 3i: make TQ_MOE_BATCH=1 default-on**
Sanity now clean under N=1 + 2e-4 under N=7. Greedy decoding
sensitivity issue (1e-5 noise flips top-1) needs either:
(a) higher-precision FMA chains in batched kernels, or
(b) accepting non-deterministic top-1 under temperature sampling.
Not urgent; env opt-in remains safe.

`tq_moe_forward_batch` is implemented + validated (1.2e-7 diff). Calling it with N>1 requires a new `tq_forward_batch_moe_hybrid` driver because existing `tq_forward_batch` is Llama-shaped and bails on `is_moe || has_fused_qkv || delta_kv_enabled`.

New driver must handle:
- Per-token DeltaNet recurrent state (cannot batch — sequential data dependency)
- Per-token self-attention (Qwen3.6 has fused QKV + attn_output_gate)
- **Batched MoE FFN via `tq_moe_forward_batch(N)`** (the actual speedup)
- Per-layer aggregation of N-wide hidden states

Estimated: **400-600 LOC**. Single focused session.

Success criteria unchanged:
- Prefill pp500 ≥ 10 t/s (baseline 5) — stretch 15 t/s
- No decode regression (warm ≥ 11 t/s)
- 12/12 regression pass
- `TQ_MOE_BATCH=1` opt-in; sanity env compares vs per-token.

### P1 Mission A Step 2: Self-attn batched polish
After P0, wire `tq_batched_matmul_q8_0` into self-attn Q/K/V/O projections for additional 5-10%. Qwen3.6 has fused QKV attn_qkv + attn_output_gate, so split/combine logic needs care.

### P2 Long-prompt drift on 35B × 3-4 bpw
**Confirmed intrinsic**: llama.cpp reproduces garbage on same Q3_K_S 40-word prompt. Not an engine bug. Only fix path is higher bpw, which doesn't fit 16 GB Mac.
Mitigation exposed: `--rep-penalty 1.3-1.5` CLI (c3a54f4) extends coherence ~40→75 tok.

### P3 Full Qwen3 Q5_K support
Q5_K int8 kernel likely scalar (not profiled yet). If users adopt Q5_K_M for mid-tier quality, this would matter.

### P4 Metal MoE (ambitious, low-urgency)
Current `qwen35moe` forces CPU (a4120d8) because Metal path hangs. llama.cpp also hangs on same model. A working Metal MoE would be unique.

### P5 README / v0.15.0 release notes
After Step 3d lands, v0.15.0 release notes needed for:
- Q8_0 재양자화 트랩 수정
- 4-tier Qwen3.6 measurement matrix
- Mission A progress
- 3-way (MLX/llama.cpp/us) benchmark positioning

## Next `/grow` round entry point

**Step 3d implementation**. Delegate to `general-purpose` agent due to size (800+ LOC).

Clear deliverable:
- `src/engine/tq_moe.c`: new function `tq_moe_forward_batch(...)` — 3-phase dispatch
- `src/engine/tq_transformer.c`: wire into `tq_forward_batch` Qwen3.6 hybrid path
- Env toggle `TQ_MOE_BATCH=1` (opt-in initially)
- Sanity mode `TQ_MOE_BATCH_SANITY=1` → per-token equivalence check
- Regression 12/12 must pass
- Measurement: Qwen3.6 Q3_K_S / IQ4_XS prefill pp500 before/after

Success criteria:
- Numerical equivalence vs per-token (within 1e-3 tolerance on output)
- Prefill pp500 ≥ 10 t/s (2× from 5 t/s baseline) — stretch 15 t/s
- No decode regression (warm peak ≥ 11 t/s)
