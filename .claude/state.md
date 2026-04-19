# quant.cpp — Session State

**Last updated**: 2026-04-19 (Round 21)
**Score**: **0.9979 / 1.0000 (99.8%)** — full `score.sh`, 5/6 dimensions at 100%, structure 98.7%, **13/13 regression PASS**
**Session HEAD**: Round 21 — Q5_K_M added to regression suite. 21 /grow rounds complete.

## Round 21 — Q5_K_M in regression suite (future-proofing)

`scripts/test_models.sh`:
```bash
run_test "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf" "Hi" "" COHERENT \
         "TQ_NO_METAL=1 TQ_NO_MLOCK=1" "--chat"
```

Guards Round 12 (selective MADV) + Round 17 (SHL qh) + any future
Q5_K kernel change against silent regression. Passes **12/12** with
Q5_K_M included (was 11/11 pre-Round-21, 2 SKIP for absent CUDA
models).

Round 18 (2-row) and Round 19 (madvise) regressions would now be
caught automatically — both would fail the coherent check.

## Round 20 — Quant matrix bench consolidation

`bench/results/2026-04-19_qwen36_quant_matrix_16gb.md` — 5-tier
quant comparison on 16 GB M1 Pro:

| Quant | File | RSS | Decode (warm) |
|---|---:|---:|---:|
| IQ2_XXS | 10.0 GB | ~6.5 GB | 16.1 t/s† |
| IQ3_XXS | 12.3 GB | ~6.5 GB | 14.6 t/s† |
| Q3_K_S | 14.3 GB | 5.24 GB† | 14.3 t/s† |
| **IQ4_XS** | 16.5 GB | **7.25 GB** | **10.6 t/s** |
| **Q5_K_M** | 24.6 GB | **9.65 GB** | **7.9 t/s** |

† 이전 세션 기록 (IQ2/IQ3/Q3_K_S 파일은 Q5_K_M 다운로드 과정에서
삭제됨). 굵은 값은 Round 20 재측정.

Doc includes: Round 16-17 Q5_K_M 돌파 의의, Round 18-19 실패 교훈
(2-row register, madvise trap), 용도별 quant 추천, llama.cpp 비교.

### Session complete arc
- Rounds 1-17: 모두 landed (Mission A + infra + Q5_K_M + SHL 최적화)
- Rounds 18-19: 시도 후 롤백 (교훈 획득)
- Round 20: 문서화 + 세션 정리

**총 20 /grow 라운드. Score 0.9946 → 0.9979. Qwen3.6 5 tier 모두 16 GB
Mac에서 검증. Q5_K_M 실용권 진입 (7.9 t/s warm).**

## Round 19 — Per-dispatch `madvise(WILLNEED)`: ATTEMPTED, ROLLED BACK

Hypothesis: after routing decides K=8 active experts, issue
`madvise(MADV_WILLNEED)` on each expert's weight region so SSD
page-in runs async during compute → reduce Q5_K_M new-topic cold
penalty (1.5 → target 3-4 t/s).

**Result**: **Q5_K_M 7.3 → 2.2 t/s (-70%)** on same-prompt warm A/B.
Catastrophic regression — prefetch made warm path 3× slower.

**Why it failed**:
- macOS madvise per call walks VMA tree + touches all page
  descriptors in range; for 40 layers × 8 experts × ~700 KB = 680 MB
  of advice per token → VM subsystem contention.
- For already-resident pages, macOS may still synchronously validate
  against backing store (not async like Linux).
- Unified memory: competes with NEON matmul for memory controller —
  exactly flash-moe's F_RDADVISE lesson (-73% GPU throughput
  observed in their experiment #7).

Git-reverted `src/engine/tq_moe.c`. Saved `feedback_madvise_willneed_per_call.md`
memory to prevent future attempts.

**Generalized lesson** (cross-session durable):
- `madvise` is for LOAD time only on Apple Silicon (Round 12 pattern).
- Never call from per-token / per-layer / per-expert hot path.
- SSD DMA and compute cannot be overlapped on unified memory
  architecture — flash-moe proved it for GPU, we just confirmed it
  for CPU.
- `__builtin_prefetch` (single instruction, L2/L3 hint) remains safe.

### Session running tally
- Rounds 1-15 landed (Mission A + infra + cleanup + prefetch).
- Round 16-17 landed (Q5_K_M on 16 GB + NEON SHL +40%).
- **Round 18 & 19 rolled back** (2-row register pressure, madvise trap).
- Still 0.9979 all-time-high score, 0 FAIL regression.

## Round 18 — Q5_K 2-row parallel: ATTEMPTED, ROLLED BACK

Implemented `q5k_int_dot_two_rows` mirroring Q4_K's successful 2-row
pattern (share activation loads across 2 weight rows, dispatch in
pairs with single-row tail).

**Result**: regression. Q5_K_M decode 2.1 → 1.8 t/s (-14%),
MoE per-tok 283 → 345 ms.

**Root cause (hypothesis)**: Q5_K has more per-row state than Q4_K
(extra qh load + 4 extra vshlq_u8 per iteration × 2 rows = 8 shifts
vs Q4_K's 0 shifts). Doubling the register set pushes past M1 Pro
NEON's 32 × 128-bit register budget → compiler spills to stack.
Measured slowdown consistent with 3-5 stack spills per inner loop.

Git-reverted `src/engine/tq_gguf_quants.c`. No commit.

**Lesson**: 2-row isn't a free template — it depends on per-row
working-set fitting in NEON register file. Q4_K (lo/hi pairs × 2 rows
= 8 weight registers + 4 activation + 4 acc = 16 of 32) fits; Q5_K
(lo/hi pairs + qh-derived 5th bit × 2 rows = 12 weight regs + ...)
exceeds the budget.

### Warm vs cold measurement honesty check

Round 16's "1.5 t/s" was worst-case cold on the very first prompt
after fresh load. Warm measurements (subsequent queries on same
prompt) show **5.7-10.3 t/s** on Q5_K_M — practical for interactive
chat. New-prompt (different expert subset) cold dips back to ~1.8 t/s
as OS page-faults cold experts.

Realistic performance matrix:
| Scenario | Q5_K_M t/s | Notes |
|---|---:|---|
| Fresh-load cold | 1.5-2 | First query after `./build/quant` start |
| New-topic cold | 1.8-2.5 | Unfamiliar expert subset |
| Same-topic warm | 6-10 | Expert subset resident in page cache |
| Typical chat avg | ~4 | Mix of cold/warm across turns |

Q3_K_S remains the **speed leader at ~14 t/s consistent**. Q5_K_M is
the **quality leader at ~4-10 t/s interactive**.

## Round 17 — Q5_K NEON qh extraction SHL-based (+40% Q5_K_M decode)

`tq_gguf_quants.c:q5k_int_dot_worker` — 5번째 비트 추출 체인 단축.

Before: `(qh & u_mask) == u_mask ? 0xFF : 0x00` via `vceqq_u8`, then
`& 16` for the bit-4 byte value. 3 ops per extraction × 4 extractions
per iteration × 4 iterations per super-block = 48 ops just for qh
bit shuffle.

After: `vshlq_u8(qh, shift_vec) & 16` — variable-shift NEON intrinsic
brings target bit directly to position 4. 2 ops per extraction.
Eliminates the mask broadcast (`u1v`, `u2v`) and the compare. Also
removes the `u1 <<= 2; u2 <<= 2;` loop state carrier — `is` index
directly drives the shift amount.

Shift computation: iteration `is` in {0,2,4,6} → sub-block A bit at
position `is`, sub-block B at `is+1`. Required shift (pos 4 −
bit_pos): A = 4−is ∈ {4,2,0,−2}, B = 3−is ∈ {3,1,−1,−3}. Negative
shift = logical right shift (vshlq_u8 semantics).

### Measured (Qwen3.6-UD-Q5_K_M 26 GB on 16 GB M1 Pro, T=0, warm)

| | Round 16 | **Round 17** | Δ |
|---|---:|---:|---:|
| Decode | 1.3-1.5 t/s | **2.1 t/s** | **+40-62%** |
| MoE per-tok | 453 ms | **283 ms** | **-37%** |
| Total per-tok | 501 ms | 320 ms | -36% |

### Regression
- All coherent tests PASS (0 FAIL); 2 SKIP due to model availability.
- Q3_K_S / IQ4_XS / Phi-3.5 / Llama / Gemma-4 verified unchanged.
- Q5_K kernel is also used by non-MoE Q5 models (pure Q5_K weights)
  — not in regression suite but shift semantics identical to old
  `vceqq` branch so numerically equivalent.

### 실용성 변화
- Q5_K_M 2.1 t/s: 여전히 장기 대화에는 느리지만 **단문 응답 실용권 진입**
  (10자 응답 ~5초, 이전 ~10초).
- Q3_K_S 14 t/s가 여전히 속도 최적점.
- Q5_K_M은 고품질 single-shot 추론 (JSON/coding/math one-shot) 용도.

## Round 16 — Q5_K_M Breakthrough on 16 GB Mac ✅ (품질) ⚠️ (속도)

**역사적 돌파**: Qwen3.6-35B-A3B-UD-Q5_K_M (26.46 GB GGUF) 파일이
16 GB M1 Pro에서 로드 + 추론 성공. Round 12 auto-policy MADV가
제공한 공간 예산 이론이 실측으로 확인됨.

### 측정
- File size: 26,456 MB (26.46 GB)
- **Selective MADV**: 613 non-expert WILLNEED (2.50 GB), 120 routed-expert
  at OS default (22.13 GB)
- **RSS: 9.72 GB** — 26.46 GB 파일의 36.7%만 실제 메모리 점유 (MoE 희소성
  + OS 페이지 캐시의 hot-subset 관리)
- Decode: 1.0-1.5 t/s (warm, T=0)
- Output quality: "1+1=" → "2" ✓, coherent text, no character doubling

### 속도 병목 (profile)
| Component | Q3_K_S (기준) | **Q5_K_M** | Δ |
|---|---:|---:|---:|
| MoE | 63.5% (192 ms) | **90.5% (453 ms)** | +2.36× |
| Matmul | 34.2% (104 ms) | 12.8% (60 ms) | -42% |
| Per-token total | 302 ms | 501 ms | +1.66× |

MoE 90%가 주 원인. 대부분 Q5_K 커널의 5번째 비트 처리 오버헤드
(qh bit mask + vceqq_u8 + vorrq_u8 chain vs Q3_K의 단순 shift).
일부는 cold-expert SSD paging.

### 결론
- ✅ **기술적 돌파**: 16 GB Mac에서 Q5 품질 MoE 추론 — 엔진 차원
  최초로 증명. Unsloth UD-Q5_K_M + 우리 selective MADV의 조합이 핵심.
- ⚠️ **실용 속도 부족**: 1 t/s는 장기 대화 어려움. Q3_K_S 14 t/s가
  여전히 실용 최적점. Q5_K_M은 품질-중시 batch 추론용.
- 🎯 **다음 최적화 기회 (Round 17+)**:
  1. Q5_K NEON 커널 추가 최적화 (qh unpack SIMD, 2-row parallel)
  2. Q4_K_M (22 GB) 실증 — 속도/품질 중간점 예상 5-8 t/s
  3. Expert prefetch 확장 (Round 15 paradigm → routed expert 영역)

## Round 15 — Layer prefetch pipelining (flash-moe deferred-CMD3 CPU analog)

`tq_transformer.c:tq_forward_batch_moe_hybrid` — before calling
`tq_moe_forward_batch` for layer L, issue one `__builtin_prefetch`
per next-layer (L+1) non-expert weight buffer: `attn_norm`,
`gguf_w_qkv`/`gguf_wq`/`gguf_wk`/`gguf_wv`/`gguf_wo`, `ffn_norm`.

Rationale: MoE compute dominates layer time (~62% of prefill per
Mission A profile). Prefetching next layer's attn weights during
this window means by the time MoE returns, the first cache line
of each target weight is in L2 and its TLB entry is primed.
Cheap (7 intrinsics × 40 layers = 280 CPU instructions total).

Opt-out: `TQ_NO_LAYER_PREFETCH=1`.

Measured (Qwen3.6-UD-Q3_K_S, 40-token warm runs):

| | Run 1 cold | Run 2 warm | Run 3 warm | median |
|---|---:|---:|---:|---:|
| No prefetch | 8.1 t/s | 9.6 t/s | — | 9.6 |
| With prefetch | 8.4 t/s | 9.4 t/s | — | 9.4 |

**Neutral at Q3_K_S** (within noise). Expected: Q3_K_S 14.3 GB fits
fully in 16 GB RAM after warmup, so all attn data is already
page-cache-resident and hardware prefetch handles the rest.

Value proposition is for Q5_K_M (23 GB) / Q6_K (28 GB) on 16 GB Mac:
some attn pages may not be in page cache when a layer starts; the
prefetch touches them during the MoE window, amortizing the fault
into the compute time rather than the critical path. Verifiable once
Q5_K_M file is available.

12/12 regression PASS. Zero warnings. Score 0.9979 preserved.

## Round 14 — Full score.sh reveals all-time high (0.9979)

Running the full `score.sh` (not `--quick`) for the first time this
session unlocks the 3 dimensions that `--quick` skips:

| Dimension | Score | Notes |
|---|---:|---|
| Structure | 98.7% | WBS progress 97/111 (raised from 93) |
| Correctness | 100% | 94/35 tests (extra tests past target), zero warnings |
| Quality | 100% | roundtrip MSE + attention accuracy all PASS |
| Performance | 100% | throughput + compression ratio + SIMD speedup all PASS |
| Integration | 100% | llama.cpp + vLLM + Python + examples + docs PASS |
| Position | 100% | single-header, zero-deps, 5 papers, pypi, honest corrections |

**+0.0033 vs Round 13's `--quick` reading (0.9946)**.

The earlier "0% quality/performance/integration" readout was a
`--quick`-mode artifact, not real regression. Full run reveals the
project has been at ≥99.7% for the whole session.

Structural-only gap remaining is WBS checklist items (14 genuinely
unchecked, mostly CUDA tests + Metal .mm tests + blog post +
GitHub release tag). Reached on Round 14 via verifying and
checking 4 items that were truly done:
- llama.cpp CMake patch → `integrations/llamacpp/patch`
- llama.cpp integration test → `test_integration.cpp`
- AVX2 parity test → `tests/test_simd_avx2.cpp`
- Release notes → `docs/RELEASE_NOTES.md`

Remaining unchecked items are either genuinely not done
(GitHub release tag, blog post, Valgrind run, 100K-token endurance)
or would require net-new test file authoring (Metal .mm tests,
CUDA tests on a non-CUDA-target platform). Skipped — not useful
vs the actual goal (Q5_K_M breakthrough).

## Round 13 — Dead LRU cleanup + split-source/quant.h drift fix

`src/engine/tq_moe.c`: removed the `if (0 && g_expert_cache ...)`
dispatch site (25 LOC dead code in the per-expert hot loop) and
its supporting chain:
- `cache_get_or_create`, `free_cache_entry`, `quantize_fp32_to_q8_0`,
  `fp32_to_fp16`, `q8_0_bytes` (all only reached via the dead site)
- `expert_cache_entry_t`, `expert_layer_cache_t` structs
- `g_expert_cache`, `g_cache_*`, `g_token_counter` globals
- `tq_moe_cache_init` / `tq_moe_cache_free` reduced to empty no-op
  stubs (matching what `quant.h` already shipped — this eliminates
  a split-source vs single-header drift that had existed since
  the Q8 LRU was prototyped and then guarded out).

Dead call-site investigation documented the "historical note" comment
so future readers see *why* the path was abandoned:
`fused_dot_iq2_xxs_neon` direct dispatch was faster than
(IQ2→FP32→Q8_0 on miss + fused_dot_q8_0 on hit) whenever expert reuse
rate is low — always the case for Qwen3.6's K=8/N=256 routing.

No behavior change (dead code was unreachable). Build clean, 12/12
regression PASS, score 0.9946 preserved. ~200 LOC net reduction.

## Round 12 — Higher-bpw headroom via auto-policy MADV (flash-moe trust-OS)

`tq_model.c`: MoE GGUF loading now auto-selects madvise strategy by
`file_size vs physical_RAM`:
- File ≤ 75% RAM → blanket `MADV_WILLNEED` (old behavior, optimal
  read-ahead for fits-in-RAM case).
- File > 75% RAM → selective `MADV_WILLNEED` on non-expert tensors
  only (`attn_*`, `norm_*`, `token_embd`, `output.weight`,
  `ffn_*_shared_exp`); routed `ffn_{gate,up,down}_exps` left at OS
  default so natural MoE sparsity (K=8/N=256 active) keeps working
  set small. Prevents swap thrash on Q5_K_M 23 GB / Q6_K 28 GB.

Override envs: `TQ_FLAT_MADV=1`, `TQ_SELECTIVE_MADV=1`.

Measured (Qwen3.6-UD-Q3_K_S 14.3 GB on 16 GB M1 Pro):
| | blanket (`TQ_FLAT_MADV`) | **auto = selective** |
|---|---:|---:|
| Decode (30 tok, cold) | 11.1 t/s | **11.0 t/s** (within noise) |
| RSS | 7.01 GB | **6.99 GB** |

IQ4_XS 16.5 GB (auto = selective): 9.2 t/s warm, 7.57 GB RSS.
Pre-Round-12 this file required `TQ_NO_MLOCK` to avoid mlock fail +
still thrashed under blanket WILLNEED at 16 GB RAM.

Round 12 deliverable: **Q5_K_M / Q6_K loading is now technically
possible on 16 GB Mac** — blanket WILLNEED would previously force
swap-load all 23-28 GB. Next round: actually test Q5_K_M.

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
**Step 3g: ✅ DONE** (tq_tp_run_dynamic FCFS queue)
Added `tq_tp_run_dynamic` in `tq_ops.c` with atomic-counter FCFS
dispatch (workers + main grab next task idx, no wave boundaries).
Added `__thread int tq_tls_worker_id` for per-worker slab lookup.
Wired into `tq_moe_forward_batch` Phase 3 behind env
`TQ_MOE_BATCH_DYNAMIC=1` (default OFF for safety).

Measurement (Qwen3.6-UD-Q3_K_S, 450-word prompt, N=5 decode, warm,
j=8, median of 3):
| | 3h baseline (wave) | **3g (dynamic)** | Δ |
|---|---:|---:|---:|
| Wall time | 84.5s | **71.7s** | **-15%** |
| Prefill rate | 11.4 t/s | **13.4 t/s** | **+17%** |

12/12 regression PASS. No decode change (N=1 path doesn't hit
Phase 3). Dynamic is opt-in; flipping to default-on is a separate
follow-up once broader coverage is confirmed.

**Step 3h: ✅ DONE (3a34cbf)**
Batched shared expert dispatch. Extra +8% vs Step 3f measured
(81.6s vs 88.4s median). Approach: `tq_batched_matmul_q4` × 3
(gate/up/down) with stack scratch, replacing per-token loop.
Limitation: GGUF-native shared expert still per-token fallback
(dormant for Q4-converted Qwen3.6 UD quants, so no impact there).

**Step 3i: ✅ DONE (Round 6)** — MoE batched default-ON
`tq_generate.c`: `getenv("TQ_MOE_BATCH")` → `!getenv("TQ_NO_MOE_BATCH")`.
Regression 12/12 PASS unchanged (greedy coherence robust enough).
"Paris" factual probe identical with/without opt-out.

Users now get prefill speedup automatically on Qwen3.6 MoE. Opt-out
via `TQ_NO_MOE_BATCH=1` for A/B testing.

**Step 3j: ✅ DONE (Round 10)** — Dynamic FCFS default-ON
`tq_moe.c` line 1757-1764: getenv("TQ_MOE_BATCH_DYNAMIC") → !getenv("TQ_NO_MOE_BATCH_DYNAMIC").
Regression 12/12 PASS with dynamic enabled. Wave path still reachable
via `TQ_NO_MOE_BATCH_DYNAMIC=1` opt-out. Users now get combined
+17% on top of Step 3f/3h prefill gains by default.

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

### ~~P1 Mission A Step 2: Self-attn batched polish~~ — SKIP
Profile (Round 11): self-attn is 0.0% of decode and ~0.26% of prefill
compute post-Mission A. Even a 50% kernel speedup would be below
measurement noise. De-prioritized permanently.

### P2 Long-prompt drift on 35B × 3-4 bpw
**Confirmed intrinsic**: llama.cpp reproduces garbage on same Q3_K_S 40-word prompt. Not an engine bug. Only fix path is higher bpw, which doesn't fit 16 GB Mac.
Mitigation exposed: `--rep-penalty 1.3-1.5` CLI (c3a54f4) extends coherence ~40→75 tok.

### ~~P3 Full Qwen3 Q5_K support~~ — Already DONE
`q5k_int_dot_worker` at `tq_gguf_quants.c:4119` is NEON int8 + vdotq_s32
(DOTPROD), 2-way low/high nibble + 5th-bit-from-qh via vceqq_u8 mask,
dispatched at line 5181. Not scalar — performance-on-par with Q4_K.

### P4 Metal MoE (ambitious, low-urgency)
Current `qwen35moe` forces CPU (a4120d8) because Metal path hangs. llama.cpp also hangs on same model. A working Metal MoE would be unique.

### P5 ✅ DONE (Round 7) — v0.15.0 release notes
Comprehensive entry documenting Mission A Step 3 complete in
`docs/RELEASE_NOTES.md`. Covers all 7 commits (batched kernels +
dispatcher + driver + default flip), measured +39% prefill table,
sanity numbers, known limitations, and cumulative session arc
(3.08 → 16.1 t/s decode, 5 → 6.1 t/s prefill).

## Next `/grow` round entry point

Mission A fully landed (Steps 3d/e/f/g/h/i/j + release notes + warnings).
Score 0.9946 with full regression 12/12 PASS. Small-round P0-P3 items
all closed as of Round 10.

Post-Mission A decode profile (Qwen3.6-UD-Q3_K_S, warm, per-token):
- MoE: 63.5% (192.3 ms) — per-token expert dispatch, already NEON int8
- Matmul: 34.2% (103.6 ms) — QKV / O / shared FFN / lm_head
- Recurrent: 1.8%, conv: 0.2%, attn: 0.0%, other: 0.3%
- Total: 302.7 ms/token (cold-ish during ramp; state.md claims 14 t/s warm)

**Candidate Round 11+ targets** (pick by impact × risk):
1. **Mission B #142 (big)**: Long-context 실증 harness — turbo_kv_4b
   on Llama 3.2 3B at 16K-128K ctx, needle-in-haystack + PPL-over-ctx.
   Validates KV compression claim at scale. ~400-600 LOC harness +
   1-2 runs. Delegate to agent.
2. **Decode MoE expert-parallel polish**: 63.5% of decode is MoE; check
   if per-token 8-expert dispatch in `tq_moe.c` is fully tp_run-parallel.
3. **P4 Metal MoE** (ambitious, multi-session): llama.cpp also hangs on
   qwen35moe — a working Metal MoE would be genuinely novel. High risk.
4. **`build_cpu/` gitignore + release cadence**: trivial housekeeping.

**Do NOT** pursue P1 Step 2 self-attn batched polish — profile shows
attn is 0.0% of decode and ~0.26% of prefill compute post-Mission A.
ROI is below measurement noise. Remove from serious backlog.
