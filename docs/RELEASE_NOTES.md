# Release Notes

All notable changes to quant.cpp are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.20.0] — 2026-04-20 ★★ (NEOX RoPE ROOT-CAUSE — Qwen3 Long-Prompt Fix)

### Headline

**Two transformer-level bugs that blocked Qwen3 family long-prompt
coherence are fixed.** Combined with v0.19.0's BPE tokenizer fix,
all three root causes of the 30+ round "Qwen3 drift" investigation
(R26-R50) are now closed. Discovered via HF reference diff
methodology (`tools/pillar1/`) after `refs/OpenMythos` analysis
crystallized the principle: compare to ground truth FIRST.

### Two fixes

**Fix 1 — Pure-Qwen3 QK-norm restored** (`tq_transformer.c:1204`):

R40 had disabled QK-norm for ALL GGUF arch strings matching "qwen".
That was correct for Qwen3.5/3.6 HYBRID (DeltaNet + self-attn,
`delta_n_heads > 0`) — those degrade with QK-norm applied. But
pure Qwen3 (0.6B..32B) REQUIRES q_norm/k_norm per HF config. Without
them, the residual stream explodes at layer 2 (norm ~5400 vs HF ~10).

Fix: restrict the QK-norm disable to `delta_n_heads > 0` only.
Pure Qwen3 now applies QK-norm as HF does.

**Fix 2 — NEOX-ordering RoPE** (`tq_ops.c` + two sites in
`tq_transformer.c`):

llama.cpp maps `LLM_ARCH_QWEN3 / QWEN3MOE / QWEN35 / QWEN35MOE` to
`LLAMA_ROPE_TYPE_NEOX / IMROPE` — half-split pairs `(q[i],
q[i+head_dim/2])`. Our engine used LLaMA-style interleaved pairs
`(q[2i], q[2i+1])`. R34 had fixed this for the partial-rotary path
(Qwen3.5/3.6 hybrid) but pure Qwen3 (full rotary) and
`tq_forward_batch` were never converted.

Fix: new `tq_rope_neox` function + arch-detection at all three
relevant call sites. Per-token full-rotary, batched learned-freq,
batched fallback. TQ_ROPE_PAIRS=1 opt-out for legacy LLaMA/Qwen2.

### Symptom (before/after, Qwen3-0.6B Q4, 50-word synthetic input)

| Path | Before | After |
|---|---|---|
| Batched prefill | `alyticsÐ°Ð½cieaâ��à¹�…` UTF-8 garbage | `" Let me try to understand this"` |
| Per-token prefill | `lenameuously…catchØ�` | `" ... and so on… So, the problem is to find the number of possible ways"` |

### Natural prose — 31 words, "Summary:" continuation

| Model | Output (first 20 tok) |
|---|---|
| Qwen3-0.6B | `"The main features of AI technology are that it has the ability to process information…"` ✓ |
| Qwen3.5-4B | `"Artificial intelligence is a field of computer science that focuses on the development of intelligent machines…"` ✓ |

### Qwen3.6-35B broad validation (8-prompt matrix, 40+ words max)

- Zero UTF-8 garbage outputs (was 100% on 40+ words before v0.19.0).
- Short story, long essay, tech explanation, factual Q&A all coherent.
- Remaining weak spots are chat-template-induced early EOS (0 tokens
  on some raw-completion prompts) — model behavior, not engine bug.

### Methodology — OpenMythos insights applied

`refs/OpenMythos` (RDT / MLA / ACT architecture reconstruction)
crystallized the principle that ENABLED this session's breakthroughs:

> Compare to ground truth (HF reference diff) BEFORE guessing at
> kernels or recurrence state. 30+ rounds R26-R50 had all been
> empirical; Pillar 1 R1-R3 + Pillar 1.5 R1-R3 solved three distinct
> root causes in 6 rounds by diffing against HF output.

Saved as `memory/project_openmythos_insights.md` for future sessions.

### Files changed

- `src/engine/tq_tokenizer.c` — BPE stale-entry check (v0.19.0 fix retained)
- `src/engine/tq_transformer.c` — QK-norm scope + NEOX in 2 call sites
- `src/engine/tq_ops.c` — new `tq_rope_neox` function
- `include/turboquant/tq_engine.h` — export `tq_rope_neox`
- `scripts/test_models.sh` + `scripts/test_tokenizer.sh` — regression expanded
- `tools/pillar1/` — HF reference diff toolchain retained for follow-on
- `bench/results/2026-04-20_bpe_fix_proof.md` — before/after evidence
- `bench/results/2026-04-20_longseq_transformer_bug.md` — R7/R8 discovery trail

### Regression

- `test_models.sh`: **15/15 PASS** (unchanged through both fixes)
- `test_tokenizer.sh`: 4/4 PASS

### Known remaining

- **Qwen3.6-35B DeltaNet state accumulation** on 40+ word natural
  prose can sometimes trigger repetition-loop detection. This is
  separate from the RoPE/QK-norm bugs and needs OpenMythos Insight
  #2 (spectral-radius monitoring of recurrent state) applied as
  diagnostic. Short-medium prompts fully coherent.
- Chat-template interactions producing 0-token responses on some
  coding prompts (Qwen3.6's thinking-mode prefix consuming the tokens).

### Compatibility

No API change. Existing code using `tq_rope` continues to work for
LLaMA/Qwen2. New `tq_rope_neox` opt-in for Qwen3 family (auto-
detected via GGUF arch string).

---

## [v0.19.0] — 2026-04-20 ★ (BPE Stale-Entry ROOT-CAUSE Fix)

### Headline

**One-line fix to `src/engine/tq_tokenizer.c:1442` eliminates the
structural tokenization bug that caused every "Qwen3 drift" symptom
across 30+ rounds of kernel/MoE/DeltaNet investigation.** Pillar 1
of the Mission E roadmap, closed in 3 rounds via HF reference diff.

### The fix

```c
  if (top.gen != gen[top.pos]) continue;
+ if (tokens[top.pos] < 0) continue;   // ★ missing dead-slot guard
  int ri = next[top.pos];
  if (ri >= n_tokens || tokens[ri] < 0) continue;
```

**Root cause**: In the heap-based BPE merge loop, a position `P` that
dies as the RIGHT neighbor of some other merge has `tokens[P]` set to
-1 but `gen[P]` is **not** bumped. Stale heap entries at position `P`
pass the gen-based staleness check, then the code overwrites dead
`tokens[P]` with a new merge result — resurrecting the slot, scrambling
the linked list, and producing malformed token sequences.

### Symptom (same prompt, before/after)

| | Tokens for "Hello" | Decoded |
|---|---|---|
| HF reference | `[9707]` | "Hello" |
| Our engine BEFORE | `[32713, 654]` | **"Helll"** (extra 'l', lost 'o') |
| Our engine AFTER | `[9707]` | "Hello" ✓ |

### What this fixes (consolidated)

| Symptom (previous attributed cause) | Actual cause |
|---|---|
| Qwen3.5/3.6 "quicck bbrrown" char doubling | tokenizer |
| Qwen3.6-35B ≥40-word prompt → UTF-8 garbage | tokenizer |
| Phi-3.5 "What is 2+2?" → hallucinating "tti" | tokenizer |
| R32 Mission C "drift is Qwen-common architecture" | WRONG — was tokenizer |
| R46-50 Mission D "structural bug needs HF Python diff" | correct diagnosis; R3 finishes it |

### Validation

- **Regression**: 15/15 `test_models.sh` + new `test_tokenizer.sh` 4/4
- **Real output**: Qwen3.6-35B on 40+ word prompts produces coherent
  Python code and full narrative text (previously garbage)
- **Phi-3.5**: "What is 2+2?" → "The sum of 2 and 2 is equal to four."
  (previously "I'm sorry but 'tti' doesn't appear to...")

### Methodology (the actual insight)

Pillar 1 R1-R3 built Python + HF Qwen3-0.6B FP32 reference env
(`tools/pillar1/`) specifically to enable per-layer diff debugging.
Before the first layer diff was ever needed, just comparing
**tokenizer output** revealed the mismatch. The entire transformer
investigation from R26-R50 had been working with corrupted input.

**Lesson**: When debugging LLM coherence, compare tokens to HF
reference FIRST. Don't "rule out" the tokenizer without actually
running `AutoTokenizer.encode(prompt)` side-by-side.

### Files changed

- `src/engine/tq_tokenizer.c` — 1-line fix + comment
- `src/engine/tq_transformer.c` — env-gated per-layer dump
  (`TQ_DUMP_HIDDEN=dir`) retained as debugging infrastructure
- `scripts/test_models.sh` — Phi-3.5 expected "answer" → "sum"
  (Phi-3 now gives actual factual math answer)
- `scripts/test_tokenizer.sh` — **NEW** 4-test regression guard
- `tools/pillar1/` — HF reference env + hf_dump.py dump tool
- `bench/results/2026-04-20_bpe_fix_proof.md` — full before/after proof

### Non-impact

- `quant.h` (single-header): uses naive O(n²) BPE merge, correct by
  construction. Embed/WASM users have NEVER hit this bug. Only the
  split-source engine needed the fix.
- No API change.
- No performance change (the stale check is O(1)).

### Compatibility

No migration needed. Users of prior versions will simply see coherent
output on previously-broken prompts. All existing models work.

---

## [v0.18.0] — 2026-04-20 (Daily-Driver UX — TTFT/decode split)

### Headline

**CLI now reports TTFT and decode rate separately**, replacing the
blended "overall tok/s" that dominated short-query reports with
cold-start latency. Individual developers evaluating the engine on
a 16 GB Mac can now distinguish prefill cost from sustained decode.

### Changes

**R51 — TTFT measurement** (`tools/quant.c`):
`print_token` callback records first-token timestamp via a
`cli_timing_ctx_t` struct passed through `user_data`. After
`tq_generate` returns, the summary line splits:

```
TTFT 0.99s | decode 29 tok in 1.99s (14.6 tok/s) | total 3.0s (10.1 tok/s overall)
```

Fallback to the old single-line format when `n_generated ≤ 1`.
25 LOC total. No engine behavior change.

**R52 — Daily-driver baseline matrix** (`bench/results/2026-04-20_ttft_daily_driver.md`):
Measured warm numbers on 16 GB M1 Pro CPU-only:

| Model | Warm TTFT | Warm decode |
|---|---:|---:|
| Phi-3.5 Q4_K_M (3.8B) | 2.3s | **14.5 t/s** |
| Llama-3.2-3B Q8→Q4 | 0.97s | **29.0 t/s** |
| Qwen3.6-35B IQ4_XS | 1.83s | **10.5 t/s** |

Cold first-run Qwen3.6 TTFT is 9.6s (5.3× warm) due to cold SSD
paging; subsequent runs benefit from macOS page cache.

**R53 — README v3.11 blurb** (`README.md` + `README.ko.md`):
External discoverability — first-visit users see the TTFT/decode
framing and warm numbers immediately above the v3.10 correctness
entry.

### Why "decode is a model property, TTFT is a warmup property"

For short queries (-n 30), cold-start mmap + MADV traversal +
transformer pass #1 can dominate wall time. Reporting only the
blended rate understates engine compute speed. Example:

- Qwen3.6-35B cold: total 19.3s → 1.6 t/s overall
- Qwen3.6-35B warm: total  4.6s → 6.5 t/s overall, **10.5 t/s decode**

The engine isn't 6× faster warm; it's doing the same compute. TTFT
just dropped from 9.6s to 1.8s. Individual devs need to see both.

### Compatibility

No API change. Existing `tq_gen_config_t::on_token` callbacks with
`user_data = NULL` continue to work — `user_data` is opaque from
the library's perspective; the CLI passes its own timing struct.

### Regression

15/15 PASS (unchanged). Existing COHERENT + STRICT + Metal-ON
tier tests traverse new stderr format without breakage.

---

## [v0.17.0] — 2026-04-20 (Qwen3.x Correctness — ALL FORMATS WIN)

### Headline

**Qwen3.5 / Qwen3.6 long-form generation restored across all formats.**
Two structural fixes (R34 NEOX RoPE + R40 arch-conditional QK-norm)
eliminate prompt-sensitive drift that manifested as "quicck bbrrown"
char doubling, digit/alphabet spam, and incoherent output past ~20 tokens.

### Fix 1 — NEOX partial RoPE (R34)

`refs/llama.cpp/src/llama-model.cpp:9298` maps
`LLM_ARCH_QWEN35/QWEN35MOE → LLAMA_ROPE_TYPE_IMROPE`. `ggml.h:1826`:
"NEOX ordering cannot be disabled for MROPE/VISION" (IMROPE is MROPE
family). Our partial-rotary path used LLaMA-style `(q[2i], q[2i+1])`;
Qwen3.x requires NEOX half-split `(q[i], q[i+rope_pairs])`. Opt-out:
`TQ_ROPE_PAIRS=1`.

### Fix 2 — arch-conditional QK-norm (R40)

Gemma 4 REQUIRES QK-norm (2+2=4 test breaks without). Qwen family
DEGRADES with QK-norm applied (40+ token drift). Empirical:
`Qwen3.6 "Once upon a time" n=60` → drift WITH, perfect story WITHOUT.
Fix: arch detection gates QK-norm; Gemma keeps, Qwen skips.
Opt-in: `TQ_FORCE_QK_NORM=1`.

### Measured (Qwen3.6-UD-IQ4_XS, T=0, --chat)

| Format | v0.16 | **v0.17** |
|---|---|---|
| "Once upon a time" n=60 | drift at ~20 tok | **60 tok Jack/compass/map story** |
| `def fibonacci(n):` | garbled | **`if n <= 0: return "Invalid input"`** |
| Haiku | char doubling | **"Silence speaks loud, Silence speaks in the quietest way."** |
| List 5 items | partial | **"1. Apple 2. Banana 3. Orange"** |
| Factual | ✓ | **"Paris"**, "12", "1945" |

### Numerical cleanup (R26-29)

Supporting fixes now complete: `l2_normalize` epsilon; DeltaNet
softplus/decay/silu + attention sigmoid + conv1d silu all use exact
`expf()` (was Schraudolph `fast_expf` with ~2% error, compounding
across 30 DeltaNet layers).

### Regression 15/15 PASS

Added R41 long-form guards: "Once upon a time" n=40 + "def fibonacci"
n=30 strict substring checks. Gemma/Llama/Phi unchanged.

### Impact

Qwen3.6-35B-A3B on 16 GB M1 Pro: v0.16 loaded Q5_K_M, v0.17 sustains
coherent long generation across story/code/poem/list/explanation/Q&A.

### Session arc (40 rounds → v0.17)

- R1-15 Mission A (MoE batched +39%)
- R16-17 Q5_K_M 16 GB load
- R25-33 drift discovery + partial fixes
- **R34 NEOX RoPE structural fix**
- **R40 arch-conditional QK-norm — ALL FORMATS WIN**
- R41 long-form regression
- R42 v0.17.0 release

---

## [v0.16.0] — 2026-04-19 (Q5_K_M on 16 GB Mac + auto-policy MADV)

### Headline

**Qwen3.6-35B-A3B at Q5 quality (5.5 bpw) on 16 GB M1 Pro** — first
engine to load the 26.5 GB Q5_K_M GGUF and produce coherent decode
at 7.9 t/s warm steady-state on a 16 GB Mac. Previously impossible
(llama.cpp + Q5_K_M OOMs on the same hardware).

### The key mechanism — auto-policy MADV (Round 12)

`tq_model.c` MoE GGUF loader now selects madvise strategy by
`file_size vs hw.memsize`:

- **File ≤ 75% RAM**: blanket `MADV_WILLNEED` (previous behavior;
  optimal read-ahead, no swap risk).
- **File > 75% RAM**: selective `MADV_WILLNEED` on non-expert tensors
  only (`attn_*`, `norm_*`, `token_embd`, `output.weight`,
  `ffn_*_shared_exp`). Routed `ffn_{gate,up,down}_exps` left at OS
  default (NORMAL with read-ahead). MoE sparsity (K=8/N=256 active)
  keeps working set bounded.

Result for Qwen3.6-UD-Q5_K_M (24.6 GB):
- Non-expert WILLNEED: 2.50 GB
- Routed-expert OS-managed: 22.13 GB
- **RSS: 9.65 GB** (36.7% of file) on 16 GB M1 Pro
- Decode warm steady-state: **7.9 t/s** (interactive range)

Override envs: `TQ_FLAT_MADV=1`, `TQ_SELECTIVE_MADV=1`.

### Q5_K NEON kernel optimization (Round 17)

`q5k_int_dot_worker`: 5th-bit extraction chain shortened from
(AND + CEQ + AND + OR) to (SHL + AND + OR) using variable-shift
`vshlq_u8` with runtime shift vector. Target bit moved directly to
position 4 via single shift — saves one instruction per qh
extraction.

- Before: 1.5 t/s cold (Round 16)
- After: **2.1 t/s cold (+40%)**, 7.9 t/s warm (+5-10× after cache warm)

### Full Qwen3.6-35B-A3B quant matrix on 16 GB Mac

| Quant | File | RSS | Decode (warm) |
|---|---:|---:|---:|
| IQ2_XXS | 10.0 GB | ~6.5 GB | 16.1 t/s |
| IQ3_XXS | 12.3 GB | ~6.5 GB | 14.6 t/s |
| Q3_K_S | 14.3 GB | 5.24 GB | 14.3 t/s |
| IQ4_XS | 16.5 GB | 7.25 GB | 10.6 t/s |
| **Q5_K_M** | **24.6 GB** | **9.65 GB** | **7.9 t/s** |

vs llama.cpp CPU 5.1 t/s (Q3_K_S): **2.8-3.2× faster** across tiers.
llama.cpp can't load Q5_K_M on 16 GB Mac at all.

### Other improvements

- **Layer prefetch pipelining** (Round 15): `__builtin_prefetch` on
  next-layer non-expert weights during current MoE compute. Neutral
  on fits-in-RAM quants (Q3_K_S), positive on Q5_K_M page-cache
  pressure. TLB priming benefit.
- **Dead LRU infrastructure removed** (Round 13): −219 LOC of
  unreachable Q8 cache code and its support chain. Eliminated
  split-source vs `quant.h` drift (quant.h already shipped as
  no-op stubs).
- **Full score.sh** first run this session: 0.9979 / 1.0000 (99.8%) —
  new all-time high. Previously `--quick` hid quality/performance
  /integration dimensions (all actually at 100%).

### Regression

**13/13 test_models.sh PASS** (added Q5_K_M tier in Round 21).
Rounds 18 (2-row register pressure, −14%) and 19 (per-dispatch
madvise, −70%) attempted and rolled back — both would now be
auto-caught by the regression suite.

### Memories added

- `feedback_madvise_willneed_per_call.md`: per-dispatch madvise on
  Apple Silicon is a trap (VM contention on resident pages). Only
  use at load time.

### Session metrics

- 21 /grow rounds completed
- Net code change: −180 LOC (Round 13 cleanup vs Round 12/17/21 adds)
- Score: 0.9946 → **0.9979**
- 5-tier Qwen3.6 coverage on 16 GB Mac (IQ2/IQ3/Q3/IQ4/**Q5**)

---

## [v0.15.0] — 2026-04-19 (Mission A: MoE Batched Prefill complete)

### Headline

**Qwen3.6-35B-A3B MoE prefill on 16 GB M1 Pro: 4.4 → 6.1 t/s (+39%), wall -29%, CPU work -41%.**

The batched prefill path is now default-on. Opt out via `TQ_NO_MOE_BATCH=1`.

### Mission A Step 3 — measured on Qwen3.6-UD-Q3_K_S, 450-word prompt, warm, j=8:

| Step | Wall | Prefill | vs baseline |
|---|:-:|:-:|:-:|
| baseline (per-token) | 103 s | 4.4 t/s | — |
| + 3e driver | 92 s | 4.9 t/s | +11% |
| + 3f cross-expert parallel | 85 s | 5.4 t/s | +23% |
| + 3h batched shared expert | 82 s | 5.5 t/s | +25% |
| + 3g dynamic FCFS queue | **73 s** | **6.1 t/s** | **+39%** |

With 951-token prompt (more favorable amortization): baseline 11.4 → 13.4 t/s (+17% over prior steps alone).

### Added
- `tq_batched_matmul_q8_0` (b7c42dd) — Q8_0 batched kernel, Qwen3.6 non-expert attn path.
- `fused_dot_iq3_xxs_int8_batched` (8dd4920, fixed 61d7ce8 — missing `qs += 8` bug caught by sanity) — 35.6% of Qwen3.6 prefill compute.
- `fused_dot_iq3_s_int8_batched` (30428f3) — 19% compute.
- `fused_dot_iq4_xs_int8_batched` (30428f3) — TBL-16 codebook.
- `fused_dot_q3_k_int8_batched` (f9e5af1) — for pure Q3_K MoE models.
- `tq_moe_forward_batch` (9fb237d) — 3-phase dispatch: batch-route → inverse index → expert-wise batched gather/matmul/scatter.
- `tq_forward_batch_moe_hybrid` (627b65e, f255b46) — Qwen3.6-style driver: per-token DeltaNet + per-token self-attn + batched MoE FFN.
- Cross-expert parallel dispatch (e5f721a) — 8 workers, one expert each, private scatter buffer reduced serially.
- Batched shared expert (3a34cbf) — `tq_batched_matmul_q4` × 3 (gate/up/down) for Q4-converted shared experts.
- `tq_tp_run_dynamic` (f195a78) — FCFS atomic-counter thread-pool dispatch, flattens expert-workload stragglers. Opt-in via `TQ_MOE_BATCH_DYNAMIC=1`.
- `TQ_MOE_BATCH_SELFTEST=1` — N=1 sanity mode proves numerical equivalence (max_abs_diff 1.2e-7).

### Changed
- `TQ_MOE_BATCH=1` is now **default-on** (3f74f3e). Opt out with `TQ_NO_MOE_BATCH=1`.

### Fixed
- `fused_dot_iq3_xxs_int8_batched` missed `qs += 8` advance per sub-block (61d7ce8). Same precedent as the single-query kernel bug. Caught by sanity infrastructure before release.

### Verification
- `scripts/test_models.sh`: **12/12 PASS** throughout all 7 commits.
- Sanity max_abs_diff: N=1 path = 1.2e-7, N=7 path ≤ 2e-4 (well under 1e-3 spec).
- Decode unchanged (13+ t/s warm peak on Qwen3.6).

### Known limitations
- Dynamic FCFS queue (`TQ_MOE_BATCH_DYNAMIC`) is opt-in pending broader model coverage verification. Measured +17% when activated.
- Non-q4_converted shared experts fall back to per-token (not triggered on current Qwen3.6 UD quants).
- Decode path remains per-token (batched only affects prefill).

### Complementary work (this release cycle, 2026-04-18→19)
- v0.14.0: Q6_K NEON int8 (+115% on Q4_K_M models).
- v0.14.1: Q3 tier breakthrough (Q3_K/IQ3_XXS/IQ4_XS int8 kernels).
- v0.14.2: RoPE TLS sin/cos cache across all 4 branches; SwiGLU fast_exp_neon.
- v0.14.3: Q3_K_S tier on Qwen3.6 (RSS 5.24 GB on 16 GB Mac).
- **v0.15.0** (this): batched MoE prefill default-on.

Cumulative Qwen3.6-35B-A3B arc (session start → v0.15.0):
- Decode: 3.08 → **16.1 t/s** (IQ2_XXS peak); **2.8× faster than llama.cpp CPU**.
- Prefill: 5 → **6.1 t/s at j=8** (+22%); **13.4 t/s at longer prompt** (+17% over prior).
- RSS: 12 GB → **5.24 GB** (TQ_NO_MLOCK).

---

## [v0.14.3] — 2026-04-18 night (Q3_K_S tier on Qwen3.6)

### Highlights

Unsloth's `UD-Q3_K_S` (3.5 bpw, 14.3 GB) variant measured end-to-end after the Q3_K int8 kernel landed earlier in the day. Outcome: **smallest RSS, best quality, same speed class**. **Recommended Qwen3.6 variant on 16 GB Macs as of this release.**

Measurements on M1 Pro 16 GB, CPU 8t, `TQ_NO_METAL=1 TQ_NO_MLOCK=1`, warm 3-run peak:

| Variant | bpw | Disk | RSS | Decode | llama.cpp CPU | Speedup |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| UD-IQ2_XXS | 2.05 | 10.0 GB | 6.54 GB | 16.1 t/s | 5.07 | 3.2× |
| UD-IQ3_XXS | 3.06 | 12.3 GB | 6.82 GB | 14.6 t/s | 5.23 | 2.8× |
| **UD-Q3_K_S** | **3.5** | **14.3 GB** | **5.24 GB** | **14.3 t/s** | 5.11 | **2.8×** |

Quality step: "William Shakespeare wrote Hamlet" answers correctly on UD-Q3_K_S where UD-IQ3_XXS drifts. Decode prose reaches "Jack loved to play with his guitar" vs IQ2's "Jack lived in the small village of the mountains".

**Why RSS is smaller despite higher bpw**: Under `TQ_NO_MLOCK=1` the OS page cache holds only hot expert pages. Q3_K_S uses uniform 256-element Q3_K blocks; UD-IQ3_XXS mixes IQ3_XXS + IQ3_S + IQ4_XS + Q4_K + Q6_K block sizes. Uniform layout → fewer distinct pages touched per matmul → smaller page-cache working set.

### Added
- **RoPE TLS sin/cos cache extended to all remaining branches**: Phi-3 LongRoPE (commit `e00ff21`, key includes factors* pointer) and Gemma 4 NeoX (commit `5a8c093`, key includes rope_base for sliding/global distinction). The earlier Qwen 3.x partial-rotary cache (`b4d7807`) and Llama / Qwen 2.5 `tq_rope()` cache (`27c6707`) remain. Only remaining uncached variant: learned `rope_freqs[]` (Gemma 4 global-attention freq_factors) — deferred.
- **`bench/results/2026-04-18_q3_k_s_tier.md`** — full Q3_K_S vs IQ3_XXS vs IQ2_XXS methodology + reproduce.

### Regression
`scripts/test_models.sh`: **12/12 PASS** — RoPE cache extensions verified; Q3_K kernel verified end-to-end on Q3_K_S as well.

### Recommended
```bash
# Qwen3.6-35B-A3B on 16 GB Mac (best quality + lowest RSS)
TQ_NO_METAL=1 TQ_NO_MLOCK=1 ./build/quant \
  models/Qwen3.6-35B-A3B-UD-Q3_K_S.gguf \
  --chat -p "..." -n 80 -T 0.7 -j 8
```

---

## [v0.14.2] — 2026-04-18 evening (RoPE + SwiGLU NEON cleanups)

### Highlights

Two structural perf fixes discovered in the post-Q3 profile. Neither is a headline speedup on its own (Qwen3.6 decode is weight-read-bound), but both lower the instruction-level ceiling for any future fusion / pipelining work.

### Added
- **RoPE TLS sin/cos cache** (`src/engine/tq_transformer.c`, partial-rotary branch). Keyed on `(pos, rope_base, rope_dim)` — those are identical across all heads and all layers in one forward pass on models with `partial_rotary_factor` (every Qwen 3.x model). First layer pays `powf + cosf + sinf` per pair; remaining ~179 head-layer combinations do array reads. ~180× reduction in libc transcendental calls per token on Qwen3.6.
- **`fast_exp_neon`** — lifts the Schraudolph bit-twiddle exp into a single NEON FMA + `vcvtq_s32_f32` + reinterpret, replacing the per-lane scalar round-trip in `swiglu_fused`. Halves the instruction count in the 8-element SwiGLU tile.

### Regression
`scripts/test_models.sh`: 12/12 PASS.

### Commits
`b4d7807` (RoPE cache), `d4c0fc6` (SwiGLU NEON).

---

## [v0.14.1] — 2026-04-18 (Q3 breakthrough)

### Highlights

**Q3 weight-class unlocked on 16 GB Mac.** Three more scalar `fused_dot_*` kernels replaced with `vdotq_s32` int8 fast paths. Primary target: raise Qwen3.6-35B-A3B quantization from IQ2_XXS (2.05 bpw) to UD-IQ3_XXS (3.06 bpw) for a measurable quality step-up, without losing the speed lead over llama.cpp.

Measured on Qwen3.6-35B-A3B-UD-IQ3_XXS (M1 Pro 16 GB, CPU 8 threads, `TQ_NO_MLOCK=1`, warm peak):

| iteration | t/s | vs llama.cpp CPU |
|---|:-:|:-:|
| scalar baseline (new kernels disabled) | 7.9 | 1.5× |
| + Q3_K int8 | 12.2 | 2.3× |
| + IQ3_XXS int8 (post qs-advance fix) | 12.8 | 2.4× |
| + IQ4_XS int8 (TBL lookup) | **14.6** | **2.8×** |
| llama.cpp CPU 8t reference | 5.23 | — |

RSS: **6.82 GB** on 16 GB Mac (vs 6.54 GB for IQ2_XXS — only +0.28 GB for the quality step-up). Coherent decode persists **~2× longer** before drift compared with IQ2_XXS.

### Added
- **Q3_K × int8 NEON fast path** (`fused_dot_q3_k_int8`). Scalar fused_dot_q3_k was latent since initial Q3_K support. 16 × `vdotq_s32` per 256-element block. `vbicq_u8` resolves the `(hmask_bit ? 0 : 4)` branch without conditional. Env `TQ_Q3K_NOINT=1` reverts. Covers Q3_K_S / Q3_K_M / Q3_K_L / Q3_K_XL.
- **IQ3_XXS × int8 NEON fast path** (`fused_dot_iq3_xxs_int8`). Previous kernel was partial NEON (float FMA end). Reuses `iq3s_build8` helper from IQ3_S int8 path. Env `TQ_IQ3XXS_NOINT=1` reverts.
- **IQ4_XS × int8 NEON fast path** (`fused_dot_iq4_xs_int8`). `kvalues_iq4nl[16]` codebook fits in one ARM NEON TBL register — single `vqtbl1q_s8` does 16 parallel byte lookups per sub-block, cleanest possible NEON kernel shape. Env `TQ_IQ4XS_NOINT=1` reverts.
- **`scripts/qwen36_quality_probe.sh`** — factual Q&A (10 prompts, greedy T=0) + 100-token coherence probe + 3 multi-turn probes. Used for Q3 vs IQ2 A/B quality comparison.
- **`bench/results/2026-04-18_q3_breakthrough.md`** — full methodology, bug-caught-during-A/B writeup, reproduce commands.

### Fixed
- **`fused_dot_iq3_xxs_int8` missing `qs += 8;` between sub-blocks** (caught during A/B before commit). Without the advance, all 8 sub-blocks read the first sub-block's grid indices → 0/10 factual and digit-soup decode. A/B toggle (`TQ_IQ3XXS_NOINT=1` vs new kernel) isolated the bug in minutes. Precedent documented in commit `11e3c32`.

### Regression
`scripts/test_models.sh`: **12/12 PASS** across the full model suite (no Q3-family model is in the regression suite, so these kernels were validated via Qwen3.6 greedy-decode + A/B against scalar).

---

## [v0.14.0] — 2026-04-18

### Highlights

**MoE & Q4_K_M throughput breakthrough** — Qwen3.6-35B-A3B (MoE) now runs at **16.1 t/s on a 16 GB M1 Pro** (CPU-only), **3.2× faster than llama.cpp's CPU path** (5.07 t/s) at **35% lower RSS** (6.5 GB vs ~10 GB). Every Q4_K_M model in the suite also picked up **+115% to +180%** decode throughput from a single kernel fix.

All three changes were driven by `sample`-based profiling done *after* model load (earlier samples were dominated by the single-threaded Q4 load conversion, which hid the real hot path).

### Added
- **Q6_K × int8 NEON fast path** (`fused_dot_q6_k_int8`, `src/engine/tq_gguf_quants.c`). The existing `fused_dot_q6_k` is pure scalar; Q4_K_M embeds Q6_K for `attention.wo` and `ffn_down`, so it silently dominated decode on every Q4_K_M model. New kernel pre-quantizes activation to int8 (Q8_0 layout) and issues one `vdotq_s32` per 16-element sub-block. Env `TQ_Q6K_NOINT=1` reverts for A/B.
- **IQ3_S × int8 NEON fast path** (`fused_dot_iq3_s_int8`). UD-IQ2_XXS quantizations (e.g., Qwen3.6) embed IQ3_S for some critical layers; same scalar-to-vdotq_s32 fix. Env `TQ_IQ3S_NOINT=1` reverts.
- **MoE router NEON vectorize** (`tq_moe_route` in `src/engine/tq_moe.c`). Previous scalar `for e in 256 experts: dot(hidden, row)` loop replaced with 4-accumulator FMA (16 floats/cycle peak). Scratch buffers moved to `static __thread` — eliminates per-call `malloc`/`calloc` (60 allocs/token on Qwen3.6).
- **`TQ_NO_MLOCK=1` environment variable** — for MoE models on memory-constrained hosts, skips `mlock()` and uses `MADV_WILLNEED` instead. On a 16GB M1 Pro this is *both* faster (OS page cache LRU tracks the small hot-expert set better than mlock pinning the whole 10 GB) and saves ~5 GB RSS.
- **pthread QoS hint** (`QOS_CLASS_USER_INTERACTIVE`) applied to thread-pool workers on macOS to prefer P-cores over E-cores on asymmetric Apple Silicon (M-series Pro / Max / Ultra).
- **Dual-accumulator pair** in `matmul_q4_rows` inner loop (kept for kernel readability even though it did not move the needle on M1 — FMA throughput was not the bound).
- **`bench/results/2026-04-18_moe_and_q4_k_m_breakthrough.md`** — full methodology, per-iteration numbers, and reproduce commands.

### Performance
Measured on M1 Pro 16 GB, macOS 24, CPU-only (`TQ_NO_METAL=1`), 8 threads, warm 3-run peak, greedy decode.

| Model | before | after | vs llama.cpp CPU 8t |
|---|:-:|:-:|:-:|
| Qwen3.6-35B-A3B-UD-IQ2_XXS | 3.08 → 7.8 | **16.1** | 5.07 — **3.2× faster** |
| Qwen3.5-4B Q4_K_M | 5.0 | **14.1** | 19.9 (71%) |
| Phi-3.5-mini Q4_K_M | 6.2 | **14.1** | 26.7 (53%) |

Qwen3.6 RSS: **12.0 GB → 6.5 GB** with `TQ_NO_MLOCK=1`.

### Fixed
- **`fused_dot_q6_k` scalar performance regression** (latent since initial Q6_K support). Sample profiling attributed its cost to "matmul" in the wall-clock profile, hiding it for multiple releases. Fixed by the int8 fast path above.
- **`tq_moe_route` hot-path heap churn** — `malloc(num_experts)` and `calloc(num_experts)` on every router call (per layer, per token). Now thread-local.

### Recommended usage
```bash
# 16 GB Mac, Qwen3.6-35B-A3B MoE (best speed AND lowest RSS)
TQ_NO_METAL=1 TQ_NO_MLOCK=1 ./build/quant \
  models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
  --chat -p "..." -n 60 -T 0.7 -j 8

# Q4_K_M dense models (Phi-3.5, Qwen3.5-4B, Llama family)
TQ_NO_METAL=1 ./build/quant <model-Q4_K_M.gguf> -p "..." -n 50 -T 0 -j 8
```

### Regression
`scripts/test_models.sh`: **12/12 PASS** across Llama 3.1/3.2, Qwen 2.5/3.5/3.6, Phi-3.5, Gemma-4.

---

## [v0.13.0] — 2026-04-12

### Highlights

**Phi-3 / Phi-3.5 architecture fully supported** — the highest-value model quant.cpp was missing. Phi-3.5-mini (3.8B params, vocab 32K) is now the recommended default, delivering the best speed/quality combo:

```bash
pip install quantcpp
quantcpp                  # downloads Phi-3.5-mini Q8_0 (~3.8 GB), starts chat
```

### Added
- **Phi-3 / Phi-3.5 architecture support** — fused QKV projection, fused gate+up FFN, LongRoPE with NeoX-style rotation. Validated end-to-end on Phi-3.5-mini-instruct-Q4_K_M and Q8_0.
- **Phi-3.5-mini as default model** — replaces SmolLM2-1.7B as the recommended model. Q8_0 variant is 2x faster than Q4_K_M on Apple Silicon NEON (3.0 vs 1.5 tok/s).
- **ChatML template marker filter** — 32-byte lookahead filter in `chat_accum_callback` catches BPE-split markers (`<|im_start|>`, `<|im_end|>`, `<end_of_turn>` etc.) across token boundaries. Prevents template tokens from leaking into chat output.
- **Unsupported architecture hard-fail** — loading a model with fused QKV that quant.cpp can't handle (e.g., before Phi-3 support) now fails fast with a clear error message instead of silently producing garbage tokens.
- **quant-server-unified** — new server binary built directly on `quant.h` (single-header amalgamation). Eliminates divergence between `quant.h` and `libturboquant` split sources. CLI `quantcpp serve` now prefers this binary.
- **SmolLM2-1.7B** and **Phi-3.5-mini** added to `_MODEL_REGISTRY` with CLI aliases (`smollm2`, `phi3.5`, `phi-3.5-mini` etc.).
- **`ChatContextOverflow` exception** — Python `Model.chat()` now raises a typed exception on context overflow instead of silently returning empty output.
- **`docs/supported_models.md`** — architecture compatibility matrix, vocab-size speed guide, model selection recommendations.
- **`tools/gguf_inspect.c`** — GGUF tensor/metadata inspector for architecture debugging.

### Fixed
- **16 chat-cache bugs eliminated** (PRs #52, #53) — two audit passes found hidden bugs in KV cache prefix matching, text accumulation, server session management, WASM state handling.
- **`tq_generate_continue` overflow** — sliding-window truncation silently desynced `cached_text` from KV positions → garbage on long histories. Now returns `-2` on overflow.
- **`chat_accum_callback` realloc failure** — silently dropped tokens AND skipped user callback. Now always passes tokens through; marks accumulator tainted.
- **Server error handling** — `gen_rc == -1` produced HTTP 200 with empty content; now returns HTTP 500 with error JSON. Streaming sends `finish_reason: "error"`.
- **Server session kv_type mismatch** — reusing a session ID with different `kv_type`/`value_quant_bits` corrupted KV blocks. Now detects and rebuilds.
- **WASM `wasm_load_model`** — didn't reset `g_generating` flag → stuck busy after interrupted run.
- **`rep_penalty` in fast-path** — silently ignored in `tq_generate_chat_text`'s fast path (slow path applied it). Now consistent.
- **BOS token for Phi-3/Llama** — `<s>` added to BOS lookup chain. Phi-3 produces garbage without BOS.
- **Python CLI overflow handling** — `cmd_run` caught `ChatContextOverflow`, drops oldest turn, retries.

### Changed
- Default model: `Llama-3.2-1B` → `SmolLM2-1.7B` → **`Phi-3.5-mini` Q8_0**.
- CLI examples and README quickstart updated to use Phi-3.5-mini.
- Metal GPU dispatch disabled for fused-tensor models (CPU is faster for sub-4B).

### Performance
- **Phi-3.5-mini Q8_0**: 3.0 tok/s on Apple M3 (2x faster than Q4_K_M).
- **Chat KV cache reuse**: turn N+1 prefill is O(new tokens), not O(history). ~50% latency reduction on multi-turn chat.

---

## [v0.3.0] — 2026-04-01

### Highlights

**Real-model validation**, **adaptive compression**, and **information-theoretic foundations**. Every theoretical claim is now backed by measured data from actual model inference.

### Added

#### Real-Model Validation (Phase A)
- **Perplexity pipeline** (`--ppl <file>`): Teacher-forced PPL measurement. Gemma 4B results: 1-bit K + Q4 V PPL = 36.00 vs FP16 PPL = 35.99 — **+0.03% degradation** (effectively lossless).
- **Formal unbiasedness** (`tests/test_unbiased.cpp`): 100K random vector pairs prove all quant.cpp types have < 0.2% relative bias. The "unbiased inner product" claim is empirically verified.
- **Activation profiling** (`--profile-kv`): Per-layer pre/post-RHT distribution statistics. RHT reduces kurtosis from 10-99 to 3.9-7.9 and eliminates skewness. Honest finding: post-RHT is not perfectly Gaussian.
- **Memory bandwidth benchmark** (`--bench-memory`): tok/s vs context length across KV types.

#### Adaptive Compression (Phase B)
- **Per-layer bit recommendation** (`--recommend`): Profiles activation kurtosis, recommends 1-bit or 3-bit per layer. Gemma 270M: average 2.0 bits (vs 3.0 uniform) → 33% memory savings potential.
- **Attention entropy analysis** (`--attn-entropy`): Per-head Shannon entropy identifies sharp vs diffuse attention patterns.
- **V highres window** (`-V N`): Recent N tokens stored as FP16 alongside Q4/Q2 V. Test showed Q4 V already near-lossless (PPL +0.03%), so hybrid adds no measurable benefit.
- **Online codebook calibration** (`--calibrate`): Lloyd-Max iteration on real activation data. **MSE improved 49.7%** over default N(0,1) codebook — proves model-specific calibration matters.

#### Engine (Phase C)
- **Fused Q4 domain attention**: Weighted sum computed directly from packed nibbles without dequantize buffer. NEON `vfmaq_f32` path. Reduces memory traffic.
- **Prefill benchmark** (`--bench-prefill`): Measures KV quantization overhead during prompt processing.
- **CoW benchmark** (`bench/cow_bench.sh`): Analytical memory savings for shared-prefix serving.
- **Auto compression profile** (`bench/auto_profile.sh`): Full pipeline: profile → recommend → calibrate → JSON output.

#### Theory (Phase D)
- **Rate-distortion bounds** (`tests/test_rate_distortion.cpp`): Computes info-theoretic minimum MSE at each bit-width. Q4 uniform: 2.41x gap. Lloyd-Max: < 0.15 bits wasted.
- **Cumulative error analysis** (`tests/test_cumulative_error.cpp`): 16-layer simulation shows errors grow sub-linearly. Cosine similarity after 16 layers: 0.998 (Q4), 0.951 (Q2).

### Measured Results

| Metric | Value | Source |
|--------|-------|--------|
| Gemma 4B PPL (uniform_4b) | 35.99 | `--ppl` |
| Gemma 4B PPL (1b K + Q4 V) | 36.00 (+0.03%) | `--ppl` |
| Gemma 4B PPL (1b K + Q2 V) | 42.23 (+17.3%) | `--ppl` |
| Unbiasedness (all types) | < 0.2% rel_bias | `test_unbiased` |
| Post-RHT kurtosis range | 3.9 – 7.9 | `--profile-kv` |
| Adaptive bit average | 2.0 bits (33% saving) | `--recommend` |
| Calibrated codebook MSE improvement | 49.7% | `--calibrate` |
| 16-layer cumulative cosine (Q4) | 0.998 | `test_cumulative_error` |
| Rate-distortion gap (Q4 uniform) | 2.41x | `test_rate_distortion` |

---

## [v0.2.0] — 2026-04-01

### Highlights

**V cache quantization** and **expert-grade validation** — total K+V compression reaches 4.9x (Q4) to 7.1x (Q2), with every claim backed by measured data.

### Added

#### V Cache Quantization
- **Q4 value quantization** (`-v q4`): 4-bit per-block scale + packed nibbles. V compression 3.8x.
- **Q2 value quantization** (`-v q2`): 2-bit Lloyd-Max codebook. V compression 7.6x.
- **FP16 value auto-enable**: Values stored as FP16 when KV quantization is active (was FP32).
- Combined 1-bit K + Q4 V: **27.62 KB/token, 4.9x total K+V** (was 136 KB FP16).
- Combined 1-bit K + Q2 V: **19.12 KB/token, 7.1x total K+V**.
- CLI flag `-v q4|q2|fp16` for value quantization control.
- Memory reporting (`-M`) shows K and V breakdown separately.

#### Validation Suite
- **NEON/scalar consistency** (`tests/test_neon_scalar.cpp`): 14 tests verify every NEON path against pure C reference — Q4 dequant, Q2 dequant, RHT butterfly, RoPE, matmul, RMSNorm, Hamming attention.
- **Attention distribution** (`tests/test_attention_distribution.cpp`): 8 tests measure cosine similarity (0.996/0.918/0.634), Spearman rank correlation, top-k overlap. Proves compression is non-trivial (random K = 0.089).
- **Codebook theory** (`tests/test_codebook_theory.cpp`): 5 tests verify Lloyd-Max centroids match N(0,1) literature values within 0.001, MSE within 1.18x of information-theoretic optimal.
- **Edge cases** (`tests/test_edge_cases.cpp`): 29 tests — n=1 (single token), dim=0, NaN input, Inf input, all-same values, all-zero, n=10000 large sequence.
- **Numerical stability**: 4 tests for overflow-safe norm computation and NaN/Inf input guards.

#### Benchmark Scripts
- `bench/ablation_test.sh`: Divergence analysis at 50-300 tokens across KV types.
- `bench/long_quality_test.sh`: Coherence at 200/500/1000 tokens.
- `bench/sampling_test.sh`: Temperature sampling (T=0.3, T=0.7) comparison.
- `bench/quant_time_bench.sh`: Quantization timing wrapper.
- `bench/bench_kv_overhead.cpp`: Microbenchmark — uniform 148 ns, 1b 659 ns, 3b 11066 ns per vector.
- `bench/attention_dist_test.sh`: Attention distribution analysis wrapper.
- `scripts/sanitize.sh`: ASan + UBSan build and full test run.

### Fixed

- **Q4 dequant NEON nibble interleaving bug**: Lo/hi nibbles were written contiguously instead of interleaved, causing MSE 0.525 (300x worse than correct). Fixed with `vzip_u8` interleave.
- **QJL sign bias**: `proj >= 0.0f` → `proj > 0.0f` across 11 occurrences (CPU, CUDA, Metal). Eliminates asymmetric bias at zero projection boundary.
- **Norm overflow**: QJL norm computation now uses max-abs rescaling to prevent float overflow on large vectors.
- **NaN/Inf input guard**: Quantization functions zero-fill output block on NaN/Inf input instead of producing undefined output.

### Changed

- **Thread safety**: Global Q8 workspace (`g_q8_buf`) and sampler probability index (`g_probindex`) protected by mutex against concurrent realloc races.
- **RHT NEON vectorized**: Walsh-Hadamard butterfly uses `float32x4_t` for stages with len >= 4.
- **Q4 dequant NEON restored**: Properly vectorized with `vzip_u8` after bug fix (was scalar fallback).
- Test suite count: 23 → 26. Edge case count: 16 → 29.

### Measured Results

| Metric | Value | Source |
|--------|-------|--------|
| Total K+V compression (1b K + Q4 V) | 4.9x | `quant -M` |
| Total K+V compression (1b K + Q2 V) | 7.1x | `quant -M` |
| 32K context savings (Q4 V) | 3.4 GB | calculated |
| Attention cosine (uniform_4b) | 0.996 | `test_attention_distribution` |
| Attention cosine (turbo_kv_3b) | 0.918 | `test_attention_distribution` |
| Attention cosine (turbo_kv_1b) | 0.634 (= 2/pi) | `test_attention_distribution` |
| Random K cosine | 0.089 | `test_attention_distribution` |
| Lloyd-Max MSE vs theory | < 1.18x | `test_codebook_theory` |
| RHT overhead | 147 ns/vec | `bench_kv_overhead` |
| 1-bit attention | 1.2 ns/key | `bench_kv_overhead` |
| ASan + UBSan | 26/26 clean | `scripts/sanitize.sh` |

---

## [v0.1.0] — 2026-03-31

### Highlights

**Initial release** — pure C inference engine with quant.cpp KV cache compression. 1-bit keys, 10.7x key compression, byte-identical greedy output at 100 tokens.

### Added

#### Core Engine
- Complete transformer inference engine in pure C11 (10,000+ lines).
- Multi-architecture support: Gemma 3 (sliding window, GeGLU, dual RoPE) + Qwen3.5 (DeltaNet hybrid).
- Multi-shard safetensors loading (Gemma 4B = 2 shards, 883 tensors).
- Dual tokenizer: GPT2 byte-level BPE + SentencePiece auto-detect.
- TQM binary format: pre-quantized mmap, instant loading.

#### KV Cache Quantization (11 types)
- **quant.cpp KV 1-bit**: Sign-only after RHT. XOR + popcount attention (NEON `vcntq_u8`).
- **quant.cpp KV 3-bit**: 2-bit Lloyd-Max codebook + 1-bit QJL residual.
- **quant.cpp KV 4-bit**: 3-bit codebook + 1-bit QJL.
- **Uniform 4-bit / 2-bit**: Standard min-max quantization.
- **PolarQuant**: Polar coordinate (theta + radius) quantization.
- **QJL**: Quantized Johnson-Lindenstrauss sign hash.
- **Mixed / quant.cpp base**: Combined polar + QJL.

#### Weight Quantization
- Q4 weight quantization (4-bit per-block).
- Q2 weight quantization (2-bit Lloyd-Max codebook, Q2xQ8 integer matmul).
- BF16 weight support.

#### Performance
- NEON vectorized: 2-row matmul batching, fused dot products, Hamming distance.
- Thread pool with configurable thread count.
- Apple Silicon optimized.

#### Quality Verification
- 30/30 byte-identical greedy matches (K-only, 100 tokens, 10 diverse prompts).
- 23 test suites (Google Test).
- Qwen3.5: 0.999 cosine similarity vs PyTorch reference.
- Gemma 270M: per-layer exact match.

### Models Verified

| Model | Params | Speed (Q4, 6T) |
|-------|--------|----------------|
| Gemma 3 4B | 4B | 20.2 tok/s |
| Qwen3.5-0.8B | 752M | 80.1 tok/s |
| Gemma 3 270M | 270M | 176 tok/s |

---

## Release Process

### Version Scheme

```
v{MAJOR}.{MINOR}.{PATCH}

MAJOR: Breaking API changes
MINOR: New features, backward-compatible
PATCH: Bug fixes, performance improvements
```

### Checklist for New Releases

1. Update version in `CMakeLists.txt` (`project(turboquant VERSION x.y.z)`)
2. Add release section to this file (newest first)
3. Update badge version in `README.md` and `README.ko.md`
4. Run full validation:
   ```bash
   cmake --build build -j$(nproc) && ctest --test-dir build
   bash scripts/sanitize.sh
   ./build/quant gemma3-4b.tqm -p "The capital of France is" -j 6 -n 20 -T 0.0 -k turbo_kv_1b -v q4
   ```
5. Tag: `git tag -a v0.x.0 -m "Release v0.x.0"`
6. Push: `git push origin v0.x.0`
7. Create GitHub release with this section's content

### What Goes in Release Notes

- **Added**: New features, new tests, new benchmarks
- **Fixed**: Bug fixes (with root cause and impact)
- **Changed**: Behavior changes, performance improvements
- **Measured Results**: Table of key metrics with source (test name or script)
- **Breaking**: API changes that require user code modification
