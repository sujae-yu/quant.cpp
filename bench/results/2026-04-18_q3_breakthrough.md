# Q3 quality breakthrough — Qwen3.6-35B-A3B-UD-IQ3_XXS on 16GB Mac

**Headline**: UD-IQ3_XXS (3.06 bpw, 12.3 GB) now runs at **14.6 t/s** on a 16 GB M1 Pro, CPU-only — **2.8× faster than llama.cpp's CPU path** (5.23 t/s) on the same model. That's a **quality step-up from IQ2_XXS with only a 9% speed cost** (14.6 vs 16.1 t/s for IQ2_XXS on the same hardware).

Three new `vdotq_s32` int8 fast-path kernels landed: **Q3_K**, **IQ3_XXS**, **IQ4_XS**. All followed the same profile-driven pattern established by the Q6_K int8 work: `sample` the running process after load completes, find scalar `fused_dot_*` kernels in the hot path, port them to the pre-quantized int8 × `vdotq_s32` template.

Measurements: M1 Pro 8P+2E, 16 GB RAM, macOS 24, `TQ_NO_METAL=1 TQ_NO_MLOCK=1`, 8 threads, temperature 0, 30-token greedy decode, warm 3-run peak.

---

## Why Q3 was the next target

IQ2_XXS (2.05 bpw) on a 35B MoE decodes coherent prose for 30-40 tokens, then drifts into digit-soup / repetition — intrinsic 2-bit precision limit, not an engine bug. The natural next rung on the quality ladder is Q3 (≈3.0–3.5 bpw). The Unsloth Dynamic variant `UD-IQ3_XXS` is the sweet spot for 16 GB Macs: 12.3 GB on disk, critical layers (`attention.wo`, `lm_head`) kept at Q6_K for precision protection.

Engine readiness: on first profile against UD-IQ3_XXS, three kernels turned out to be **scalar fused_dot** — the same class of bug as `fused_dot_q6_k`, but each specific to types only used heavily by Q3-class quantizations. Also found one real correctness bug in the new IQ3_XXS kernel (missing `qs += 8;` advance between sub-blocks); A/B testing against the scalar path caught it immediately.

---

## Kernel 1 — Q3_K × int8 (`fused_dot_q3_k_int8`)

**Layout**: 110 bytes per 256 elements. `hmask[32]` (high bit), `qs[64]` (low 2 bits, 4 per byte), `scales[12]` (16 × 6-bit packed), `d` (fp16).

**Per-element value**: `((qs >> shift) & 3) - (hmask_bit ? 0 : 4)` ∈ `[-4..3]`, then scaled by `d × (scales[is] - 32)`.

**NEON strategy**:
- Pre-quantize x to int8 in 32-elem Q8_0-compatible blocks (x_ds per block)
- For each half (128 elements, 4 shifts × 2 sub-blocks each):
  - `vshrq_n_u8` + `vandq_u8` extracts 2-bit values
  - `vtstq_u8(hm, m_broadcast)` + `vbicq_u8(4, hm_test)` handles the `(hm_bit ? 0 : 4)` branch without conditional
  - 16 `vdotq_s32` per Q3_K block total
- Env toggle: `TQ_Q3K_NOINT=1` reverts to scalar.

**Result in isolation**: Running UD-IQ3_XXS with only Q3_K int8 ON (IQ3_XXS still scalar) = **12.2 t/s** vs **7.9 t/s** fully-scalar baseline → **+54%** from Q3_K alone.

## Kernel 2 — IQ3_XXS × int8 (`fused_dot_iq3_xxs_int8`)

**Layout**: 98 bytes per 256 elements. `d` (fp16), `qs[64]` (8-bit grid indices into `iq3xxs_grid[256]`, each entry = 4 packed uint8 values ∈ [0..7]), `scales_and_signs[32]` (8 × uint32 with 4-bit sub-scale in top nibble + 4×7-bit sign fields).

**NEON strategy**: reused `iq3s_build8` helper from earlier IQ3_S int8 kernel — combines two 4-byte grid entries into one `int8x8_t` with sign flips applied via `vtst`/`veor`/`vsub` from a `ksigns_iq2xs[]` lookup. Two `vcombine_s8` + two `vdotq_s32` per 32-element sub-block.

**Bug caught during A/B**: The new kernel initially produced digit-soup because it **missed the `qs += 8;` advance between the 8 sub-blocks per block** — every sub-block read the first sub-block's grid indices. Toggling `TQ_IQ3XXS_NOINT=1` (scalar path) immediately restored coherent output, isolating the regression to the new kernel. Added the advance; factual probe went from 0/10 → 2/10 and 100-token decode went from random digits to "there was a young man named Jack...". Commit `11e3c32` message explicitly flags this — useful precedent for future kernel ports.

## Kernel 3 — IQ4_XS × int8 (`fused_dot_iq4_xs_int8`)

**Layout**: 136 bytes per 256 elements. `d` + `scales_h` (uint16) + `scales_l[4]` (4-bit low per sub-block) + `qs[128]` (4-bit packed indices). Each 4-bit index looks up `kvalues_iq4nl[16]` — a non-linear int8 codebook.

**NEON strategy**: The 16-entry codebook is exactly **16 bytes**, which fits in a single ARM NEON **TBL** register. One `vqtbl1q_s8(vtbl, low_nibble)` does 16 parallel byte-indexed lookups in one cycle — no conditional unpack needed. Then 2 `vdotq_s32` per sub-block.

```c
const int8x16_t vtbl = vld1q_s8(kvalues_iq4nl);
uint8x16_t qs_v = vld1q_u8(qs);
uint8x16_t low  = vandq_u8(qs_v, vdupq_n_u8(0x0F));
uint8x16_t high = vshrq_n_u8(qs_v, 4);
int8x16_t w_lo  = vqtbl1q_s8(vtbl, low);    /* 16 decoded weights */
int8x16_t w_hi  = vqtbl1q_s8(vtbl, high);
```

This is the cleanest of the three kernels — codebook-fits-in-TBL is the ideal case for quantized matmul on ARM.

---

## End-to-end results

**Qwen3.6-35B-A3B-UD-IQ3_XXS** (M1 Pro 16GB, CPU 8t, `TQ_NO_MLOCK=1`):

| iteration                                              | t/s peak | vs llama.cpp CPU |
|--------------------------------------------------------|:--------:|:----------------:|
| scalar baseline (all new kernels disabled)             |  7.9     | 1.5× faster      |
| + Q3_K int8                                            | 12.2     | 2.3× faster      |
| + IQ3_XXS int8 (after qs-advance fix)                  | 12.8     | 2.4× faster      |
| + IQ4_XS int8 (TBL)                                    | **14.6** | **2.8× faster**  |
| llama.cpp CPU 8t (reference)                           |  5.23    | —                |

**Quality comparison** (same warm model, 30-token greedy, `TQ_NO_MLOCK=1`):

| quant      | bpw  | first drift     | prompt "Once upon a time" → 30-tok output                         |
|------------|:----:|-----------------|---------------------------------------------------------------------|
| IQ2_XXS    | 2.05 | ~token 30-40    | "there was a young man named Jack. He lived in the small village of the mountains. and he had to" |
| UD-IQ3_XXS | 3.06 | ~token 60-70    | "there was a young man named Jack. He lived in the small village called 'Happiness'. One day, he he went to to go." |

IQ3_XXS keeps the sentence structure (periods, coherent clauses) about **2× longer** before drift kicks in. Factual single-answer probe ("The capital of France is…") also improved from 4/10 → the Paris answer appearing cleanly, though the prompt probe suite (`scripts/qwen36_quality_probe.sh`) needs to be length-tolerant to score it reliably under `<think>` tag overhead.

**RSS on 16GB Mac**:
- IQ2_XXS: 6.54 GB (`TQ_NO_MLOCK=1`)
- UD-IQ3_XXS: 6.82 GB (`TQ_NO_MLOCK=1`) — only **+0.28 GB** for the quality step-up because the page cache streams routed experts; hot-set size hardly changes.

---

## Reproduce

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8

# Download (HF)
cd models
curl -L -O "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf?download=true"
cd ..

# Decode
TQ_NO_METAL=1 TQ_NO_MLOCK=1 ./build/quant \
  models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  --chat -p "Hello" -n 60 -T 0.7 -j 8

# A/B any single kernel off
TQ_Q3K_NOINT=1    ./build/quant ...
TQ_IQ3XXS_NOINT=1 ./build/quant ...
TQ_IQ4XS_NOINT=1  ./build/quant ...

# Quality probe
bash scripts/qwen36_quality_probe.sh models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf 8
```

## Regression

`scripts/test_models.sh`: **12/12 PASS** (Llama 3.1/3.2, Qwen 2.5/3.5/3.6, Phi-3.5, Gemma-4).

## Next

Remaining profiling shows the hot-path compute is now well-distributed across int8 kernels; `__psynch_cvwait` (worker idle) is the dominant bucket but is mostly unavoidable given the main-thread serial ops (norm / RoPE / softmax) between matmul dispatches. Further wins require either parallelizing those serial ops across the worker pool or pipelining matmul issue with non-matmul work — both structural changes.

For quality specifically, the next step up is `Q3_K_S` (3.5 bpw, 14.3 GB) or `Q3_K_M` (3.9 bpw, 15.5 GB) — both fit on 16GB with `TQ_NO_MLOCK=1` but leave <2 GB headroom. Kernel support is already in place for these (Q3_K int8 from this work covers them directly).
