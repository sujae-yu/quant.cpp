# TurboQuant Gap Analysis: Paper vs quant.cpp

> Comparison of Google TurboQuant (Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) against the current `tq_polar` / `tq_qjl` / `tq_turbo` implementations in quant.cpp.

## TL;DR

quant.cpp's existing `TQ_TURBO_*` types implement an *earlier* generation of the algorithm — specifically the PolarQuant paper ([arXiv:2502.02617](https://arxiv.org/abs/2502.02617)) plus QJL residual. **They do not implement the algorithm published as "TurboQuant" by Google in April 2026.**

The 2-stage architecture (large quantizer + 1-bit QJL residual) is correct, but Stage 1 is the wrong quantizer. We need a new type — `TQ_TURBOQUANT_*` — that implements Google's actual algorithm.

## Algorithm comparison

| Component | Google TurboQuant (ICLR 2026) | quant.cpp `TQ_TURBO_*` (current) |
|---|---|---|
| **Stage 1 transform** | Random orthogonal rotation Π via QR(N(0,1)) | Polar coordinate conversion `(x,y) → (r,θ)` |
| **Stage 1 quantizer** | Lloyd-Max scalar quantizer (Beta-distribution-aware centroids) | Min-max linear quantization |
| **Stage 1 bit budget** | (b−1) bits per coordinate (e.g. 2.5 bits at 3.5-bit total) | 2 bits θ + 2 bits r per pair = 2 bpc |
| **Stage 2 residual** | 1-bit QJL on residual + ‖r‖₂ scalar | QJL on residual (no explicit norm) |
| **Block size** | None — operates on full d-dim vector | TQ_BK = 128 |
| **Outlier handling** | Per-channel: 32 outlier channels at higher bit width | None |
| **Inner product estimator** | ⟨y, x̃_mse⟩ + ‖r‖₂·⟨y, Q_qjl⁻¹(Q_qjl(r))⟩ | ⟨y, x̃_polar + x̃_qjl⟩ (no norm) |
| **Centroid storage** | Precomputed (Lloyd-Max output cached) | Per-block min/max in FP16 |

## What's missing

### 1. Random rotation matrix Π

Google generates Π by QR-decomposing a Gaussian random matrix. For our embedded targets we need a fast deterministic alternative:

- **Hadamard transform** (Walsh-Hadamard) — already used by Arclabs001's research (cited in llama.cpp #20969 as actually outperforming random rotation in their tests)
- **Householder reflectors** (small constant memory)
- **Givens rotation network**
- **Pseudo-random Rademacher diagonal × WHT × Rademacher** — what our `tq_rht.c` already implements!

Our `tq_rht.c` (Random Hadamard Transform) is exactly what Google's "random rotation" needs. We just don't compose it with the right quantizer.

### 2. Lloyd-Max scalar quantizer

After random rotation, the rotated coordinates follow a (concentrated) Beta distribution. The optimal quantizer for this distribution is **not** uniform min-max — it's Lloyd-Max with precomputed centroids:

- 1-bit: `{±√(2/πd)}`
- 2-bit: `{±0.453/√d, ±1.51/√d}`
- 3-bit, 4-bit, 5-bit: derived numerically from Beta(d/2, d/2)

We already have `src/core/tq_codebook.c` for codebook quantization — we just need to populate it with the Lloyd-Max centroids the paper specifies (or computes).

### 3. Stored ‖r‖₂

The residual norm is a single FP16 scalar per vector. We don't store this currently. Trivial to add to the `block_tq_turbo` struct.

### 4. Inner product estimator

The current `tq_turbo_attention_ref` does straight dot product on the dequantized sum. Google's estimator combines the two stages explicitly with the residual norm. This affects accuracy at low bit budgets.

## What we already have (good news)

| Building block | Status |
|---|---|
| `tq_rht.c` Random Hadamard Transform | ✅ Implemented |
| `tq_qjl.c` Quantized Johnson-Lindenstrauss | ✅ Implemented (1-bit sketch) |
| `tq_codebook.c` Codebook quantizer infrastructure | ✅ Implemented |
| `tq_turbo.c` 2-stage composition framework | ✅ Implemented (wrong Stage 1) |
| Plugin architecture (`tq_traits.c` 3-function registration) | ✅ Implemented |
| Block-wise KV cache with per-step quantization | ✅ Implemented |
| Multi-architecture inference (Llama 3, Qwen, Gemma, etc.) | ✅ Implemented |

We have ~90% of the infrastructure. The missing 10% is:
1. Compose RHT + Lloyd-Max into a new `TQ_TURBOQUANT_*` type
2. Precompute Lloyd-Max centroids
3. Wire the proper inner product estimator

## Implementation plan

### Phase 1: Add `TQ_TURBOQUANT_3B` and `TQ_TURBOQUANT_4B` types (1–2 days)

```c
// New block layout
typedef struct {
    uint16_t residual_norm_fp16;   // ‖r‖₂ stored as FP16
    uint8_t  rotated_quant[BLOCK_BYTES];  // Lloyd-Max codes after rotation
    uint8_t  qjl_residual[QJL_BYTES];     // 1-bit QJL on residual
} block_tq_turboquant_3b;

void tq_turboquant_quantize_ref(const float* src, void* dst, int n) {
    // 1. Apply random Hadamard transform: x̃ = Π·x
    float rotated[MAX_DIM];
    tq_rht_apply(src, rotated, n, /*seed=*/FIXED_SEED);

    // 2. Lloyd-Max quantize each coordinate
    quantize_lloyd_max(rotated, block->rotated_quant, n, /*bits=*/2);

    // 3. Compute residual r = x̃ − dequant(rotated_quant)
    float recon[MAX_DIM], residual[MAX_DIM];
    dequant_lloyd_max(block->rotated_quant, recon, n, 2);
    for (int i = 0; i < n; i++) residual[i] = rotated[i] - recon[i];

    // 4. Store ‖r‖₂ in FP16
    float r_norm = vec_l2_norm(residual, n);
    block->residual_norm_fp16 = fp32_to_fp16(r_norm);

    // 5. Normalize residual and apply 1-bit QJL
    for (int i = 0; i < n; i++) residual[i] /= r_norm;
    tq_qjl_quantize_ref(residual, block->qjl_residual, n);
}
```

### Phase 2: Precompute Lloyd-Max centroids (4 hours)

Run Lloyd-Max iteration offline for d=64, 128, 256, 512, 1024 at b=1,2,3,4,5 bits. Store as static const float arrays in `tq_codebook.c`. Cite the closed-form approximation from the paper for spot-check.

### Phase 3: Inner product estimator (4 hours)

```c
void tq_turboquant_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim) {
    // Apply same rotation to query (or use pre-rotated query)
    float q_rot[MAX_DIM];
    tq_rht_apply(query, q_rot, head_dim, FIXED_SEED);

    for (int s = 0; s < seq_len; s++) {
        const block_tq_turboquant_3b* b = &kv[s];

        // Stage 1: ⟨q_rot, x̃_mse⟩
        float dot1 = 0;
        float k_recon[MAX_DIM];
        dequant_lloyd_max(b->rotated_quant, k_recon, head_dim, 2);
        for (int d = 0; d < head_dim; d++) dot1 += q_rot[d] * k_recon[d];

        // Stage 2: ‖r‖₂·⟨q_rot, Q_qjl⁻¹(qjl)⟩
        float r_norm = fp16_to_fp32(b->residual_norm_fp16);
        float qjl_dot = tq_qjl_dot_with_query(q_rot, b->qjl_residual, head_dim);

        scores[s] = dot1 + r_norm * qjl_dot;
    }
}
```

### Phase 4: Validation against the paper (2 days)

Reproduce the paper's headline numbers within ±1%:

| Model | Method | Paper LongBench-E | Our target |
|---|---|---|---|
| Llama-3.1-8B | TurboQuant 2.5-bit | 49.44 | 48.4–50.4 |
| Llama-3.1-8B | TurboQuant 3.5-bit | 50.06 | 49.0–51.0 |
| Llama-3.1-8B | Full cache | 50.06 | (baseline) |
| Ministral-7B | TurboQuant 2.5-bit | 49.62 | 48.6–50.6 |
| Ministral-7B | Full cache | 49.89 | (baseline) |

Plus Needle-in-Haystack at 0.997 vs 1.000 baseline.

### Phase 5: llama.cpp PR (3–5 days)

Once Phase 4 passes, port the kernel to ggml as a new `GGML_TYPE_TURBOQUANT_K` and submit to [Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969). Differentiator vs the existing forks: (a) reproduces paper numbers, (b) clean ggml type registration, (c) backed by an end-to-end working C reference.

## Naming hygiene going forward

| Current name | Keeps | Future name |
|---|---|---|
| `TQ_POLAR_3B` / `TQ_POLAR_4B` | ✅ kept | "PolarQuant-style: polar coordinate quantization (predates Google paper)" |
| `TQ_TURBO_3B` / `TQ_TURBO_4B` | ✅ kept | "Turbo-style: PolarQuant + QJL residual (our original composition)" |
| (new) `TQ_TURBOQUANT_3B` / `TQ_TURBOQUANT_4B` | new | "Google TurboQuant (ICLR 2026): RHT + Lloyd-Max + 1-bit QJL residual + ‖r‖ scalar" |

This way users can choose between the two compositions and compare directly.

## Expected outcome

After Phase 1–4, quant.cpp becomes the **first single-header C implementation** of the published TurboQuant algorithm with reproduced paper numbers. After Phase 5, we have a credible llama.cpp PR with the strongest narrative in Discussion #20969.
