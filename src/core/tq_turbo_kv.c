/**
 * tq_turbo_kv.c -- TurboQuant KV cache quantization pipeline
 *
 * Implements the TurboQuant algorithm from arXiv 2504.19874:
 *   1. Random Hadamard Transform (RHT) to decorrelate channels
 *   2. Optimal scalar quantization (Lloyd-Max codebook) on rotated data
 *   3. QJL 1-bit sign hash on the residual for unbiased inner product estimation
 *
 * Two variants:
 *   - TQ_TYPE_TURBO_KV_3B: 2-bit codebook + 1-bit QJL = 3 effective bits
 *   - TQ_TYPE_TURBO_KV_4B: 3-bit codebook + 1-bit QJL = 4 effective bits
 *
 * Key design: QJL is used for INNER PRODUCT estimation (attention), not for
 * point-wise reconstruction. The dequantize path uses MSE-only (codebook),
 * while the attention path adds the QJL residual correction for better scores.
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Forward declarations from other modules */
extern void tq_codebook_quantize(const float* src, uint8_t* dst_indices,
                                  int n, int bits, float inv_std);
extern void tq_codebook_dequantize(const uint8_t* indices, float* dst,
                                    int n, int bits, float inv_std);

/* ============================================================
 * FP16 helpers (local copies to avoid cross-module dependencies)
 * ============================================================ */

static uint16_t tkv_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float tkv_fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t u; } bits;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) { bits.u = sign; return bits.f; }
    if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); return bits.f; }
    exp = exp - 15 + 127;
    bits.u = sign | (exp << 23) | (mant << 13);
    return bits.f;
}

/* ============================================================
 * QJL random entry helper (must match tq_qjl.c exactly)
 * ============================================================ */

static float tkv_qjl_random_entry(int dim_idx, int sketch_idx) {
    uint32_t h = (uint32_t)(dim_idx * 2654435761u + sketch_idx * 340573321u);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return (h & 1) ? 1.0f : -1.0f;
}

/* ============================================================
 * Block seed: deterministic per-block seed based on position
 * ============================================================ */

#define TKV_DEFAULT_SEED 0x12345678u

/* ============================================================
 * Bit packing helpers for codebook indices
 * ============================================================ */

/* Pack 2-bit indices: 4 values per byte, LSB-first */
static void pack_2bit(const uint8_t* indices, uint8_t* packed, int n) {
    memset(packed, 0, (size_t)((n + 3) / 4));
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 4;
        int bit_pos  = (i % 4) * 2;
        packed[byte_idx] |= (uint8_t)((indices[i] & 0x03) << bit_pos);
    }
}

static void unpack_2bit(const uint8_t* packed, uint8_t* indices, int n) {
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 4;
        int bit_pos  = (i % 4) * 2;
        indices[i] = (packed[byte_idx] >> bit_pos) & 0x03;
    }
}

/* Pack 3-bit indices: using LSB-first bit-stream packing */
static void pack_3bit(const uint8_t* indices, uint8_t* packed, int n) {
    int total_bytes = (n * 3 + 7) / 8;
    memset(packed, 0, (size_t)total_bytes);
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos  = bit_offset % 8;
        uint16_t val = (uint16_t)(indices[i] & 0x07);
        packed[byte_idx] |= (uint8_t)(val << bit_pos);
        if (bit_pos > 5) {
            packed[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
        }
    }
}

static void unpack_3bit(const uint8_t* packed, uint8_t* indices, int n) {
    for (int i = 0; i < n; i++) {
        int bit_offset = i * 3;
        int byte_idx = bit_offset / 8;
        int bit_pos  = bit_offset % 8;
        uint16_t val = (uint16_t)packed[byte_idx];
        if (bit_pos > 5 && byte_idx + 1 < (n * 3 + 7) / 8) {
            val |= (uint16_t)packed[byte_idx + 1] << 8;
        }
        indices[i] = (uint8_t)((val >> bit_pos) & 0x07);
    }
}

/* ============================================================
 * QJL sign hash on residual (simplified, inline)
 * ============================================================ */

static void compute_qjl_signs(const float* residual, uint8_t* signs,
                                int dim, int n_sketch) {
    int hash_bytes = n_sketch / 8;
    memset(signs, 0, (size_t)hash_bytes);
    for (int s = 0; s < n_sketch; s++) {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++) {
            proj += residual[d] * tkv_qjl_random_entry(d, s);
        }
        if (proj >= 0.0f) {
            signs[s / 8] |= (uint8_t)(1 << (s % 8));
        }
    }
}

/* ============================================================
 * Internal: MSE-only dequantize in rotated space (shared helper)
 * Returns the reconstructed vector in rotated space (before inverse RHT).
 * ============================================================ */

static void dequant_mse_rotated_2bit(const block_tq_turbo_kv_3b* block,
                                      float* rotated, int dim) {
    float inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    unpack_2bit(block->mse_indices, indices, dim);
    tq_codebook_dequantize(indices, rotated, dim, 2, inv_std);
}

static void dequant_mse_rotated_3bit(const block_tq_turbo_kv_4b* block,
                                      float* rotated, int dim) {
    float inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    unpack_3bit(block->mse_indices, indices, dim);
    tq_codebook_dequantize(indices, rotated, dim, 3, inv_std);
}

/* ============================================================
 * TurboQuant KV 3-bit: quantize
 * Pipeline: normalize -> RHT -> 2-bit codebook -> residual -> QJL 1-bit
 * ============================================================ */

void tq_turbo_kv_3b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_turbo_kv_3b* block = (block_tq_turbo_kv_3b*)dst;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    /* Step 1: Compute L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_sq += src[i] * src[i];
    }
    float norm = sqrtf(norm_sq);
    block->norm = tkv_fp32_to_fp16(norm);

    /* Step 2: Normalize and copy to working buffer */
    float rotated[TQ_BK];
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) {
        rotated[i] = src[i] * inv_norm;
    }
    for (int i = dim; i < TQ_BK; i++) {
        rotated[i] = 0.0f;
    }

    /* Step 3: Apply RHT (in-place on rotated) */
    uint32_t seed = TKV_DEFAULT_SEED;
    block->rht_seed = seed;
    tq_rht_transform(rotated, dim, seed);

    /* Step 4: Scalar quantize with 2-bit codebook
     * After RHT, coordinates are approximately N(0, 1/sqrt(dim)).
     * inv_std = sqrt(dim) to normalize to N(0,1). */
    float inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    tq_codebook_quantize(rotated, indices, dim, 2, inv_std);

    /* Pack 2-bit indices */
    pack_2bit(indices, block->mse_indices, dim);

    /* Step 5: Dequantize MSE stage to compute residual */
    float reconstructed[TQ_BK];
    tq_codebook_dequantize(indices, reconstructed, dim, 2, inv_std);

    /* Step 6: Compute residual in rotated space */
    float residual[TQ_BK];
    for (int i = 0; i < dim; i++) {
        residual[i] = rotated[i] - reconstructed[i];
    }
    for (int i = dim; i < TQ_BK; i++) {
        residual[i] = 0.0f;
    }

    /* Step 7: Compute residual norm */
    float r_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        r_norm_sq += residual[i] * residual[i];
    }
    block->residual_norm = tkv_fp32_to_fp16(sqrtf(r_norm_sq));

    /* Step 8: QJL 1-bit sign hash on residual
     * sketch_dim = dim to fit in TQ_BK/8 bytes. */
    compute_qjl_signs(residual, block->qjl_signs, dim, dim);
}

/* ============================================================
 * TurboQuant KV 3-bit: dequantize (MSE-only)
 *
 * For point-wise reconstruction, we use only the codebook (MSE) stage.
 * The QJL residual is designed for inner product estimation (attention)
 * and adds noise in point-wise reconstruction.
 * ============================================================ */

void tq_turbo_kv_3b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo_kv_3b* block = (const block_tq_turbo_kv_3b*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);
    uint32_t seed = block->rht_seed;

    /* MSE-only dequantize in rotated space */
    float rotated[TQ_BK];
    dequant_mse_rotated_2bit(block, rotated, dim);

    /* Inverse RHT */
    tq_rht_inverse(rotated, dim, seed);

    /* Scale by original norm */
    for (int i = 0; i < dim; i++) {
        dst[i] = rotated[i] * norm;
    }
}

/* ============================================================
 * TurboQuant KV 3-bit: attention (two-stage inner product estimation)
 *
 * The paper's formula for inner product estimation:
 *   <q, k_approx> = norm * (<q_rot, k_mse_rot> + r_norm * correction)
 *
 * Where the QJL correction for <q_residual, r> is computed as:
 *   correction = sqrt(pi/2) / sketch_dim * sum_s(q_proj_s * key_sign_s)
 * with q_proj_s = sum_d(q_rot[d] * S[d,s]) being the query projection.
 *
 * This is more efficient: project query once, then inner product
 * with each key's sign vector is just Hamming distance arithmetic.
 * ============================================================ */

void tq_turbo_kv_3b_attention_ref(const float* query, const void* kv_cache,
                                    float* scores, int seq_len, int head_dim) {
    const block_tq_turbo_kv_3b* blocks = (const block_tq_turbo_kv_3b*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    int sketch_dim = dim;  /* sketch dimension = block dim */
    float qjl_scale = sqrtf(TQ_PI_2) / (float)sketch_dim;

    /* Project query into QJL sketch space (once, reuse across all keys) */
    float q_proj[TQ_BK];
    for (int s = 0; s < sketch_dim; s++) {
        float proj = 0.0f;
        for (int d = 0; d < dim; d++) {
            proj += query[d] * tkv_qjl_random_entry(d, s);
        }
        q_proj[s] = proj;
    }

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_3b* block = &blocks[seq];
        float norm = tkv_fp16_to_fp32(block->norm);
        float r_norm = tkv_fp16_to_fp32(block->residual_norm);
        uint32_t seed = block->rht_seed;

        /* MSE stage: dequantize -> inverse RHT -> dot with query */
        float rotated[TQ_BK];
        dequant_mse_rotated_2bit(block, rotated, dim);
        tq_rht_inverse(rotated, dim, seed);

        float mse_dot = 0.0f;
        for (int d = 0; d < dim; d++) {
            mse_dot += query[d] * rotated[d];
        }

        /* QJL residual correction:
         * The residual was computed in rotated space, so the QJL signs
         * encode sign(S . residual_rotated). The inner product correction is:
         *   <q, residual_original> = <q_rot, residual_rotated>  (RHT is orthogonal)
         * And QJL estimates <q_rot, residual_rotated> via:
         *   r_norm * sqrt(pi/2) / m * sum_s(q_rot_proj_s * sign_s)
         *
         * But our query is in original space, not rotated. We need to
         * rotate the query first to get q_rot, then project into sketch space.
         * However, we can also compute <q, r_original> directly since
         * the QJL random matrix S operates on the rotated residual.
         *
         * Since q_original and r_original are related by RHT:
         *   <q, r_original> = <q_rot, r_rotated>  (orthogonal transform preserves dot)
         *
         * So we need q_rot = RHT(query) to compute the QJL correction.
         * But RHT(query) changes per query -- not per key -- so we could
         * precompute it. For simplicity in this reference implementation,
         * we compute it inside the loop (can be hoisted later).
         */

        /* Rotate query to match the RHT space of this key */
        float q_rot[TQ_BK];
        memcpy(q_rot, query, (size_t)dim * sizeof(float));
        tq_rht_transform(q_rot, dim, seed);

        /* Project rotated query into QJL sketch space */
        float qjl_correction = 0.0f;
        for (int s = 0; s < sketch_dim; s++) {
            float proj = 0.0f;
            for (int d = 0; d < dim; d++) {
                proj += q_rot[d] * tkv_qjl_random_entry(d, s);
            }
            int bit = (block->qjl_signs[s / 8] >> (s % 8)) & 1;
            float key_sign = bit ? 1.0f : -1.0f;
            qjl_correction += proj * key_sign;
        }
        qjl_correction *= qjl_scale * r_norm;

        scores[seq] = norm * mse_dot + norm * qjl_correction;
    }
}

/* ============================================================
 * TurboQuant KV 4-bit: quantize
 * Same pipeline but with 3-bit codebook (8 levels) + 1-bit QJL
 * ============================================================ */

void tq_turbo_kv_4b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_turbo_kv_4b* block = (block_tq_turbo_kv_4b*)dst;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_sq += src[i] * src[i];
    }
    float norm = sqrtf(norm_sq);
    block->norm = tkv_fp32_to_fp16(norm);

    float rotated[TQ_BK];
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) {
        rotated[i] = src[i] * inv_norm;
    }
    for (int i = dim; i < TQ_BK; i++) {
        rotated[i] = 0.0f;
    }

    uint32_t seed = TKV_DEFAULT_SEED;
    block->rht_seed = seed;
    tq_rht_transform(rotated, dim, seed);

    float inv_std = sqrtf((float)dim);
    uint8_t indices[TQ_BK];
    tq_codebook_quantize(rotated, indices, dim, 3, inv_std);
    pack_3bit(indices, block->mse_indices, dim);

    float reconstructed[TQ_BK];
    tq_codebook_dequantize(indices, reconstructed, dim, 3, inv_std);

    float residual[TQ_BK];
    for (int i = 0; i < dim; i++) {
        residual[i] = rotated[i] - reconstructed[i];
    }
    for (int i = dim; i < TQ_BK; i++) {
        residual[i] = 0.0f;
    }

    float r_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        r_norm_sq += residual[i] * residual[i];
    }
    block->residual_norm = tkv_fp32_to_fp16(sqrtf(r_norm_sq));

    compute_qjl_signs(residual, block->qjl_signs, dim, dim);
}

/* ============================================================
 * TurboQuant KV 4-bit: dequantize (MSE-only)
 * ============================================================ */

void tq_turbo_kv_4b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_turbo_kv_4b* block = (const block_tq_turbo_kv_4b*)src;
    int dim = n;
    if (dim > TQ_BK) dim = TQ_BK;

    float norm = tkv_fp16_to_fp32(block->norm);
    uint32_t seed = block->rht_seed;

    float rotated[TQ_BK];
    dequant_mse_rotated_3bit(block, rotated, dim);

    tq_rht_inverse(rotated, dim, seed);

    for (int i = 0; i < dim; i++) {
        dst[i] = rotated[i] * norm;
    }
}

/* ============================================================
 * TurboQuant KV 4-bit: attention (two-stage inner product estimation)
 * ============================================================ */

void tq_turbo_kv_4b_attention_ref(const float* query, const void* kv_cache,
                                    float* scores, int seq_len, int head_dim) {
    const block_tq_turbo_kv_4b* blocks = (const block_tq_turbo_kv_4b*)kv_cache;
    int dim = head_dim;
    if (dim > TQ_BK) dim = TQ_BK;

    int sketch_dim = dim;
    float qjl_scale = sqrtf(TQ_PI_2) / (float)sketch_dim;

    for (int seq = 0; seq < seq_len; seq++) {
        const block_tq_turbo_kv_4b* block = &blocks[seq];
        float norm = tkv_fp16_to_fp32(block->norm);
        float r_norm = tkv_fp16_to_fp32(block->residual_norm);
        uint32_t seed = block->rht_seed;

        /* MSE stage */
        float rotated[TQ_BK];
        dequant_mse_rotated_3bit(block, rotated, dim);
        tq_rht_inverse(rotated, dim, seed);

        float mse_dot = 0.0f;
        for (int d = 0; d < dim; d++) {
            mse_dot += query[d] * rotated[d];
        }

        /* QJL residual correction */
        float q_rot[TQ_BK];
        memcpy(q_rot, query, (size_t)dim * sizeof(float));
        tq_rht_transform(q_rot, dim, seed);

        float qjl_correction = 0.0f;
        for (int s = 0; s < sketch_dim; s++) {
            float proj = 0.0f;
            for (int d = 0; d < dim; d++) {
                proj += q_rot[d] * tkv_qjl_random_entry(d, s);
            }
            int bit = (block->qjl_signs[s / 8] >> (s % 8)) & 1;
            float key_sign = bit ? 1.0f : -1.0f;
            qjl_correction += proj * key_sign;
        }
        qjl_correction *= qjl_scale * r_norm;

        scores[seq] = norm * mse_dot + norm * qjl_correction;
    }
}
