/*
 * tq_llama_kernels.c — Bit-exact ports of llama.cpp ggml-cpu kernels
 *
 * The goal is to eliminate ULP-level divergence between our engine and
 * llama.cpp on quantized matmul. After R52-R60 the divergence was
 * localized to expert gate/up/down matmul in MoE layers: our custom
 * NEON kernel (tq_batched_matmul_iq4_xs) uses per-32 int8 activation
 * quantization, while llama.cpp uses per-256 Q8_K. Same weights, same
 * activation values — but different activation quant representation →
 * different output per expert → compounded logit divergence → different
 * top-1 token → attractor.
 *
 * This file holds verbatim C ports of llama.cpp's kernels, keeping
 * exact rounding, summation order, and intermediate precision. We
 * deliberately do NOT use SIMD in this file — the whole point is
 * bit-exact reproducibility for verification. A NEON-optimized path
 * (that still produces the same output) can be added later once the
 * generic path is proven correct.
 *
 * Source references (refs/llama.cpp @ b8849):
 *   ggml-common.h     — block format definitions
 *   ggml-quants.c     — dequantize + quantize helpers
 *   ggml-cpu/quants.c — vec_dot kernels (generic fallback)
 *
 * Ported kernels (this file, incremental):
 *   [R61 P2.1] block_q8_K definition + quantize_row_q8_K
 *   [R61 P2.2] ggml_vec_dot_iq4_xs_q8_K_generic
 *   [future ] IQ3_S, Q6_K, Q8_0 × Q8_K variants
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef QK_K
#define QK_K 256
#endif

/* block_iq4_xs — same layout as our existing block in tq_gguf_quants.c */
typedef struct {
    uint16_t d;           /* fp16 super-block scale */
    uint16_t scales_h;    /* high 2 bits of 8 sub-block scales */
    uint8_t  scales_l[4]; /* low 4 bits of 8 sub-block scales */
    uint8_t  qs[128];     /* 4-bit packed values */
} tqlk_block_iq4_xs;

/* block_q8_K — llama.cpp's intermediate activation format for dot
 * products against super-block weight types (IQ4_XS, IQ3_S, Q{3,4,5,6}_K).
 * Super-block of 256 int8 values with a single FP32 scale and 16
 * int16 partial sums (one per 16-value group). */
typedef struct {
    float   d;              /* FP32 scale */
    int8_t  qs[QK_K];       /* int8 quants */
    int16_t bsums[QK_K/16]; /* sum of qs[i*16 .. (i+1)*16) */
} tqlk_block_q8_K;

/* kvalues_iq4nl — llama.cpp dequant codebook for IQ4_XS/IQ4_NL.
 * Verified identical at src/engine/tq_gguf_quants.c:985 (R60). */
static const int8_t tqlk_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

/* FP16 → FP32 helper (IEEE 754-2008 half precision, no subnormal flush).
 * Matches ggml's GGML_CPU_FP16_TO_FP32 for normal + subnormal values. */
static inline float tqlk_fp16_to_fp32(uint16_t h) {
    uint32_t sign     = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp16    = ((uint32_t)h >> 10) & 0x1f;
    uint32_t mant16   =  (uint32_t)h & 0x3ff;
    uint32_t f;
    if (exp16 == 0) {
        if (mant16 == 0) {
            f = sign;
        } else {
            /* subnormal: normalize */
            int e = -1;
            do { e++; mant16 <<= 1; } while ((mant16 & 0x400) == 0);
            mant16 &= 0x3ff;
            f = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant16 << 13);
        }
    } else if (exp16 == 0x1f) {
        f = sign | 0x7f800000 | (mant16 << 13); /* inf/nan */
    } else {
        f = sign | ((exp16 + (127 - 15)) << 23) | (mant16 << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

/* ============================================================
 * [R61 P2.1] quantize_row_q8_K — FP32 activation → Q8_K format
 * Ported from refs/llama.cpp/ggml/src/ggml-quants.c:2692.
 * ============================================================ */
void tqlk_quantize_row_q8_K(const float* x, tqlk_block_q8_K* y, int64_t k) {
    /* assert k % QK_K == 0 — caller responsibility */
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        float max  = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (amax == 0.0f) {
            y[i].d = 0.0f;
            memset(y[i].qs, 0, QK_K);
            memset(y[i].bsums, 0, (QK_K/16) * sizeof(int16_t));
            x += QK_K;
            continue;
        }
        const float iscale = -128.0f / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = (int)nearbyintf(iscale * x[j]);
            y[i].qs[j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += y[i].qs[j*16 + ii];
            y[i].bsums[j] = (int16_t)sum;
        }
        y[i].d = 1.0f / iscale;
        x += QK_K;
    }
}

/* ============================================================
 * [R61 P2.2] ggml_vec_dot_iq4_xs_q8_K — bit-exact port
 * Ported from refs/llama.cpp/ggml/src/ggml-cpu/quants.c:1226.
 *
 * Computes s = dot(weight, activation) where:
 *   weight (vx) is n values packed as blocks of block_iq4_xs (4.25 bpw,
 *                256 values per block).
 *   activation (vy) is n values packed as blocks of block_q8_K.
 *   n must be divisible by 256 (QK_K).
 *
 * Summation order MUST match llama.cpp exactly for bit-exact reproduction:
 * outermost super-block loop, inner sub-block pairs with d1/d2 scaling,
 * then two 16-element chunk inner dot products accumulated in int32 then
 * multiplied by d1 or d2 and added to FP32 running sumf.
 * ============================================================ */
void tqlk_vec_dot_iq4_xs_q8_K(int n, float* s,
                               const void* vx, const void* vy) {
    /* caller ensures n % QK_K == 0 */
    const tqlk_block_iq4_xs* x = (const tqlk_block_iq4_xs*)vx;
    const tqlk_block_q8_K  * y = (const tqlk_block_q8_K  *)vy;
    const int nb = n / QK_K;

    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = tqlk_fp16_to_fp32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t* qs = x[ibl].qs;
        const int8_t*  q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8 * (ls1 - 32);
            const float d2 = d4d8 * (ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0]  * tqlk_kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * tqlk_kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0]  * tqlk_kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * tqlk_kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}

/* ============================================================
 * tqlk_matmul_iq4_xs — drop-in replacement for existing MoE expert matmul.
 * Signature compatible with single-row IQ4_XS weight × FP32 activation.
 *
 * For N=1 (decode): quantize activation to Q8_K once, run dot for each
 * output row against the shared activation.
 * For N>1 (prefill): caller should run N separate matmuls for now (batch
 * reuse possible in future).
 * ============================================================ */
void tqlk_matmul_iq4_xs(float* out, const void* weight, const float* x,
                         int out_dim, int in_dim) {
    /* Quantize activation to Q8_K */
    const int nb = in_dim / QK_K;
    tqlk_block_q8_K* yq = (tqlk_block_q8_K*)malloc((size_t)nb * sizeof(tqlk_block_q8_K));
    if (!yq) { for (int i = 0; i < out_dim; i++) out[i] = 0.0f; return; }
    tqlk_quantize_row_q8_K(x, yq, (int64_t)in_dim);

    const size_t row_bytes = (size_t)nb * sizeof(tqlk_block_iq4_xs);
    const uint8_t* w = (const uint8_t*)weight;

    for (int r = 0; r < out_dim; r++) {
        float sum;
        tqlk_vec_dot_iq4_xs_q8_K(in_dim, &sum, w + (size_t)r * row_bytes, yq);
        out[r] = sum;
    }

    free(yq);
}
