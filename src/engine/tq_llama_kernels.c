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
 *   [R63 P1  ] ggml_vec_dot_iq3_s_q8_K_generic + iq3s_grid table
 *   [future ] Q6_K, Q8_0 × Q8_K variants
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

/* ============================================================
 * [R63 P1] IQ3_S × Q8_K bit-exact port
 *
 * Qwen3.6-35B-A3B UD-IQ4_XS uses IQ3_S for 67% of routed expert
 * tensors (80 of 120). Without bit-exact parity here, the gate/up/down
 * of majority of experts still diverges from llama.cpp, explaining
 * why TQ_USE_LLAMA_KERNELS=1 only moves cosine 0.47→0.48. Porting
 * this kernel should close the gap on expert compute.
 *
 * Source: refs/llama.cpp/ggml/src/ggml-cpu/quants.c
 *   :: ggml_vec_dot_iq3_s_q8_K_generic (verbatim port)
 * Tables: refs/llama.cpp/ggml/src/ggml-common.h
 *   :: iq3s_grid (512 × uint32_t — quarter-sign grid)
 *   :: kmask_iq2xs (8 × uint8_t — bit masks for sign application)
 * ============================================================ */

#define TQLK_IQ3S_N_SCALE (QK_K/64)

typedef struct {
    uint16_t d;                      /* fp16 super-block scale */
    uint8_t  qs[QK_K/4];             /* 64 bytes: low 8 bits of grid idx */
    uint8_t  qh[QK_K/32];            /* 8 bytes: high bit of grid idx */
    uint8_t  signs[QK_K/8];          /* 32 bytes: sign packs */
    uint8_t  scales[TQLK_IQ3S_N_SCALE]; /* 4 bytes: 4-bit scales per 32 */
} tqlk_block_iq3_s;

static const uint8_t tqlk_kmask_iq2xs[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

static const uint32_t tqlk_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

/* Verbatim port of ggml_vec_dot_iq3_s_q8_K_generic.
 * Same order, same accumulation, same sign-bit logic. No SIMD. */
void tqlk_vec_dot_iq3_s_q8_K(int n, float* s,
                              const void* vx, const void* vy) {
    const tqlk_block_iq3_s* x = (const tqlk_block_iq3_s*)vx;
    const tqlk_block_q8_K*  y = (const tqlk_block_q8_K*)vy;

    const int nb = n / QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = tqlk_fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t* qs    = x[i].qs;
        const uint8_t* qh    = x[i].qh;
        const uint8_t* signs = x[i].signs;
        const int8_t*  q8    = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(tqlk_iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(tqlk_iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & tqlk_kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & tqlk_kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * (int32_t)ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = (const uint8_t*)(tqlk_iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(tqlk_iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & tqlk_kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & tqlk_kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * (int32_t)ls2;
        }
        sumf += d * (float)bsum;
    }
    *s = sumf;
}

/* Drop-in IQ3_S matmul — quantize activation once, dot for each row. */
void tqlk_matmul_iq3_s(float* out, const void* weight, const float* x,
                        int out_dim, int in_dim) {
    const int nb = in_dim / QK_K;
    tqlk_block_q8_K* yq = (tqlk_block_q8_K*)malloc((size_t)nb * sizeof(tqlk_block_q8_K));
    if (!yq) { for (int i = 0; i < out_dim; i++) out[i] = 0.0f; return; }
    tqlk_quantize_row_q8_K(x, yq, (int64_t)in_dim);

    const size_t row_bytes = (size_t)nb * sizeof(tqlk_block_iq3_s);
    const uint8_t* w = (const uint8_t*)weight;
    for (int r = 0; r < out_dim; r++) {
        float sum;
        tqlk_vec_dot_iq3_s_q8_K(in_dim, &sum,
                                 w + (size_t)r * row_bytes, yq);
        out[r] = sum;
    }
    free(yq);
}

/* ============================================================
 * [R63 P2] Q6_K × Q8_K bit-exact port
 *
 * Q6_K is used for Qwen3.6-A3B UD-IQ4_XS lm_head (vocab projection,
 * hit once per token at the very end of forward pass) and 3 routed
 * expert tensors. lm_head divergence directly shows up in final
 * logits — if our Q6_K path is even slightly off, every token's
 * argmax can flip. Porting this is high-leverage per-tensor.
 *
 * Source: refs/llama.cpp/ggml/src/ggml-cpu/quants.c
 *   :: ggml_vec_dot_q6_K_q8_K_generic (verbatim port)
 * ============================================================ */

typedef struct {
    uint8_t   ql[QK_K/2];        /* 128 bytes: lower 4 bits */
    uint8_t   qh[QK_K/4];        /* 64 bytes: upper 2 bits */
    int8_t    scales[QK_K/16];   /* 16 bytes: int8 scales */
    uint16_t  d;                 /* fp16 super-block scale */
} tqlk_block_q6_K;

void tqlk_vec_dot_q6_K_q8_K(int n, float* s,
                             const void* vx, const void* vy) {
    const tqlk_block_q6_K* x = (const tqlk_block_q6_K*)vx;
    const tqlk_block_q8_K* y = (const tqlk_block_q8_K*)vy;

    const int nb = n / QK_K;
    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums[8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t* a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += scale * (int32_t)aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = (int16_t)(q8[l] * a[l]);
            for (int l = 0; l < 8; ++l) aux32[l] += scale * (int32_t)aux16[l];
            q8 += 8; a += 8;
        }
        const float d = tqlk_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * (float)aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void tqlk_matmul_q6_K(float* out, const void* weight, const float* x,
                       int out_dim, int in_dim) {
    const int nb = in_dim / QK_K;
    tqlk_block_q8_K* yq = (tqlk_block_q8_K*)malloc((size_t)nb * sizeof(tqlk_block_q8_K));
    if (!yq) { for (int i = 0; i < out_dim; i++) out[i] = 0.0f; return; }
    tqlk_quantize_row_q8_K(x, yq, (int64_t)in_dim);

    const size_t row_bytes = (size_t)nb * sizeof(tqlk_block_q6_K);
    const uint8_t* w = (const uint8_t*)weight;
    for (int r = 0; r < out_dim; r++) {
        float sum;
        tqlk_vec_dot_q6_K_q8_K(in_dim, &sum,
                                w + (size_t)r * row_bytes, yq);
        out[r] = sum;
    }
    free(yq);
}
