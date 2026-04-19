/**
 * tq_gguf_quants.c — GGUF weight dequantization and on-the-fly dequant matmul
 *
 * Implements dequantization for all major GGML quant types:
 *   Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32, BF16
 * Plus stub implementations for IQ types (IQ2_XXS, IQ3_XXS, IQ4_XS).
 *
 * The matmul path includes NEON-optimized inner loop for Apple Silicon.
 *
 * Pure C11, no external dependencies.
 */

#include "turboquant/tq_gguf.h"
#include "turboquant/tq_engine.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_HAS_NEON 1
#else
#define TQ_HAS_NEON 0
#endif

/* MSVC: no __builtin_* intrinsics */
#ifdef _MSC_VER
#define __builtin_prefetch(addr, ...) ((void)0)
#define __builtin_return_address(n) ((void*)0)
#endif

/* ============================================================
 * FP16 / BF16 helpers
 * ============================================================ */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* positive/negative zero */
            float r;
            uint32_t v = sign;
            memcpy(&r, &v, 4);
            return r;
        }
        /* subnormal: normalize by shifting mantissa up */
        exp = 1;
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
        exp += 112;  /* fp16 bias (15) -> fp32 bias (127): 127-15 = 112 */
    } else if (exp == 31) {
        /* inf / nan */
        exp = 255;
    } else {
        exp += 112;
    }

    uint32_t bits = sign | ((uint32_t)exp << 23) | (mant << 13);
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

/* ============================================================
 * Block structures (matching llama.cpp / ggml exactly)
 * ============================================================ */

/* Q8_0: 34 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    int8_t   qs[32];  /* quantized values */
} block_q8_0;

/* Q4_K: 144 bytes, 256 elements */
typedef struct {
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
    uint8_t  scales[12]; /* sub-block scales+mins, 6-bit packed */
    uint8_t  qs[128];    /* 4-bit values, 2 per byte */
} block_q4_K;

/* Q2_K: 84 bytes, 256 elements */
typedef struct {
    uint8_t  scales[16]; /* sub-block scales+mins, 4-bit each */
    uint8_t  qs[64];     /* 2-bit values, 4 per byte */
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
} block_q2_K;

/* Q3_K: 110 bytes, 256 elements */
typedef struct {
    uint8_t  hmask[32];  /* high bits */
    uint8_t  qs[64];     /* low 2 bits, 4 per byte */
    uint8_t  scales[12]; /* sub-block scales, packed */
    uint16_t d;          /* fp16 scale */
} block_q3_K;

/* Q6_K: 210 bytes, 256 elements */
typedef struct {
    uint8_t  ql[128];    /* low 4 bits */
    uint8_t  qh[64];     /* high 2 bits */
    int8_t   scales[16]; /* sub-block scales (signed int8) */
    uint16_t d;          /* fp16 super-block scale */
} block_q6_K;

/* Q5_K: 176 bytes, 256 elements */
typedef struct {
    uint16_t d;          /* fp16 super-block scale */
    uint16_t dmin;       /* fp16 super-block min */
    uint8_t  scales[12]; /* sub-block scales+mins, 6-bit packed */
    uint8_t  qh[32];     /* high bit for each of 256 elements */
    uint8_t  qs[128];    /* low 4 bits, 2 per byte */
} block_q5_K;

/* Q4_0: 18 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qs[16];  /* 4-bit values, 2 per byte */
} block_q4_0;

/* Q4_1: 20 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint16_t m;       /* fp16 min */
    uint8_t  qs[16];  /* 4-bit values, 2 per byte */
} block_q4_1;

/* Q5_0: 22 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qh[4];   /* high bits packed */
    uint8_t  qs[16];  /* low 4 bits, 2 per byte */
} block_q5_0;

/* Q5_1: 24 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale */
    uint16_t m;       /* fp16 min */
    uint8_t  qh[4];   /* high bits packed */
    uint8_t  qs[16];  /* low 4 bits, 2 per byte */
} block_q5_1;

/* Q8_1: 36 bytes, 32 elements */
typedef struct {
    uint16_t d;       /* fp16 scale (delta) */
    uint16_t s;       /* fp16 sum */
    int8_t   qs[32];
} block_q8_1;

/* Type size / block size / name — defined in tq_gguf.c, just declared in header */

/* ============================================================
 * Per-type dequantization
 * ============================================================ */

/* --- F32: passthrough --- */
static void dequant_f32(const void* src, float* dst, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

/* --- F16 --- */
static void dequant_f16(const void* src, float* dst, int n) {
    const uint16_t* s = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(s[i]);
    }
}

/* --- BF16 --- */
static void dequant_bf16(const void* src, float* dst, int n) {
    const uint16_t* s = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = bf16_to_fp32(s[i]);
    }
}

/* --- Q8_0: 34 bytes, 32 elements --- */
static void dequant_q8_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q8_0* blk = (const block_q8_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = d * blk[b].qs[j];
        }
    }
}

/* --- Q8_1: 36 bytes, 32 elements --- */
static void dequant_q8_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q8_1* blk = (const block_q8_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = d * blk[b].qs[j];
        }
    }
}

/* --- Q4_0: 18 bytes, 32 elements --- */
static void dequant_q4_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q4_0* blk = (const block_q4_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            dst[b * 32 + j]      = d * ((int)(byte & 0x0F) - 8);
            dst[b * 32 + j + 16] = d * ((int)(byte >> 4) - 8);
        }
    }
}

/* --- Q4_1: 20 bytes, 32 elements --- */
static void dequant_q4_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q4_1* blk = (const block_q4_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float m = fp16_to_fp32(blk[b].m);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            dst[b * 32 + j]      = d * (byte & 0x0F) + m;
            dst[b * 32 + j + 16] = d * (byte >> 4) + m;
        }
    }
}

/* --- Q5_0: 22 bytes, 32 elements --- */
static void dequant_q5_0(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q5_0* blk = (const block_q5_0*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        uint32_t qh;
        memcpy(&qh, blk[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            int lo = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            dst[b * 32 + j]      = d * (lo - 16);
            dst[b * 32 + j + 16] = d * (hi - 16);
        }
    }
}

/* --- Q5_1: 24 bytes, 32 elements --- */
static void dequant_q5_1(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_q5_1* blk = (const block_q5_1*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float m = fp16_to_fp32(blk[b].m);
        uint32_t qh;
        memcpy(&qh, blk[b].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            int lo = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)   | (((qh >> (j + 16)) & 1) << 4);
            dst[b * 32 + j]      = d * lo + m;
            dst[b * 32 + j + 16] = d * hi + m;
        }
    }
}

/* --- Q2_K: 84 bytes, 256 elements --- */
static void dequant_q2_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q2_K* blk = (const block_q2_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        const uint8_t* q = blk[b].qs;
        float* y = dst + b * 256;

        int is = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc_byte = blk[b].scales[is++];
                float dl = d * (sc_byte & 0x0F);
                float ml = dmin * (sc_byte >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc_byte = blk[b].scales[is++];
                dl = d * (sc_byte & 0x0F);
                ml = dmin * (sc_byte >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

/* --- Q3_K: 110 bytes, 256 elements ---
 * 3-bit = 2 low bits (qs) + 1 high bit (hmask)
 * 16 sub-blocks with 6-bit scales packed into 12 bytes */
static void dequant_q3_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q3_K* blk = (const block_q3_K*)src;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int b = 0; b < nb; b++) {
        const float d_all = fp16_to_fp32(blk[b].d);

        const uint8_t* q  = blk[b].qs;
        const uint8_t* hm = blk[b].hmask;
        uint8_t m = 1;

        /* Decode 16 x 6-bit scales using the ggml bit-manipulation trick.
         * The 12 packed bytes are loaded as three uint32, then rearranged
         * into four uint32 that are reinterpreted as sixteen int8 values. */
        memcpy(aux, blk[b].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float* y = dst + b * 256;

        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

/* --- Q4_K: 144 bytes, 256 elements ---
 * 8 sub-blocks of 32 elements each
 * 6-bit scale/min packed in 12 bytes */
static void dequant_q4_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q4_K* blk = (const block_q4_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* Decode 8 scale/min pairs from 12 bytes.
         * bytes 0..3:  low 6 bits of scales[0..3]
         * bytes 4..7:  low 6 bits of mins[0..3]
         * bytes 8..9:  high 2 bits of scales[0..3] + scales[4..7]
         * bytes 10..11: high 2 bits of mins[0..3] + mins[4..7]
         *
         * Actually ggml Q4_K packing:
         *   scales[0..5]: low 6 bits of scale for sub-blocks 0..5
         *                 but the first 4 bytes have scale low 6,
         *                 bytes 4..7 have min low 6,
         *                 bytes 8..11 have the high bits.
         *
         * Match ggml exactly:
         */
        uint8_t sc[8], mn[8];

        /* Low 6 bits */
        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;

        /* High 2 bits from bytes 8..11 */
        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        /* 4 groups of 64 elements (2 sub-blocks each).
         * Within each 64-element group, the first 32 elements use the low
         * nibble and the next 32 use the high nibble of the same 32 bytes.
         * This matches the ggml Q4_K packing exactly. */
        const uint8_t* q = blk[b].qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + l]      = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + 32 + l] = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

/* --- Q5_K: 176 bytes, 256 elements ---
 * Like Q4_K but with an extra high bit per element */
static void dequant_q5_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q5_K* blk = (const block_q5_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        /* Same scale/min packing as Q4_K */
        uint8_t sc[8], mn[8];

        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;

        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        /* 4 groups of 64 elements (2 sub-blocks each), matching ggml Q5_K.
         * Low 4 bits: low nibble for first 32 elems, high nibble for next 32.
         * High bit: from qh, using bitmasks u1/u2 that shift left by 2 each group. */
        const uint8_t* ql = blk[b].qs;
        const uint8_t* qh = blk[b].qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + l]      = d1 * ((ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                dst[b * 256 + j + 32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/* --- Q6_K: 210 bytes, 256 elements ---
 * 6-bit = 4 low bits (ql) + 2 high bits (qh)
 * 16 sub-blocks of 16 elements, int8 scales */
static void dequant_q6_k(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_q6_K* blk = (const block_q6_K*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);

        /* Match ggml dequantize_row_q6_K exactly.
         * Processes in two 128-element halves. Within each half, 32
         * iterations produce 4 output elements each by interleaving
         * ql low/high nibbles and qh 2-bit fields. */
        const uint8_t* ql = blk[b].ql;
        const uint8_t* qh = blk[b].qh;
        const int8_t*  sc = blk[b].scales;
        float* y = dst + b * 256;

        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ============================================================
 * IQ2_XXS dequantization — E8 lattice codebook
 *
 * Block: 66 bytes per 256 elements (2.0625 bpw)
 *   - d (fp16): super-block scale
 *   - qs[32] (uint16): 8 groups of 4 uint16 (8 bytes each)
 *     Each 8-byte group decodes 32 floats:
 *     - aux32[0]: 4 grid indices (1 byte each) → 4×8=32 values from iq2xxs_grid
 *     - aux32[1] bits 0-27: 4×7-bit sign fields → ksigns_iq2xs → 8-bit patterns
 *     - aux32[1] bits 28-31: 4-bit sub-block scale
 * ============================================================ */

static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static void dequant_iq2_xxs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 66; /* 66 bytes per block */
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint16_t* qs = (const uint16_t*)(blk + 2);

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs + 4 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;

            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];

                for (int j = 0; j < 8; j++) {
                    dst[b * 256 + ib32 * 32 + l * 8 + j] =
                        db * (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
        }
    }
}

/* ============================================================
 * IQ2_S dequantization — 82 bytes per 256 elements (2.5625 bpw)
 *
 * Block layout:
 *   d (fp16, 2 bytes): super-block scale
 *   qs[64]: first 32 bytes = grid index low bits, next 32 = sign bits
 *   qh[8]: high 2 bits of 10-bit grid index
 *   scales[8]: 4-bit sub-block scales (2 per byte)
 *
 * Uses iq2s_grid[1024] lookup table (10-bit index).
 * ============================================================ */

/* iq2s_grid: 1024-entry E8 lattice codebook for IQ2_S (from ggml-common.h).
 * Each uint64 packs 8 unsigned magnitude bytes from {0x08, 0x19, 0x2b}. */

static const uint64_t iq2s_grid[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919,
    0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908,
    0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19,
    0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919,
    0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908,
    0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b,
    0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b,
    0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b,
    0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b,
    0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808,
    0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b,
    0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08,
    0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808,
    0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819,
    0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908,
    0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b,
    0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808,
    0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919,
    0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808,
    0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b,
    0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b,
    0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919,
    0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919,
    0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819,
    0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908,
    0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908,
    0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819,
    0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819,
    0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08,
    0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819,
    0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808,
    0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908,
    0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819,
    0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19,
    0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919,
    0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808,
    0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908,
    0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08,
    0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908,
    0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808,
    0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908,
    0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819,
    0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808,
    0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808,
    0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b,
    0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808,
    0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808,
    0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19,
    0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08,
    0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b,
    0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808,
    0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908,
    0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08,
    0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919,
    0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b,
    0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b,
    0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08,
    0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808,
    0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919,
    0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808,
    0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819,
    0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808,
    0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919,
    0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b,
    0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808,
    0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b,
    0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b,
    0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b,
    0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08,
    0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b,
    0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919,
    0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19,
    0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919,
    0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919,
    0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808,
    0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919,
    0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b,
    0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919,
    0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08,
    0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808,
    0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908,
    0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808,
    0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908,
    0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808,
    0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908,
    0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808,
    0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808,
    0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b,
    0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08,
    0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919,
    0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908,
    0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808,
    0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908,
    0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808,
    0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808,
    0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908,
    0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08,
    0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b,
    0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819,
    0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819,
    0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808,
    0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908,
    0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908,
    0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819,
    0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b,
    0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819,
    0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b,
    0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819,
    0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819,
    0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

/* Public accessor for the IQ2_S codebook — used by Metal backend */
const uint64_t* tq_iq2s_grid(void) {
    return iq2s_grid;
}

static void dequant_iq2_s(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 82;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);

        const uint8_t* qs = blk + 2;           /* grid index low bytes */
        const uint8_t* signs = qs + 32;         /* sign bytes (second half of qs) */
        const uint8_t* qh = blk + 66;           /* high bits: blk + 2 + 64 */
        const uint8_t* scales = blk + 74;       /* scales: blk + 2 + 64 + 8 */

        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db0 = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                float dl = (l < 2) ? db0 : db1;
                /* 10-bit grid index: low 8 from qs, high 2 from qh */
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + grid_idx);
                uint8_t sign = signs[l];

                for (int j = 0; j < 8; j++) {
                    dst[b * 256 + ib32 * 32 + l * 8 + j] =
                        dl * (float)grid[j] * ((sign & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
            qs += 4;
            signs += 4;
        }
    }
}

/* ============================================================
 * IQ4_NL dequantization — 18 bytes per 32 elements (4.5 bpw)
 *
 * Non-linear 4-bit quantization using a 16-entry lookup table.
 * Block: d (fp16, 2 bytes) + qs[16] (4-bit packed pairs)
 * ============================================================ */

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

typedef struct {
    uint16_t d;       /* fp16 scale */
    uint8_t  qs[16];  /* 4-bit packed values, 2 per byte */
} block_iq4_nl;

static void dequant_iq4_nl(const void* src, float* dst, int n) {
    const int nb = n / 32;
    const block_iq4_nl* blk = (const block_iq4_nl*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        for (int j = 0; j < 16; ++j) {
            dst[b * 32 + j]      = d * kvalues_iq4nl[qs[j] & 0xf];
            dst[b * 32 + j + 16] = d * kvalues_iq4nl[qs[j] >> 4];
        }
    }
}

/* ============================================================
 * IQ4_XS dequantization — 136 bytes per 256 elements (4.25 bpw)
 *
 * Like IQ4_NL but with 256-element super-blocks and 6-bit sub-block scales.
 * Block: d (fp16, 2) + scales_h (uint16, 2) + scales_l[4] + qs[128]
 * ============================================================ */

typedef struct {
    uint16_t d;           /* fp16 super-block scale */
    uint16_t scales_h;    /* high 2 bits of 8 sub-block scales */
    uint8_t  scales_l[4]; /* low 4 bits of 8 sub-block scales, packed 2 per byte */
    uint8_t  qs[128];     /* 4-bit packed values */
} block_iq4_xs;

static void dequant_iq4_xs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_iq4_xs* blk = (const block_iq4_xs*)src;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        float* y = dst + b * 256;

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((blk[b].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf)
                         | (((blk[b].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

/* ============================================================
 * IQ3_XXS dequantization — 3.0625 bpw grid codebook
 *
 * Block: 98 bytes per 256 elements
 *   - d (fp16): super-block scale
 *   - qs[64]: grid indices (8 groups × 8 bytes each)
 *     First 64 bytes: 2 uint8 grid indices per sub-group (4 sub-groups × 2 = 8 per group)
 *     Next 32 bytes: scales_and_signs (4 bytes per group × 8 groups)
 *       Each uint32: bits 0-27 = 4×7-bit sign fields → ksigns_iq2xs
 *                    bits 28-31 = 4-bit sub-block scale
 *   Each grid index lookups iq3xxs_grid[idx] → 4 uint8 values (4 floats)
 *   2 grid indices per sub-group → 8 floats, 4 sub-groups per group → 32 floats
 * ============================================================ */

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

static void dequant_iq3_xxs(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)src;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98; /* 98 bytes per block */
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64; /* QK_K/4 = 64 */
        float* y = dst + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; j++) {
                    y[j + 0] = db * (float)grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db * (float)grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

/* --- IQ3_S dequant (ported from ggml dequantize_row_iq3_s) ---
 * 3.4375 bpw, block size 256, 110 bytes per block.
 * Uses iq3s_grid lookup table (512 entries × 4 bytes). */
static const uint32_t iq3s_grid[512] = {
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

/* IQ3_S block: 110 bytes per 256 elements.
 * d (fp16) | qs (64B) | qh (8B) | signs (32B) | scales (4B) */
typedef struct {
    uint16_t d;
    uint8_t  qs[64];    /* QK_K/4 = 256/4 */
    uint8_t  qh[8];     /* QK_K/32 = 256/32 */
    uint8_t  signs[32]; /* QK_K/8 = 256/8 */
    uint8_t  scales[4]; /* QK_K/64 = 256/64 */
} block_iq3_s_t;

static void dequant_iq3_s(const void* src, float* dst, int n) {
    const int nb = n / 256;
    const block_iq3_s_t* blk = (const block_iq3_s_t*)src;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blk[i].d);
        const uint8_t* qs = blk[i].qs;
        const uint8_t* qh = blk[i].qh;
        const uint8_t* signs = blk[i].signs;
        float* y = dst + i * 256;

        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (blk[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (blk[i].scales[ib32 / 2] >> 4));
            /* First sub-block of 32 */
            for (int l = 0; l < 4; l++) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    y[j + 0] = db1 * grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                    y[j + 4] = db1 * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            /* Second sub-block of 32 */
            for (int l = 0; l < 4; l++) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    y[j + 0] = db2 * grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                    y[j + 4] = db2 * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

/* ============================================================
 * IQ3_S × int8 fused dot (NEON vdotq_s32 path)
 *
 * The float fused_dot_iq3_s below is pure scalar — sample on
 * Qwen3.6-35B-A3B (UD-IQ2_XXS uses IQ3_S for some critical layers)
 * shows it in the hot path. Same fix pattern as IQ2_XXS / Q6_K:
 * pre-quantize x to int8, gather grid + signs into int8x16,
 * vdotq_s32 against x_q8.
 *
 * Per 256-element block: 8 sub-blocks × 32 elements = 16 vdotq_s32 calls.
 * ============================================================ */
#if TQ_HAS_NEON
/* Local copy — iq2_sign_bit_masks is defined later in this TU; we redeclare
 * a private constant here to avoid forward-reference issues. Same values
 * as the original (powers of two by bit position). */
static const uint8_t iq3s_neon_bit_masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static inline int8x8_t iq3s_build8(uint32_t g1, uint32_t g2, uint8_t signs_byte,
                                    const uint8x8_t vbit_masks)
{
    /* g1, g2: each 4-byte grid entry (uint8 values in [0..7])
     * signs_byte: 8 bits — bits 0..3 sign of g1, bits 4..7 sign of g2 */
    uint64_t combined = ((uint64_t)g2 << 32) | g1;
    uint8x8_t grid_u = vreinterpret_u8_u64(vdup_n_u64(combined));
    /* Build sign mask: 0xFF where bit set in signs_byte, 0x00 otherwise */
    uint8x8_t sb = vtst_u8(vdup_n_u8(signs_byte), vbit_masks);
    int8x8_t  sn = vreinterpret_s8_u8(sb);  /* -1 (all ones) or 0 */
    int8x8_t  gi = vreinterpret_s8_u8(grid_u);
    /* (gi xor sn) - sn  ==  sign-flip when sn is -1, identity when 0 */
    return vsub_s8(veor_s8(gi, sn), sn);
}

static float fused_dot_iq3_s_int8(const void* row, const int8_t* x_qs,
                                    const float* x_ds, int n)
{
    const int nb = n / 256;
    const block_iq3_s_t* blk = (const block_iq3_s_t*)row;
    const uint8x8_t vbit_masks = vld1_u8(iq3s_neon_bit_masks);

    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) {
            __builtin_prefetch((const uint8_t*)&blk[b+1], 0, 3);
            __builtin_prefetch((const uint8_t*)&blk[b+1] + 64, 0, 3);
        }
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const uint8_t* qh = blk[b].qh;
        const uint8_t* signs = blk[b].signs;
        const int8_t* xb = x_qs + b * 256;
        const float*  xd = x_ds + b * 8;

        float sub_sum = 0.0f;

        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (blk[b].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (blk[b].scales[ib32 / 2] >> 4));

            /* ---- First sub-block of 32 elements (uses qh[0]) ---- */
            uint32_t g[8];  /* 8 grid entries × 4 bytes = 32 weights */
            for (int l = 0; l < 4; l++) {
                int idx1 = qs[2*l + 0] | ((qh[0] << (8 - 2*l)) & 256);
                int idx2 = qs[2*l + 1] | ((qh[0] << (7 - 2*l)) & 256);
                g[2*l + 0] = iq3s_grid[idx1];
                g[2*l + 1] = iq3s_grid[idx2];
            }
            int8x8_t w0 = iq3s_build8(g[0], g[1], signs[0], vbit_masks);
            int8x8_t w1 = iq3s_build8(g[2], g[3], signs[1], vbit_masks);
            int8x8_t w2 = iq3s_build8(g[4], g[5], signs[2], vbit_masks);
            int8x8_t w3 = iq3s_build8(g[6], g[7], signs[3], vbit_masks);
            int8x16_t vw_lo = vcombine_s8(w0, w1);
            int8x16_t vw_hi = vcombine_s8(w2, w3);

            int8x16_t vx_lo = vld1q_s8(xb + 0);
            int8x16_t vx_hi = vld1q_s8(xb + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vacc = vdotq_s32(vdupq_n_s32(0), vw_lo, vx_lo);
            vacc = vdotq_s32(vacc, vw_hi, vx_hi);
            int32_t isum1 = vaddvq_s32(vacc);
#else
            int16x8_t pa = vmull_s8(vget_low_s8(vw_lo), vget_low_s8(vx_lo));
            int16x8_t pb = vmull_s8(vget_high_s8(vw_lo), vget_high_s8(vx_lo));
            int16x8_t pc = vmull_s8(vget_low_s8(vw_hi), vget_low_s8(vx_hi));
            int16x8_t pd = vmull_s8(vget_high_s8(vw_hi), vget_high_s8(vx_hi));
            int32x4_t vacc = vpaddlq_s16(vaddq_s16(vaddq_s16(pa, pb), vaddq_s16(pc, pd)));
            int32_t isum1 = vaddvq_s32(vacc);
#endif
            sub_sum += (float)isum1 * db1 * xd[ib32];

            qs += 8;
            signs += 4;
            xb += 32;

            /* ---- Second sub-block of 32 elements (uses qh[1]) ---- */
            for (int l = 0; l < 4; l++) {
                int idx1 = qs[2*l + 0] | ((qh[1] << (8 - 2*l)) & 256);
                int idx2 = qs[2*l + 1] | ((qh[1] << (7 - 2*l)) & 256);
                g[2*l + 0] = iq3s_grid[idx1];
                g[2*l + 1] = iq3s_grid[idx2];
            }
            int8x8_t w0b = iq3s_build8(g[0], g[1], signs[0], vbit_masks);
            int8x8_t w1b = iq3s_build8(g[2], g[3], signs[1], vbit_masks);
            int8x8_t w2b = iq3s_build8(g[4], g[5], signs[2], vbit_masks);
            int8x8_t w3b = iq3s_build8(g[6], g[7], signs[3], vbit_masks);
            int8x16_t vw_lo2 = vcombine_s8(w0b, w1b);
            int8x16_t vw_hi2 = vcombine_s8(w2b, w3b);

            int8x16_t vx_lo2 = vld1q_s8(xb + 0);
            int8x16_t vx_hi2 = vld1q_s8(xb + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vacc2 = vdotq_s32(vdupq_n_s32(0), vw_lo2, vx_lo2);
            vacc2 = vdotq_s32(vacc2, vw_hi2, vx_hi2);
            int32_t isum2 = vaddvq_s32(vacc2);
#else
            int16x8_t pa2 = vmull_s8(vget_low_s8(vw_lo2), vget_low_s8(vx_lo2));
            int16x8_t pb2 = vmull_s8(vget_high_s8(vw_lo2), vget_high_s8(vx_lo2));
            int16x8_t pc2 = vmull_s8(vget_low_s8(vw_hi2), vget_low_s8(vx_hi2));
            int16x8_t pd2 = vmull_s8(vget_high_s8(vw_hi2), vget_high_s8(vx_hi2));
            int32x4_t vacc2 = vpaddlq_s16(vaddq_s16(vaddq_s16(pa2, pb2), vaddq_s16(pc2, pd2)));
            int32_t isum2 = vaddvq_s32(vacc2);
#endif
            sub_sum += (float)isum2 * db2 * xd[ib32 + 1];

            qs += 8;
            signs += 4;
            qh += 2;
            xb += 32;
        }

        sumf += sub_sum;
    }

    return sumf;
}

/* ============================================================
 * IQ3_S × int8 BATCHED dot — Mission A Step 3b.
 *
 * Same pattern as IQ3_XXS batched (Step 3a): amortize the per-block
 * weight unpack (grid + signs) across N activations. IQ3_S is 19% of
 * Qwen3.6 prefill compute per profile 2026-04-18.
 *
 * Block layout: 110 bytes per 256 elems (see block_iq3_s_t struct).
 * Inner structure: 8 sub-blocks × 2 qh halves × 4 l-iters × 2 grids
 * — but cache hot weight-unpack vectors across the N loop.
 * ============================================================ */
#if TQ_HAS_NEON
static void fused_dot_iq3_s_int8_batched(
    float* out,              /* [N, n_rows] row-major */
    const void* weight,      /* block_iq3_s_t array, n_rows × n_super × 110 */
    size_t row_bytes,        /* = n_super × sizeof(block_iq3_s_t) */
    int out_row_stride_n,
    const int8_t* X_qs,      /* [N, n_super × 256] int8 */
    const float* X_ds,       /* [N, n_super × 8] */
    int start_row, int end_row,
    int n_super, int N)
{
    const uint8x8_t vbit_masks = vld1_u8(iq3s_neon_bit_masks);

    for (int d = start_row; d < end_row; d++) {
        const uint8_t* base = (const uint8_t*)weight + (size_t)d * row_bytes;

        float acc[64];
        if (N > 64) return;
        memset(acc, 0, (size_t)N * sizeof(float));

        for (int b = 0; b < n_super; b++) {
            const block_iq3_s_t* blk = (const block_iq3_s_t*)(base + (size_t)b * sizeof(block_iq3_s_t));
            if (b + 1 < n_super) {
                __builtin_prefetch((const uint8_t*)blk + 110, 0, 3);
                __builtin_prefetch((const uint8_t*)blk + 110 + 64, 0, 3);
            }
            const float d_super = fp16_to_fp32(blk->d);
            const uint8_t* qs = blk->qs;
            const uint8_t* qh = blk->qh;
            const uint8_t* signs = blk->signs;

            /* 8 sub-blocks (4 pairs using qh[0] + qh[1]).
             * Mirror single-query layout exactly: outer ib32 stride 2, inner 2 halves.
             * Restructure outer loops per-block, inner loop vectorizes N dim. */
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                const float db1 = d_super * (1 + 2 * (blk->scales[ib32 / 2] & 0xf));
                const float db2 = d_super * (1 + 2 * (blk->scales[ib32 / 2] >> 4));

                /* ---- first sub-block of 32 elements (uses qh[0]) ---- */
                uint32_t g[8];
                for (int l = 0; l < 4; l++) {
                    int idx1 = qs[2*l + 0] | ((qh[0] << (8 - 2*l)) & 256);
                    int idx2 = qs[2*l + 1] | ((qh[0] << (7 - 2*l)) & 256);
                    g[2*l + 0] = iq3s_grid[idx1];
                    g[2*l + 1] = iq3s_grid[idx2];
                }
                int8x8_t w0 = iq3s_build8(g[0], g[1], signs[0], vbit_masks);
                int8x8_t w1 = iq3s_build8(g[2], g[3], signs[1], vbit_masks);
                int8x8_t w2 = iq3s_build8(g[4], g[5], signs[2], vbit_masks);
                int8x8_t w3 = iq3s_build8(g[6], g[7], signs[3], vbit_masks);
                int8x16_t vw_lo1 = vcombine_s8(w0, w1);
                int8x16_t vw_hi1 = vcombine_s8(w2, w3);

                /* Inner N loop for first sub-block */
                for (int n = 0; n < N; n++) {
                    const int8_t* xb_n = X_qs + (size_t)n * (n_super * 256) + b * 256 + ib32 * 32;
                    int8x16_t vx_lo = vld1q_s8(xb_n + 0);
                    int8x16_t vx_hi = vld1q_s8(xb_n + 16);
#ifdef __ARM_FEATURE_DOTPROD
                    int32x4_t vacc = vdotq_s32(vdupq_n_s32(0), vw_lo1, vx_lo);
                    vacc = vdotq_s32(vacc, vw_hi1, vx_hi);
                    int32_t isum = vaddvq_s32(vacc);
#else
                    int16x8_t pa = vmull_s8(vget_low_s8(vw_lo1), vget_low_s8(vx_lo));
                    int16x8_t pb = vmull_s8(vget_high_s8(vw_lo1), vget_high_s8(vx_lo));
                    int16x8_t pc = vmull_s8(vget_low_s8(vw_hi1), vget_low_s8(vx_hi));
                    int16x8_t pd = vmull_s8(vget_high_s8(vw_hi1), vget_high_s8(vx_hi));
                    int32x4_t vacc = vpaddlq_s16(vaddq_s16(vaddq_s16(pa, pb), vaddq_s16(pc, pd)));
                    int32_t isum = vaddvq_s32(vacc);
#endif
                    float xd_n = X_ds[(size_t)n * (n_super * 8) + b * 8 + ib32];
                    acc[n] += (float)isum * db1 * xd_n;
                }

                /* advance qs & signs pointers for second sub-block */
                qs += 8;
                signs += 4;

                /* ---- second sub-block of 32 elements (uses qh[1]) ---- */
                for (int l = 0; l < 4; l++) {
                    int idx1 = qs[2*l + 0] | ((qh[1] << (8 - 2*l)) & 256);
                    int idx2 = qs[2*l + 1] | ((qh[1] << (7 - 2*l)) & 256);
                    g[2*l + 0] = iq3s_grid[idx1];
                    g[2*l + 1] = iq3s_grid[idx2];
                }
                int8x8_t w0b = iq3s_build8(g[0], g[1], signs[0], vbit_masks);
                int8x8_t w1b = iq3s_build8(g[2], g[3], signs[1], vbit_masks);
                int8x8_t w2b = iq3s_build8(g[4], g[5], signs[2], vbit_masks);
                int8x8_t w3b = iq3s_build8(g[6], g[7], signs[3], vbit_masks);
                int8x16_t vw_lo2 = vcombine_s8(w0b, w1b);
                int8x16_t vw_hi2 = vcombine_s8(w2b, w3b);

                for (int n = 0; n < N; n++) {
                    const int8_t* xb_n = X_qs + (size_t)n * (n_super * 256) + b * 256 + (ib32 + 1) * 32;
                    int8x16_t vx_lo = vld1q_s8(xb_n + 0);
                    int8x16_t vx_hi = vld1q_s8(xb_n + 16);
#ifdef __ARM_FEATURE_DOTPROD
                    int32x4_t vacc = vdotq_s32(vdupq_n_s32(0), vw_lo2, vx_lo);
                    vacc = vdotq_s32(vacc, vw_hi2, vx_hi);
                    int32_t isum = vaddvq_s32(vacc);
#else
                    int16x8_t pa = vmull_s8(vget_low_s8(vw_lo2), vget_low_s8(vx_lo));
                    int16x8_t pb = vmull_s8(vget_high_s8(vw_lo2), vget_high_s8(vx_lo));
                    int16x8_t pc = vmull_s8(vget_low_s8(vw_hi2), vget_low_s8(vx_hi));
                    int16x8_t pd = vmull_s8(vget_high_s8(vw_hi2), vget_high_s8(vx_hi));
                    int32x4_t vacc = vpaddlq_s16(vaddq_s16(vaddq_s16(pa, pb), vaddq_s16(pc, pd)));
                    int32_t isum = vaddvq_s32(vacc);
#endif
                    float xd_n = X_ds[(size_t)n * (n_super * 8) + b * 8 + ib32 + 1];
                    acc[n] += (float)isum * db2 * xd_n;
                }

                qs += 8;
                signs += 4;
                qh += 2;
            }
        }
        for (int n = 0; n < N; n++) {
            out[(size_t)n * out_row_stride_n + d] = acc[n];
        }
    }
}
#endif

typedef struct {
    float* out; const void* weight; const int8_t* x_qs;
    const float* x_ds; size_t row_bytes; int in_dim;
    int start_row; int end_row;
} iq3s_int_task_t;

static void* iq3_s_int_dot_worker(void* arg) {
    iq3s_int_task_t* task = (iq3s_int_task_t*)arg;
    for (int d = task->start_row; d < task->end_row; d++) {
        const void* row = (const uint8_t*)task->weight + (size_t)d * task->row_bytes;
        task->out[d] = fused_dot_iq3_s_int8(row, task->x_qs, task->x_ds, task->in_dim);
    }
    return NULL;
}
#endif /* TQ_HAS_NEON */

/* Fused IQ3_S dot product for tq_matmul_gguf */
static float fused_dot_iq3_s(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_iq3_s_t* blk = (const block_iq3_s_t*)row;
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(blk[i].d);
        const uint8_t* qs = blk[i].qs;
        const uint8_t* qh = blk[i].qh;
        const uint8_t* signs = blk[i].signs;
        const float* xp = x + i * 256;

        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1 + 2 * (blk[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (blk[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; l++) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    float v1 = db1 * grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                    float v2 = db1 * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                    sum += v1 * xp[0] + v2 * xp[4];
                    xp++;
                }
                xp += 4; /* skip to next 8-element group */
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; l++) {
                const uint8_t* grid1 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = (const uint8_t*)(iq3s_grid +
                    (qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    float v1 = db2 * grid1[j] * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
                    float v2 = db2 * grid2[j] * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                    sum += v1 * xp[0] + v2 * xp[4];
                    xp++;
                }
                xp += 4;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
    return sum;
}

/* --- Other IQ type stubs --- */
static void dequant_iq_stub(const char* type_name, float* dst, int n) {
    static int warned_iq2_xs = 0, warned_iq1_s = 0, warned_iq3_s = 0, warned_other = 0;
    int* flag = &warned_other;
    if (strcmp(type_name, "IQ2_XS") == 0) flag = &warned_iq2_xs;
    else if (strcmp(type_name, "IQ1_S") == 0) flag = &warned_iq1_s;
    else if (strcmp(type_name, "IQ3_S") == 0) flag = &warned_iq3_s;
    if (!*flag) {
        fprintf(stderr, "tq_gguf_quants: WARNING: %s dequant not yet implemented, "
                        "returning zeros\n", type_name);
        *flag = 1;
    }
    memset(dst, 0, (size_t)n * sizeof(float));
}

/* ============================================================
 * Main dequantization dispatcher
 * ============================================================ */

void tq_dequant_row_gguf(tq_ggml_dtype type, const void* src, float* dst, int n) {
    switch (type) {
        case TQ_GGML_TYPE_F32:
            dequant_f32(src, dst, n);
            break;
        case TQ_GGML_TYPE_F16:
            dequant_f16(src, dst, n);
            break;
        case TQ_GGML_TYPE_BF16:
            dequant_bf16(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_0:
            dequant_q4_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_1:
            dequant_q4_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_0:
            dequant_q5_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_1:
            dequant_q5_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q8_0:
            dequant_q8_0(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q8_1:
            dequant_q8_1(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q2_K:
            dequant_q2_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q3_K:
            dequant_q3_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q4_K:
            dequant_q4_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q5_K:
            dequant_q5_k(src, dst, n);
            break;
        case TQ_GGML_TYPE_Q6_K:
            dequant_q6_k(src, dst, n);
            break;

        /* IQ stubs */
        case TQ_GGML_TYPE_IQ2_XXS:
            dequant_iq2_xxs(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ2_XS:
            dequant_iq_stub("IQ2_XS", dst, n);
            break;
        case TQ_GGML_TYPE_IQ3_XXS:
            dequant_iq3_xxs(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ1_S:
            dequant_iq_stub("IQ1_S", dst, n);
            break;
        case TQ_GGML_TYPE_IQ4_NL:
            dequant_iq4_nl(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ3_S:
            dequant_iq3_s(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ2_S:
            dequant_iq2_s(src, dst, n);
            break;
        case TQ_GGML_TYPE_IQ4_XS:
            dequant_iq4_xs(src, dst, n);
            break;

        default:
            fprintf(stderr, "tq_gguf_quants: ERROR: unsupported type %d\n", (int)type);
            memset(dst, 0, (size_t)n * sizeof(float));
            break;
    }
}

/* ============================================================
 * Fused dequant-dot product functions
 *
 * These compute dot(dequant(weight_row), input) in a single pass
 * without writing dequantized values to memory. All intermediate
 * values stay in registers, eliminating the temporary FP32 buffer.
 *
 * This is the critical optimization for MoE inference where
 * IQ2_XXS dequant dominates runtime.
 * ============================================================ */

/* Fused IQ2_XXS dot product: dot(dequant(row), x) for one 256-element block */
static inline float dot_block_iq2_xxs(const uint8_t* blk, const float* x) {
    uint16_t d_raw;
    memcpy(&d_raw, blk, 2);
    const float d = fp16_to_fp32(d_raw);
    const uint16_t* qs = (const uint16_t*)(blk + 2);
    float sum = 0.0f;

    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux32[2];
        memcpy(aux32, qs + 4 * ib32, 8);
        const uint8_t* aux8 = (const uint8_t*)aux32;
        const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
        const float* xb = x + ib32 * 32;

        float group_sum = 0.0f;
        for (int l = 0; l < 4; l++) {
            const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
            const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
            const float* xp = xb + l * 8;

#if TQ_HAS_NEON
            /* Load 8 grid values into two int8x8 vectors, apply signs, dot with input */
            /* Grid values are uint8_t (0x08, 0x19, 0x2b), signs are bitmask */
            float local_sum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float w = (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                local_sum += w * xp[j];
            }
            group_sum += local_sum;
#else
            float local_sum = 0.0f;
            for (int j = 0; j < 8; j++) {
                float w = (float)grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                local_sum += w * xp[j];
            }
            group_sum += local_sum;
#endif
        }
        sum += db * group_sum;
    }
    return sum;
}

/* Fused IQ2_XXS row dot: dot product of entire quantized row with input vector.
 * Processes all 256-element super-blocks without any intermediate FP32 buffer.
 * Reserved for future fused matmul optimization path. */
#ifdef _MSC_VER
__pragma(warning(suppress: 4505))
#else
__attribute__((unused))
#endif
static float fused_dot_iq2_xxs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        sum += dot_block_iq2_xxs(base + b * 66, x + b * 256);
    }
    return sum;
}

#if TQ_HAS_NEON

/* Vectorized sign application helper: given 8 grid bytes and an 8-bit sign mask,
 * produce signed int8x8 where negative signs are applied.
 * Uses NEON bit test: broadcast sign byte, AND with bit masks, compare to produce
 * negation mask, then apply via (grid ^ neg) - neg (conditional negate). */
static const uint8_t iq2_sign_bit_masks[8] = {1, 2, 4, 8, 16, 32, 64, 128};

/* NEON-optimized fused IQ2_XXS dot product.
 * Optimizations over baseline:
 *   1. Vectorized sign expansion via NEON bit-test (replaces 8 scalar shifts)
 *   2. Apply signs in int8 domain before float conversion (fewer instructions)
 *   3. Fully unrolled inner loop (4 groups per ib32)
 *   4. Prefetch next block's weight data
 *   5. Two accumulator strategy to reduce FMA dependency chains */
static float fused_dot_iq2_xxs_neon(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float32x4_t vtotal0 = vdupq_n_f32(0.0f);

    /* Preload sign bit masks into a NEON register */
    const uint8x8_t vbit_masks = vld1_u8(iq2_sign_bit_masks);

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 66;

        /* Prefetch next block */
        if (b + 1 < nb) {
            __builtin_prefetch(blk + 66, 0, 3);
            __builtin_prefetch(blk + 66 + 32, 0, 3);
        }

        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs_bytes = blk + 2;
        const float* xbase = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs_bytes + 8 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;
            const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
            const float* xb = xbase + ib32 * 32;

            /* Accumulate across all 4 sub-groups before scaling by db.
             * Use two accumulators to break FMA dependency chains. */
            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);

            /* --- Group 0 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[0]);
                const uint8_t signs = ksigns_iq2xs[aux32[1] & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                /* Vectorized sign expansion:
                 * Broadcast sign byte to all lanes, AND with bit masks,
                 * compare != 0 produces 0xFF for negative lanes.
                 * Then: signed = (grid ^ neg_mask) - neg_mask
                 *       which is grid when neg_mask=0, -grid when neg_mask=0xFF */
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                /* vsign_bits is 0xFF where negative, 0x00 where positive */
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                /* Widen to int16, then int32, then float */
                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 4));
            }

            /* --- Group 1 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[1]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 8));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 12));
            }

            /* --- Group 2 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[2]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 14) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 16));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 20));
            }

            /* --- Group 3 --- */
            {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[3]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 21) & 127];

                uint8x8_t vgrid = vld1_u8(grid);
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg_mask = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg_mask), vneg_mask);

                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xb + 24));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xb + 28));
            }

            /* Combine accumulators, scale by db, accumulate to total */
            float32x4_t vgroup = vaddq_f32(vacc0, vacc1);
            vtotal0 = vfmaq_n_f32(vtotal0, vgroup, db);
        }
    }
    return vaddvq_f32(vtotal0);
}

/* ============================================================
 * IQ2_XXS × int8 fused dot (NEON vdotq_s32 path)
 *
 * Takes pre-quantized activation (int8) instead of float to enable
 * vdotq_s32 (16 int8 ops/cycle on M1 Pro vs vfmaq_f32's 4 float ops).
 *
 * Caller pre-quantizes x to int8[n] with per-32-element scales x_ds[n/32].
 * Result: 0.25 * sum_blocks ( d_block * sum_subblocks( sub_scale *
 *         x_ds[sb] * sum_int8( signed_grid × x_int8 ) ) )
 *
 * Matches llama.cpp's ggml_vec_dot_iq2_xxs_q8_K structure but uses
 * Q8_0-compatible per-32 activation scales (smaller memory traffic).
 * ============================================================ */
static float fused_dot_iq2_xxs_int8(const void* row, const int8_t* x_qs,
                                     const float* x_ds, int n)
{
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    const uint8x8_t vbit_masks = vld1_u8(iq2_sign_bit_masks);

    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 66;
        if (b + 1 < nb) {
            __builtin_prefetch(blk + 66, 0, 3);
            __builtin_prefetch(blk + 66 + 32, 0, 3);
        }

        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs_bytes = blk + 2;

        const int8_t* x_sub = x_qs + b * 256;
        const float*  xd_sub = x_ds + b * 8;

        float sub_sum = 0.0f;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, qs_bytes + 8 * ib32, 8);
            const uint8_t* aux8 = (const uint8_t*)aux32;

            /* Build 4 groups × 8 int8 = 32 signed int8 weights.
             * Each group: grid[aux8[g]] × sign(aux32[1] bits).
             * GNU statement-expression macro: suppress the clang pedantic
             * warning because we compile with extensions enabled. */
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
            #define BUILD_GROUP(g, shift) ({ \
                uint8x8_t vg = vld1_u8((const uint8_t*)(iq2xxs_grid + aux8[g])); \
                uint8_t signs = ksigns_iq2xs[(aux32[1] >> (shift)) & 127]; \
                uint8x8_t vsbits = vtst_u8(vdup_n_u8(signs), vbit_masks); \
                int8x8_t  vs = vreinterpret_s8_u8(vsbits); \
                int8x8_t  vw = vreinterpret_s8_u8(vg); \
                vsub_s8(veor_s8(vw, vs), vs); \
            })

            int8x8_t vs0 = BUILD_GROUP(0,  0);
            int8x8_t vs1 = BUILD_GROUP(1,  7);
            int8x8_t vs2 = BUILD_GROUP(2, 14);
            int8x8_t vs3 = BUILD_GROUP(3, 21);
            #undef BUILD_GROUP
            #pragma clang diagnostic pop

            int8x16_t vw_lo = vcombine_s8(vs0, vs1);
            int8x16_t vw_hi = vcombine_s8(vs2, vs3);

            /* Load 32 int8 activation values for this sub-block */
            int8x16_t vx_lo = vld1q_s8(x_sub + ib32 * 32);
            int8x16_t vx_hi = vld1q_s8(x_sub + ib32 * 32 + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vacc = vdotq_s32(vdupq_n_s32(0), vw_lo, vx_lo);
            vacc = vdotq_s32(vacc, vw_hi, vx_hi);
            int32_t isum = vaddvq_s32(vacc);
#else
            int16x8_t vp_lo_lo = vmull_s8(vs0, vget_low_s8(vx_lo));
            int16x8_t vp_lo_hi = vmull_s8(vs1, vget_high_s8(vx_lo));
            int16x8_t vp_hi_lo = vmull_s8(vs2, vget_low_s8(vx_hi));
            int16x8_t vp_hi_hi = vmull_s8(vs3, vget_high_s8(vx_hi));
            int32x4_t vacc = vpaddlq_s16(vp_lo_lo);
            vacc = vpadalq_s16(vacc, vp_lo_hi);
            vacc = vpadalq_s16(vacc, vp_hi_lo);
            vacc = vpadalq_s16(vacc, vp_hi_hi);
            int32_t isum = vaddvq_s32(vacc);
            (void)vone;
#endif

            float sub_scale = 0.5f + (float)(aux32[1] >> 28);
            sub_sum += (float)isum * sub_scale * xd_sub[ib32];
        }

        sumf += d * sub_sum;
    }

    return 0.25f * sumf;
}

/* Row-sliced thread worker for IQ2_XXS × int8 matmul. */
typedef struct {
    float* out; const void* weight; const int8_t* x_qs;
    const float* x_ds; size_t row_bytes; int in_dim;
    int start_row; int end_row;
} iq2_int_task_t;

void* iq2_xxs_int_dot_worker(void* arg) {
    iq2_int_task_t* task = (iq2_int_task_t*)arg;
    for (int d = task->start_row; d < task->end_row; d++) {
        const void* row = (const uint8_t*)task->weight + (size_t)d * task->row_bytes;
        task->out[d] = fused_dot_iq2_xxs_int8(row, task->x_qs, task->x_ds, task->in_dim);
    }
    return NULL;
}

/* ============================================================
 * IQ2_S × int8 fused dot (NEON vdotq_s32 path)
 *
 * Same structure as IQ2_XXS int8 kernel, but IQ2_S's 10-bit grid index
 * and per-half-subblock sub-scales. Block = 82 bytes (vs 66 for XXS).
 * ============================================================ */
static float fused_dot_iq2_s_int8(const void* row, const int8_t* x_qs,
                                   const float* x_ds, int n)
{
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    const uint8x8_t vbit_masks = vld1_u8(iq2_sign_bit_masks);

    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 82;
        if (b + 1 < nb) {
            __builtin_prefetch(blk + 82, 0, 3);
            __builtin_prefetch(blk + 82 + 32, 0, 3);
        }

        uint16_t d_raw; memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs_base = blk + 2;
        const uint8_t* signs_base = blk + 34;
        const uint8_t* qh = blk + 66;
        const uint8_t* scales = blk + 74;

        const int8_t* x_sub = x_qs + b * 256;
        const float*  xd_sub = x_ds + b * 8;

        float sub_sum = 0.0f;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            const uint8_t* qs = qs_base + ib32 * 4;
            const uint8_t* sn = signs_base + ib32 * 4;
            float sub_scale_lo = 0.5f + (float)(scales[ib32] & 0xF);
            float sub_scale_hi = 0.5f + (float)(scales[ib32] >> 4);

            int8x8_t vg[4];
            for (int l = 0; l < 4; l++) {
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                uint8x8_t raw = vld1_u8((const uint8_t*)(iq2s_grid + grid_idx));
                uint8_t sign = sn[l];
                uint8x8_t sbits = vtst_u8(vdup_n_u8(sign), vbit_masks);
                int8x8_t vneg = vreinterpret_s8_u8(sbits);
                int8x8_t vr = vreinterpret_s8_u8(raw);
                vg[l] = vsub_s8(veor_s8(vr, vneg), vneg);
            }

            int8x16_t vw_lo = vcombine_s8(vg[0], vg[1]);  /* groups 0-1, sub_scale_lo */
            int8x16_t vw_hi = vcombine_s8(vg[2], vg[3]);  /* groups 2-3, sub_scale_hi */

            int8x16_t vx_lo = vld1q_s8(x_sub + ib32 * 32);
            int8x16_t vx_hi = vld1q_s8(x_sub + ib32 * 32 + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32_t isum_lo = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vw_lo, vx_lo));
            int32_t isum_hi = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vw_hi, vx_hi));
#else
            int16x8_t p0 = vmull_s8(vg[0], vget_low_s8(vx_lo));
            int16x8_t p1 = vmull_s8(vg[1], vget_high_s8(vx_lo));
            int16x8_t p2 = vmull_s8(vg[2], vget_low_s8(vx_hi));
            int16x8_t p3 = vmull_s8(vg[3], vget_high_s8(vx_hi));
            int32_t isum_lo = vaddvq_s32(vpaddlq_s16(p0)) + vaddvq_s32(vpaddlq_s16(p1));
            int32_t isum_hi = vaddvq_s32(vpaddlq_s16(p2)) + vaddvq_s32(vpaddlq_s16(p3));
#endif

            sub_sum += xd_sub[ib32] *
                       (sub_scale_lo * (float)isum_lo + sub_scale_hi * (float)isum_hi);
        }

        sumf += d * sub_sum;
    }

    return 0.25f * sumf;
}

void* iq2_s_int_dot_worker(void* arg) {
    iq2_int_task_t* task = (iq2_int_task_t*)arg;
    for (int d = task->start_row; d < task->end_row; d++) {
        const void* row = (const uint8_t*)task->weight + (size_t)d * task->row_bytes;
        task->out[d] = fused_dot_iq2_s_int8(row, task->x_qs, task->x_ds, task->in_dim);
    }
    return NULL;
}

#endif /* TQ_HAS_NEON */

/* Fused IQ2_S dot product */
static float fused_dot_iq2_s(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 82;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);

        const uint8_t* qs_base = blk + 2;
        const uint8_t* signs_base = qs_base + 32;
        const uint8_t* qh = blk + 66;
        const uint8_t* scales = blk + 74;
        const float* xbase = x + b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            const uint8_t* qs = qs_base + ib32 * 4;
            const uint8_t* sn = signs_base + ib32 * 4;
            float db0 = d * (0.5f + (float)(scales[ib32] & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;

            for (int l = 0; l < 4; l++) {
                float dl = (l < 2) ? db0 : db1;
                int grid_idx = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + grid_idx);
                uint8_t sign = sn[l];
                const float* xp = xbase + ib32 * 32 + l * 8;

                float local_sum = 0.0f;
                for (int j = 0; j < 8; j++) {
                    float w = (float)grid[j] * ((sign & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    local_sum += w * xp[j];
                }
                sum += dl * local_sum;
            }
        }
    }
    return sum;
}

/* Fused Q8_0 dot product: 34 bytes per 32 elements */
static float fused_dot_q8_0(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_q8_0* blk = (const block_q8_0*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    /* Two-accumulator NEON with prefetch for Q8_0.
     * Process 2 blocks per iteration to break FMA dependency chains. */
    float32x4_t vtotal0 = vdupq_n_f32(0.0f);
    float32x4_t vtotal1 = vdupq_n_f32(0.0f);

    int b = 0;
    for (; b + 1 < nb; b += 2) {
        /* Prefetch next pair */
        if (b + 3 < nb) __builtin_prefetch(&blk[b + 2], 0, 3);

        /* Block b */
        const float d0 = fp16_to_fp32(blk[b].d);
        const float* xp0 = x + b * 32;
        float32x4_t vs0 = vdupq_n_f32(0.0f);
        float32x4_t vs1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs0 = vfmaq_f32(vs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp0 + j));
            vs1 = vfmaq_f32(vs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp0 + j + 4));
        }
        vtotal0 = vfmaq_n_f32(vtotal0, vaddq_f32(vs0, vs1), d0);

        /* Block b+1 */
        const float d1 = fp16_to_fp32(blk[b + 1].d);
        const float* xp1 = x + (b + 1) * 32;
        float32x4_t vs2 = vdupq_n_f32(0.0f);
        float32x4_t vs3 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b + 1].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs2 = vfmaq_f32(vs2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp1 + j));
            vs3 = vfmaq_f32(vs3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp1 + j + 4));
        }
        vtotal1 = vfmaq_n_f32(vtotal1, vaddq_f32(vs2, vs3), d1);
    }
    /* Handle odd block */
    for (; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;
        float32x4_t vs0 = vdupq_n_f32(0.0f), vs1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs0 = vfmaq_f32(vs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp + j));
            vs1 = vfmaq_f32(vs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp + j + 4));
        }
        vtotal0 = vfmaq_n_f32(vtotal0, vaddq_f32(vs0, vs1), d);
    }
    sum = vaddvq_f32(vaddq_f32(vtotal0, vtotal1));
#else
    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++) block_sum += (float)blk[b].qs[j] * xp[j];
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused Q8_1 dot product: 36 bytes per 32 elements.
 * Same math as Q8_0 (val = d * qs[i]) but different block layout.
 * The `s` (sum) field is only used for Q4×Q8_1 mixed-type dot products. */
static float fused_dot_q8_1(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_q8_1* blk = (const block_q8_1*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    float32x4_t vtotal0 = vdupq_n_f32(0.0f);
    float32x4_t vtotal1 = vdupq_n_f32(0.0f);

    int b = 0;
    for (; b + 1 < nb; b += 2) {
        if (b + 3 < nb) __builtin_prefetch(&blk[b + 2], 0, 3);

        const float d0 = fp16_to_fp32(blk[b].d);
        const float* xp0 = x + b * 32;
        float32x4_t vs0 = vdupq_n_f32(0.0f), vs1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs0 = vfmaq_f32(vs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp0 + j));
            vs1 = vfmaq_f32(vs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp0 + j + 4));
        }
        vtotal0 = vfmaq_n_f32(vtotal0, vaddq_f32(vs0, vs1), d0);

        const float d1 = fp16_to_fp32(blk[b + 1].d);
        const float* xp1 = x + (b + 1) * 32;
        float32x4_t vs2 = vdupq_n_f32(0.0f), vs3 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b + 1].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs2 = vfmaq_f32(vs2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp1 + j));
            vs3 = vfmaq_f32(vs3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp1 + j + 4));
        }
        vtotal1 = vfmaq_n_f32(vtotal1, vaddq_f32(vs2, vs3), d1);
    }
    for (; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;
        float32x4_t vs0 = vdupq_n_f32(0.0f), vs1 = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 8) {
            int8x8_t vq = vld1_s8(blk[b].qs + j);
            int16x8_t vq16 = vmovl_s8(vq);
            vs0 = vfmaq_f32(vs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(vq16))),  vld1q_f32(xp + j));
            vs1 = vfmaq_f32(vs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(vq16))), vld1q_f32(xp + j + 4));
        }
        vtotal0 = vfmaq_n_f32(vtotal0, vaddq_f32(vs0, vs1), d);
    }
    sum = vaddvq_f32(vaddq_f32(vtotal0, vtotal1));
#else
    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 32; j++) block_sum += (float)blk[b].qs[j] * xp[j];
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused IQ4_NL dot product: 18 bytes per 32 elements */
static float fused_dot_iq4_nl(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_iq4_nl* blk = (const block_iq4_nl*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    /* Preload IQ4_NL lookup table into 2 NEON registers (16 values split into 2×8).
     * kvalues_iq4nl[0..15] are int8 values, we load as two int8x8 for tbl lookup. */
    int8x16_t vlut = vld1q_s8(kvalues_iq4nl);
    float32x4_t vsum0 = vdupq_n_f32(0.0f);
    uint8x16_t vmask_lo = vdupq_n_u8(0x0f);

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xp = x + b * 32;

        /* Load 16 bytes of qs, split into low/high nibbles → lookup → signed int8 weights */
        uint8x16_t vqs = vld1q_u8(qs);
        uint8x16_t vlo = vandq_u8(vqs, vmask_lo);         /* low nibbles [0..15] */
        uint8x16_t vhi = vshrq_n_u8(vqs, 4);              /* high nibbles [0..15] */
        int8x16_t wlo = vqtbl1q_s8(vlut, vlo);            /* lookup low → signed weights */
        int8x16_t whi = vqtbl1q_s8(vlut, vhi);            /* lookup high → signed weights */

        /* Convert to float and accumulate: wlo[j]*xp[j] for j=0..15, whi[j]*xp[j+16] */
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (int k = 0; k < 4; k++) {
            /* Low nibble: 4 elements */
            int16x8_t w16 = vmovl_s8(vget_low_s8(wlo));
            if (k >= 2) w16 = vmovl_s8(vget_high_s8(wlo));
            int16x4_t w16_part = (k & 1) ? vget_high_s16(w16) : vget_low_s16(w16);
            float32x4_t vw = vcvtq_f32_s32(vmovl_s16(w16_part));
            float32x4_t vx = vld1q_f32(xp + k * 4);
            vacc = vfmaq_f32(vacc, vw, vx);

            /* High nibble: 4 elements at xp+16 */
            int16x8_t wh16 = vmovl_s8(vget_low_s8(whi));
            if (k >= 2) wh16 = vmovl_s8(vget_high_s8(whi));
            int16x4_t wh16_part = (k & 1) ? vget_high_s16(wh16) : vget_low_s16(wh16);
            float32x4_t vwh = vcvtq_f32_s32(vmovl_s16(wh16_part));
            float32x4_t vxh = vld1q_f32(xp + 16 + k * 4);
            vacc = vfmaq_f32(vacc, vwh, vxh);
        }
        float block_sum = vaddvq_f32(vacc);
        vsum0 = vfmaq_n_f32(vsum0, vdupq_n_f32(block_sum), d);
    }
    sum = vaddvq_f32(vsum0);
#else
    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xp = x + b * 32;

        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            block_sum += (float)kvalues_iq4nl[qs[j] & 0xf] * xp[j];
            block_sum += (float)kvalues_iq4nl[qs[j] >> 4]  * xp[j + 16];
        }
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused IQ4_XS dot product: 136 bytes per 256 elements */
/* ============================================================
 * IQ4_XS × int8 fused dot (NEON vdotq_s32 with vqtbl1q_s8 lookup)
 *
 * Layout (136 bytes per 256 elements):
 *   d         fp16 super-scale
 *   scales_h  uint16, 2-bit high scale per 8 sub-blocks
 *   scales_l  uint8[4], 4-bit low scale per 8 sub-blocks (2 per byte)
 *   qs[128]   4-bit quantized indices (low nibble = elems 0..15,
 *             high nibble = elems 16..31 within each 32-elem sub-block)
 *
 * Each 4-bit index looks up kvalues_iq4nl[16] (non-linear int8 codebook).
 * The codebook fits in 16 bytes — perfect for ARM NEON `vqtbl1q_s8`,
 * which does 16 parallel byte-indexed lookups in one cycle.
 *
 * Per 32-elem sub-block: 2 vqtbl1q_s8 + 2 vdotq_s32 + scalar combine.
 * Per 256-element block: 8 sub-blocks × 2 = 16 vdotq_s32.
 * ============================================================ */
#if TQ_HAS_NEON
static float fused_dot_iq4_xs_int8(const void* row, const int8_t* x_qs,
                                     const float* x_ds, int n)
{
    const int nb = n / 256;
    const block_iq4_xs* blk = (const block_iq4_xs*)row;
    const int8x16_t vtbl = vld1q_s8(kvalues_iq4nl);  /* 16-entry codebook */

    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) {
            __builtin_prefetch((const uint8_t*)&blk[b+1], 0, 3);
            __builtin_prefetch((const uint8_t*)&blk[b+1] + 64, 0, 3);
        }
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const int8_t* xb = x_qs + b * 256;
        const float*  xd = x_ds + b * 8;

        float block_sum = 0.0f;

        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((blk[b].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                         | (((blk[b].scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);

            uint8x16_t qs_v = vld1q_u8(qs);        /* 16 qs bytes = 32 elems */
            uint8x16_t low  = vandq_u8(qs_v, vdupq_n_u8(0x0F));
            uint8x16_t high = vshrq_n_u8(qs_v, 4);
            int8x16_t w_lo  = vqtbl1q_s8(vtbl, low);   /* 16 int8 weights [0..15] */
            int8x16_t w_hi  = vqtbl1q_s8(vtbl, high);  /* 16 int8 weights [16..31] */

            int8x16_t vx_lo = vld1q_s8(xb + ib * 32);
            int8x16_t vx_hi = vld1q_s8(xb + ib * 32 + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vacc = vdotq_s32(vdupq_n_s32(0), w_lo, vx_lo);
            vacc = vdotq_s32(vacc, w_hi, vx_hi);
            int32_t isum = vaddvq_s32(vacc);
#else
            int16x8_t pa = vmull_s8(vget_low_s8(w_lo),  vget_low_s8(vx_lo));
            int16x8_t pb = vmull_s8(vget_high_s8(w_lo), vget_high_s8(vx_lo));
            int16x8_t pc = vmull_s8(vget_low_s8(w_hi),  vget_low_s8(vx_hi));
            int16x8_t pd = vmull_s8(vget_high_s8(w_hi), vget_high_s8(vx_hi));
            int32x4_t vacc = vpaddlq_s16(vaddq_s16(vaddq_s16(pa, pb), vaddq_s16(pc, pd)));
            int32_t isum = vaddvq_s32(vacc);
#endif
            block_sum += (float)isum * dl * xd[ib];
            qs += 16;
        }

        sumf += block_sum;
    }
    return sumf;
}

/* ============================================================
 * IQ4_XS × int8 BATCHED dot — Mission A Step 3c.
 *
 * Same amortization pattern. IQ4_XS is simplest of the three because
 * weight unpack = one vqtbl1q_s8 (TBL-16 lookup of kvalues_iq4nl).
 * Small share of Qwen3.6 compute (0.9%) but completes the expert
 * kernel trio (IQ3_XXS + IQ3_S + IQ4_XS = 55.5% of prefill compute).
 * ============================================================ */
#if TQ_HAS_NEON
static void fused_dot_iq4_xs_int8_batched(
    float* out,
    const void* weight,
    size_t row_bytes,
    int out_row_stride_n,
    const int8_t* X_qs,
    const float* X_ds,
    int start_row, int end_row,
    int n_super, int N)
{
    const int8x16_t vtbl = vld1q_s8(kvalues_iq4nl);

    for (int d = start_row; d < end_row; d++) {
        const uint8_t* base = (const uint8_t*)weight + (size_t)d * row_bytes;

        float acc[64];
        if (N > 64) return;
        memset(acc, 0, (size_t)N * sizeof(float));

        for (int b = 0; b < n_super; b++) {
            const block_iq4_xs* blk = (const block_iq4_xs*)(base + (size_t)b * sizeof(block_iq4_xs));
            if (b + 1 < n_super) {
                __builtin_prefetch((const uint8_t*)blk + 136, 0, 3);
                __builtin_prefetch((const uint8_t*)blk + 136 + 64, 0, 3);
            }
            const float d_super = fp16_to_fp32(blk->d);
            const uint8_t* qs = blk->qs;

            /* 8 sub-blocks × 32 elems each */
            for (int ib = 0; ib < 8; ib++) {
                const int ls = ((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                             | (((blk->scales_h >> (2 * ib)) & 3) << 4);
                const float dl = d_super * (float)(ls - 32);

                /* Unpack weights ONCE for this (row, block, sub-block) */
                uint8x16_t qs_v = vld1q_u8(qs);
                uint8x16_t low  = vandq_u8(qs_v, vdupq_n_u8(0x0F));
                uint8x16_t high = vshrq_n_u8(qs_v, 4);
                int8x16_t w_lo  = vqtbl1q_s8(vtbl, low);
                int8x16_t w_hi  = vqtbl1q_s8(vtbl, high);

                /* Inner batch loop */
                for (int n = 0; n < N; n++) {
                    const int8_t* xb_n = X_qs + (size_t)n * (n_super * 256) + b * 256 + ib * 32;
                    int8x16_t vx_lo = vld1q_s8(xb_n +  0);
                    int8x16_t vx_hi = vld1q_s8(xb_n + 16);
#ifdef __ARM_FEATURE_DOTPROD
                    int32x4_t vacc = vdotq_s32(vdupq_n_s32(0), w_lo, vx_lo);
                    vacc = vdotq_s32(vacc, w_hi, vx_hi);
                    int32_t isum = vaddvq_s32(vacc);
#else
                    int16x8_t pa = vmull_s8(vget_low_s8(w_lo),  vget_low_s8(vx_lo));
                    int16x8_t pb = vmull_s8(vget_high_s8(w_lo), vget_high_s8(vx_lo));
                    int16x8_t pc = vmull_s8(vget_low_s8(w_hi),  vget_low_s8(vx_hi));
                    int16x8_t pd = vmull_s8(vget_high_s8(w_hi), vget_high_s8(vx_hi));
                    int32x4_t vacc = vpaddlq_s16(vaddq_s16(vaddq_s16(pa, pb), vaddq_s16(pc, pd)));
                    int32_t isum = vaddvq_s32(vacc);
#endif
                    float xd_n = X_ds[(size_t)n * (n_super * 8) + b * 8 + ib];
                    acc[n] += (float)isum * dl * xd_n;
                }
                qs += 16;
            }
        }
        for (int n = 0; n < N; n++) {
            out[(size_t)n * out_row_stride_n + d] = acc[n];
        }
    }
}
#endif

typedef struct {
    float* out; const void* weight; const int8_t* x_qs;
    const float* x_ds; size_t row_bytes; int in_dim;
    int start_row; int end_row;
} iq4xs_int_task_t;

static void* iq4_xs_int_dot_worker(void* arg) {
    iq4xs_int_task_t* task = (iq4xs_int_task_t*)arg;
    for (int d = task->start_row; d < task->end_row; d++) {
        const void* row = (const uint8_t*)task->weight + (size_t)d * task->row_bytes;
        task->out[d] = fused_dot_iq4_xs_int8(row, task->x_qs, task->x_ds, task->in_dim);
    }
    return NULL;
}
#endif /* TQ_HAS_NEON */

static float fused_dot_iq4_xs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_iq4_xs* blk = (const block_iq4_xs*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* qs = blk[b].qs;
        const float* xbase = x + b * 256;

        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((blk[b].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf)
                         | (((blk[b].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            const float* xp = xbase + ib * 32;

            float block_sum = 0.0f;
            for (int j = 0; j < 16; j++) {
                block_sum += (float)kvalues_iq4nl[qs[j] & 0xf] * xp[j];
                block_sum += (float)kvalues_iq4nl[qs[j] >> 4]  * xp[j + 16];
            }
            sum += dl * block_sum;
            qs += 16;
        }
    }
    return sum;
}

/* ============================================================
 * IQ3_XXS × int8 fused dot (NEON vdotq_s32 path)
 *
 * IQ3_XXS layout (98 bytes per 256 elements):
 *   d (fp16): 2 bytes
 *   qs[64]: 8-bit grid indices (each indexes iq3xxs_grid[256] → 4 uint8 values)
 *   scales_and_signs[32]: 8 × uint32, each holds 4 sign fields (7-bit each in
 *     lower 28 bits) + sub-block scale index (4-bit in upper 4 bits).
 *
 * Per 32-elem sub-block (8 sub-blocks per block):
 *   db = (0.5 + (aux32 >> 28)) * 0.5
 *   For l=0..3: fetch grid[qs[2l]] + grid[qs[2l+1]], apply signs from
 *     ksigns_iq2xs[(aux32 >> (7l)) & 127] → 8 signed int8 values per l
 *   32 weights total per sub-block = 2 × int8x16 = 2 vdotq_s32
 * ============================================================ */
#if TQ_HAS_NEON
static float fused_dot_iq3_xxs_int8(const void* row, const int8_t* x_qs,
                                     const float* x_ds, int n)
{
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    const uint8x8_t vbit_masks = vld1_u8(iq3s_neon_bit_masks);

    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98;
        if (b + 1 < nb) {
            __builtin_prefetch(blk + 98, 0, 3);
            __builtin_prefetch(blk + 98 + 64, 0, 3);
        }
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;

        const int8_t* xb = x_qs + b * 256;
        const float*  xd = x_ds + b * 8;

        float block_sum = 0.0f;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = (0.5f + (float)(aux32 >> 28)) * 0.5f;

            /* Build 32 signed int8 weights from 8 grid lookups (4 bytes each)
             * with signs from (aux32 >> 7l) & 127 lookup. */
            uint32_t g[8];
            for (int l = 0; l < 4; l++) {
                g[2*l + 0] = iq3xxs_grid[qs[2*l + 0]];
                g[2*l + 1] = iq3xxs_grid[qs[2*l + 1]];
            }
            /* signs_byte for each l-pair comes from ksigns_iq2xs lookup */
            uint8_t s0 = ksigns_iq2xs[(aux32 >> (7 * 0)) & 127];
            uint8_t s1 = ksigns_iq2xs[(aux32 >> (7 * 1)) & 127];
            uint8_t s2 = ksigns_iq2xs[(aux32 >> (7 * 2)) & 127];
            uint8_t s3 = ksigns_iq2xs[(aux32 >> (7 * 3)) & 127];

            int8x8_t w0 = iq3s_build8(g[0], g[1], s0, vbit_masks);
            int8x8_t w1 = iq3s_build8(g[2], g[3], s1, vbit_masks);
            int8x8_t w2 = iq3s_build8(g[4], g[5], s2, vbit_masks);
            int8x8_t w3 = iq3s_build8(g[6], g[7], s3, vbit_masks);

            int8x16_t vw_lo = vcombine_s8(w0, w1);
            int8x16_t vw_hi = vcombine_s8(w2, w3);

            int8x16_t vx_lo = vld1q_s8(xb + ib32 * 32 + 0);
            int8x16_t vx_hi = vld1q_s8(xb + ib32 * 32 + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vacc = vdotq_s32(vdupq_n_s32(0), vw_lo, vx_lo);
            vacc = vdotq_s32(vacc, vw_hi, vx_hi);
            int32_t isum = vaddvq_s32(vacc);
#else
            int16x8_t pa = vmull_s8(vget_low_s8(vw_lo),  vget_low_s8(vx_lo));
            int16x8_t pb = vmull_s8(vget_high_s8(vw_lo), vget_high_s8(vx_lo));
            int16x8_t pc = vmull_s8(vget_low_s8(vw_hi),  vget_low_s8(vx_hi));
            int16x8_t pd = vmull_s8(vget_high_s8(vw_hi), vget_high_s8(vx_hi));
            int32x4_t vacc = vpaddlq_s16(vaddq_s16(vaddq_s16(pa, pb), vaddq_s16(pc, pd)));
            int32_t isum = vaddvq_s32(vacc);
#endif
            block_sum += (float)isum * db * xd[ib32];
            qs += 8;  /* advance past the 8 qs bytes consumed by this sub-block */
        }

        sumf += d * block_sum;
    }
    return sumf;
}

/* ============================================================
 * IQ3_XXS × int8 BATCHED dot — Mission A Step 3 core kernel.
 *
 * Shape: out[N, out_dim] = X[N, in_dim] @ W[out_dim, in_dim]^T
 * where W is block_iq3_xxs_gguf (98 bytes per 256 elements).
 *
 * The key optimization vs calling fused_dot_iq3_xxs_int8 N times:
 *   For each (output_row, block), unpack the 32 signed int8 weights ONCE,
 *   then dot them against all N batched activations. This amortizes the
 *   grid lookup + sign application across the batch dim.
 *
 * Expected speedup at N≥8: ≥ 3× on the IQ3_XXS compute path (most of
 * Qwen3.6 MoE expert cost). Mission A Step 3 core.
 * ============================================================ */
#if TQ_HAS_NEON
static void fused_dot_iq3_xxs_int8_batched(
    float* out,              /* [N, n_rows] row-major */
    const void* weight,      /* block_iq3_xxs_gguf array, n_rows × 98 × n_super */
    size_t row_bytes,        /* = n_super × 98 */
    int out_row_stride_n,    /* = n_rows (for indexing out[n*n_rows + d]) */
    const int8_t* X_qs,      /* [N, n_super × 256] int8 */
    const float* X_ds,       /* [N, n_super × 8] fp32 scales */
    int start_row, int end_row,
    int n_super, int N)
{
    const uint8x8_t vbit_masks = vld1_u8(iq3s_neon_bit_masks);

    for (int d = start_row; d < end_row; d++) {
        const uint8_t* base = (const uint8_t*)weight + (size_t)d * row_bytes;

        /* Per-batch float accumulator across blocks */
        float acc[64];
        if (N > 64) return;  /* safety cap */
        memset(acc, 0, (size_t)N * sizeof(float));

        for (int b = 0; b < n_super; b++) {
            const uint8_t* blk = base + (size_t)b * 98;
            if (b + 1 < n_super) {
                __builtin_prefetch(blk + 98, 0, 3);
                __builtin_prefetch(blk + 98 + 64, 0, 3);
            }
            uint16_t d_raw;
            memcpy(&d_raw, blk, 2);
            float d_super = fp16_to_fp32(d_raw);
            const uint8_t* qs = blk + 2;
            const uint8_t* scales_and_signs = qs + 64;

            for (int ib32 = 0; ib32 < 8; ib32++) {
                uint32_t aux32;
                memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
                const float db = (0.5f + (float)(aux32 >> 28)) * 0.5f;

                /* Build 32 signed int8 weights — N-invariant (compute once). */
                uint32_t g[8];
                for (int l = 0; l < 4; l++) {
                    g[2*l + 0] = iq3xxs_grid[qs[2*l + 0]];
                    g[2*l + 1] = iq3xxs_grid[qs[2*l + 1]];
                }
                uint8_t s0 = ksigns_iq2xs[(aux32 >> (7 * 0)) & 127];
                uint8_t s1 = ksigns_iq2xs[(aux32 >> (7 * 1)) & 127];
                uint8_t s2 = ksigns_iq2xs[(aux32 >> (7 * 2)) & 127];
                uint8_t s3 = ksigns_iq2xs[(aux32 >> (7 * 3)) & 127];

                int8x8_t w0 = iq3s_build8(g[0], g[1], s0, vbit_masks);
                int8x8_t w1 = iq3s_build8(g[2], g[3], s1, vbit_masks);
                int8x8_t w2 = iq3s_build8(g[4], g[5], s2, vbit_masks);
                int8x8_t w3 = iq3s_build8(g[6], g[7], s3, vbit_masks);
                int8x16_t vw_lo = vcombine_s8(w0, w1);
                int8x16_t vw_hi = vcombine_s8(w2, w3);

                /* Inner batch loop — reuses the unpacked weight across N activations */
                const float d_db = d_super * db;  /* combined super-scale × sub-scale */
                for (int n = 0; n < N; n++) {
                    const int8_t* xb_n = X_qs + (size_t)n * (n_super * 256) + b * 256 + ib32 * 32;
                    int8x16_t vx_lo = vld1q_s8(xb_n +  0);
                    int8x16_t vx_hi = vld1q_s8(xb_n + 16);

#ifdef __ARM_FEATURE_DOTPROD
                    int32x4_t vacc = vdotq_s32(vdupq_n_s32(0), vw_lo, vx_lo);
                    vacc = vdotq_s32(vacc, vw_hi, vx_hi);
                    int32_t isum = vaddvq_s32(vacc);
#else
                    int16x8_t pa = vmull_s8(vget_low_s8(vw_lo),  vget_low_s8(vx_lo));
                    int16x8_t pb = vmull_s8(vget_high_s8(vw_lo), vget_high_s8(vx_lo));
                    int16x8_t pc = vmull_s8(vget_low_s8(vw_hi),  vget_low_s8(vx_hi));
                    int16x8_t pd = vmull_s8(vget_high_s8(vw_hi), vget_high_s8(vx_hi));
                    int32x4_t vacc = vpaddlq_s16(vaddq_s16(vaddq_s16(pa, pb), vaddq_s16(pc, pd)));
                    int32_t isum = vaddvq_s32(vacc);
#endif
                    float xd_n = X_ds[(size_t)n * (n_super * 8) + b * 8 + ib32];
                    /* Apply combined scale (d_super × db × x_scale) per-N.
                     * Match single-query scale convention:
                     * single: sub_sum += isum * db * xd[ib32];  sumf += d_super * sub_sum;
                     *         returns sumf (NO 0.25f — unlike IQ2_XXS/IQ2_S).
                     * Equivalent: acc[n] += d_super * db * isum * xd_n */
                    acc[n] += d_db * xd_n * (float)isum;
                }
                /* Advance qs by 8 per ib32 iteration (single-query does qs += 8).
                 * Without this the kernel re-reads qs[0..7] for every sub-block. */
                qs += 8;
            }
        }

        /* Store results row-wise */
        for (int n = 0; n < N; n++) {
            out[(size_t)n * out_row_stride_n + d] = acc[n];
        }
    }
}
#endif

typedef struct {
    float* out; const void* weight; const int8_t* x_qs;
    const float* x_ds; size_t row_bytes; int in_dim;
    int start_row; int end_row;
} iq3xxs_int_task_t;

static void* iq3_xxs_int_dot_worker(void* arg) {
    iq3xxs_int_task_t* task = (iq3xxs_int_task_t*)arg;
    for (int d = task->start_row; d < task->end_row; d++) {
        const void* row = (const uint8_t*)task->weight + (size_t)d * task->row_bytes;
        task->out[d] = fused_dot_iq3_xxs_int8(row, task->x_qs, task->x_ds, task->in_dim);
    }
    return NULL;
}
#endif /* TQ_HAS_NEON */

/* Fused IQ3_XXS dot product: 98 bytes per 256 elements
 * Same layout as dequant_iq3_xxs but computes dot product without materializing FP32.
 * 8 groups of 32 elements per block. Each group: 4 sub-groups of 8 elements.
 * Grid lookup + sign application + dot in one pass. */
static float fused_dot_iq3_xxs(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const uint8_t* base = (const uint8_t*)row;
    float sum = 0.0f;

#if TQ_HAS_NEON
    const uint8x8_t vbit_masks = vld1_u8(iq2_sign_bit_masks);

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        const float* xbase = x + b * 256;

        float block_sum = 0.0f;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const float* xb = xbase + ib32 * 32;

            float32x4_t vacc0 = vdupq_n_f32(0.0f);
            float32x4_t vacc1 = vdupq_n_f32(0.0f);

            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);

                /* Load 4+4 grid bytes, combine into one 8-byte vector */
                uint8_t grid8[8];
                memcpy(grid8, grid1, 4);
                memcpy(grid8 + 4, grid2, 4);
                uint8x8_t vgrid = vld1_u8(grid8);

                /* Vectorized sign: broadcast sign byte, AND with masks, compare */
                uint8x8_t vsign_bcast = vdup_n_u8(signs);
                uint8x8_t vsign_bits = vtst_u8(vsign_bcast, vbit_masks);
                int8x8_t vgrid_s = vreinterpret_s8_u8(vgrid);
                int8x8_t vneg = vreinterpret_s8_u8(vsign_bits);
                int8x8_t vsigned = vsub_s8(veor_s8(vgrid_s, vneg), vneg);

                /* Widen to float and dot with input */
                int16x8_t vs16 = vmovl_s8(vsigned);
                float32x4_t vf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vs16)));
                float32x4_t vf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vs16)));

                const float* xp = xb + l * 8;
                vacc0 = vfmaq_f32(vacc0, vf_lo, vld1q_f32(xp));
                vacc1 = vfmaq_f32(vacc1, vf_hi, vld1q_f32(xp + 4));
            }
            block_sum += db * vaddvq_f32(vaddq_f32(vacc0, vacc1));
            qs += 8;
        }
        sum += d * block_sum;
    }
#else
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = base + b * 98;
        uint16_t d_raw;
        memcpy(&d_raw, blk, 2);
        const float d = fp16_to_fp32(d_raw);
        const uint8_t* qs = blk + 2;
        const uint8_t* scales_and_signs = qs + 64;
        const float* xbase = x + b * 256;

        float block_sum = 0.0f;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint32_t aux32;
            memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
            const float db = (0.5f + (float)(aux32 >> 28)) * 0.5f;
            const float* xb = xbase + ib32 * 32;

            float group_sum = 0.0f;
            for (int l = 0; l < 4; l++) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                const float* xp = xb + l * 8;
                for (int j = 0; j < 4; j++) {
                    float w1 = (float)grid1[j] * ((signs & kmask_iq2xs[j])     ? -1.0f : 1.0f);
                    float w2 = (float)grid2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                    group_sum += w1 * xp[j] + w2 * xp[j + 4];
                }
            }
            block_sum += db * group_sum;
            qs += 8;
        }
        sum += d * block_sum;
    }
#endif
    return sum;
}

/* Fused Q4_K dot product: 144 bytes per 256 elements
 * Layout: 4 groups of 64 elements, each group uses 32 bytes of qs.
 *   First 32 elements: d1 * (q[l] & 0xF) - m1  (low nibble, scale pair[0])
 *   Next 32 elements:  d2 * (q[l] >> 4)  - m2   (high nibble, scale pair[1])
 */
/* Fused Q2_K dot product: 84 bytes per 256 elements.
 * 2-bit values packed 4-per-byte in qs[64]. scales[16] holds:
 *   low 4 bits = sub-block scale (× d)
 *   high 4 bits = sub-block min (× dmin)
 * 16 sub-blocks of 16 elements each. Formula per sub-block:
 *   sum += dl * dot(q2_values, x) - ml * sum(x)
 * where q2_values are unsigned 2-bit values in [0, 3]. */
static float fused_dot_q2_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q2_K* blk = (const block_q2_K*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);
        const uint8_t* q = blk[b].qs;
        const float* xp = x + b * 256;

        int is = 0;
        /* 2 halves, 4 shifts per half, 2 sub-blocks per shift */
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                /* Sub-block 0: q[0..15] >> shift & 3 */
                uint8_t sc0 = blk[b].scales[is++];
                float dl0 = d * (sc0 & 0x0F);
                float ml0 = dmin * (sc0 >> 4);

                /* Sub-block 1: q[16..31] >> shift & 3 */
                uint8_t sc1 = blk[b].scales[is++];
                float dl1 = d * (sc1 & 0x0F);
                float ml1 = dmin * (sc1 >> 4);

#if TQ_HAS_NEON
                /* Load 16 packed bytes, extract 2-bit values, dot with x.
                 * Mask is uint8x16_t of 0x03; shift applied via vshrq_n_u8. */
                uint8x16_t qv0 = vld1q_u8(q);        /* sub-block 0 bytes */
                uint8x16_t qv1 = vld1q_u8(q + 16);   /* sub-block 1 bytes */
                uint8x16_t m03 = vdupq_n_u8(0x03);
                uint8x16_t v0, v1;
                switch (shift) {
                    case 0: v0 = vandq_u8(qv0, m03); v1 = vandq_u8(qv1, m03); break;
                    case 2: v0 = vandq_u8(vshrq_n_u8(qv0, 2), m03); v1 = vandq_u8(vshrq_n_u8(qv1, 2), m03); break;
                    case 4: v0 = vandq_u8(vshrq_n_u8(qv0, 4), m03); v1 = vandq_u8(vshrq_n_u8(qv1, 4), m03); break;
                    default: v0 = vshrq_n_u8(qv0, 6); v1 = vshrq_n_u8(qv1, 6); break;
                }
                /* Expand u8 → float32, accumulate dot and sum_x */
                #define TQ_Q2K_ACC(nv, off, dl_v, ml_v)                                        \
                    do {                                                                        \
                        uint16x8_t w16_l = vmovl_u8(vget_low_u8(nv));                           \
                        uint16x8_t w16_h = vmovl_u8(vget_high_u8(nv));                          \
                        float32x4_t wf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_l)));        \
                        float32x4_t wf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_l)));       \
                        float32x4_t wf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_h)));        \
                        float32x4_t wf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_h)));       \
                        float32x4_t x0 = vld1q_f32(xp + (off));                                 \
                        float32x4_t x1 = vld1q_f32(xp + (off) + 4);                             \
                        float32x4_t x2 = vld1q_f32(xp + (off) + 8);                             \
                        float32x4_t x3 = vld1q_f32(xp + (off) + 12);                            \
                        float32x4_t vd = vdupq_n_f32(0.0f);                                     \
                        vd = vfmaq_f32(vd, wf0, x0);                                            \
                        vd = vfmaq_f32(vd, wf1, x1);                                            \
                        vd = vfmaq_f32(vd, wf2, x2);                                            \
                        vd = vfmaq_f32(vd, wf3, x3);                                            \
                        float dot_s = vaddvq_f32(vd);                                           \
                        float sum_s = vaddvq_f32(vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3))); \
                        sum += (dl_v) * dot_s - (ml_v) * sum_s;                                 \
                    } while (0)

                int yi = half * 128 + j * 32;
                TQ_Q2K_ACC(v0, yi, dl0, ml0);
                TQ_Q2K_ACC(v1, yi + 16, dl1, ml1);
                #undef TQ_Q2K_ACC
#else
                int yi = half * 128 + j * 32;
                float dot0 = 0, sumx0 = 0;
                for (int l = 0; l < 16; l++) {
                    float xv = xp[yi + l];
                    dot0 += (float)((q[l] >> shift) & 3) * xv;
                    sumx0 += xv;
                }
                sum += dl0 * dot0 - ml0 * sumx0;

                float dot1 = 0, sumx1 = 0;
                for (int l = 0; l < 16; l++) {
                    float xv = xp[yi + 16 + l];
                    dot1 += (float)((q[l + 16] >> shift) & 3) * xv;
                    sumx1 += xv;
                }
                sum += dl1 * dot1 - ml1 * sumx1;
#endif
                shift += 2;
            }
            q += 32;
        }
    }
    return sum;
}

static float fused_dot_q4_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q4_K* blk = (const block_q4_K*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d    = fp16_to_fp32(blk[b].d);
        const float dmin = fp16_to_fp32(blk[b].dmin);

        uint8_t sc[8], mn[8];
        sc[0] = blk[b].scales[0] & 63;
        sc[1] = blk[b].scales[1] & 63;
        sc[2] = blk[b].scales[2] & 63;
        sc[3] = blk[b].scales[3] & 63;
        mn[0] = blk[b].scales[4] & 63;
        mn[1] = blk[b].scales[5] & 63;
        mn[2] = blk[b].scales[6] & 63;
        mn[3] = blk[b].scales[7] & 63;
        sc[4] = (blk[b].scales[8] & 0x0F) | ((blk[b].scales[0] >> 6) << 4);
        sc[5] = (blk[b].scales[9] & 0x0F) | ((blk[b].scales[1] >> 6) << 4);
        sc[6] = (blk[b].scales[10] & 0x0F) | ((blk[b].scales[2] >> 6) << 4);
        sc[7] = (blk[b].scales[11] & 0x0F) | ((blk[b].scales[3] >> 6) << 4);
        mn[4] = (blk[b].scales[8] >> 4) | ((blk[b].scales[4] >> 6) << 4);
        mn[5] = (blk[b].scales[9] >> 4) | ((blk[b].scales[5] >> 6) << 4);
        mn[6] = (blk[b].scales[10] >> 4) | ((blk[b].scales[6] >> 6) << 4);
        mn[7] = (blk[b].scales[11] >> 4) | ((blk[b].scales[7] >> 6) << 4);

        const uint8_t* q = blk[b].qs;
        const float* xp = x + b * 256;
        int is = 0;

#if TQ_HAS_NEON
        /* 4 groups of 64 elements, NEON-accelerated */
        const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];

            /* Load 32 bytes = 32 pairs of nibbles covering 64 elements */
            uint8x16_t qa = vld1q_u8(q);
            uint8x16_t qb = vld1q_u8(q + 16);
            /* Extract low nibbles (→ elements 0..31) and high nibbles (→ 32..63) */
            uint8x16_t lo_a = vandq_u8(qa, mask_lo);
            uint8x16_t lo_b = vandq_u8(qb, mask_lo);
            uint8x16_t hi_a = vshrq_n_u8(qa, 4);
            uint8x16_t hi_b = vshrq_n_u8(qb, 4);

            /* Convert u8 nibbles [0..15] to float32 and dot with xp[...] */
            float32x4_t vdot1 = vdupq_n_f32(0.0f);
            float32x4_t vsum1 = vdupq_n_f32(0.0f);
            float32x4_t vdot2 = vdupq_n_f32(0.0f);
            float32x4_t vsum2 = vdupq_n_f32(0.0f);
            /* Helper: process 16 elements of x[off:off+16] with nibble vector `nv` */
            #define TQ_Q4K_ACC(nv, off, vdot, vsum)                                              \
                do {                                                                             \
                    uint16x8_t w16_l = vmovl_u8(vget_low_u8(nv));                                \
                    uint16x8_t w16_h = vmovl_u8(vget_high_u8(nv));                               \
                    float32x4_t wf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_l)));             \
                    float32x4_t wf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_l)));            \
                    float32x4_t wf2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_h)));             \
                    float32x4_t wf3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_h)));            \
                    float32x4_t x0 = vld1q_f32(xp + (off));                                      \
                    float32x4_t x1 = vld1q_f32(xp + (off) + 4);                                  \
                    float32x4_t x2 = vld1q_f32(xp + (off) + 8);                                  \
                    float32x4_t x3 = vld1q_f32(xp + (off) + 12);                                 \
                    (vdot) = vfmaq_f32((vdot), wf0, x0);                                         \
                    (vdot) = vfmaq_f32((vdot), wf1, x1);                                         \
                    (vdot) = vfmaq_f32((vdot), wf2, x2);                                         \
                    (vdot) = vfmaq_f32((vdot), wf3, x3);                                         \
                    (vsum) = vaddq_f32((vsum), vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3))); \
                } while (0)
            TQ_Q4K_ACC(lo_a, j +  0, vdot1, vsum1);
            TQ_Q4K_ACC(lo_b, j + 16, vdot1, vsum1);
            TQ_Q4K_ACC(hi_a, j + 32, vdot2, vsum2);
            TQ_Q4K_ACC(hi_b, j + 48, vdot2, vsum2);
            #undef TQ_Q4K_ACC

            float dot1_s = vaddvq_f32(vdot1);
            float sum1_s = vaddvq_f32(vsum1);
            float dot2_s = vaddvq_f32(vdot2);
            float sum2_s = vaddvq_f32(vsum2);
            sum += d1 * dot1_s - m1 * sum1_s;
            sum += d2 * dot2_s - m2 * sum2_s;

            q += 32;
            is += 2;
        }
#else
        /* Scalar fallback (non-ARM) */
        for (int j = 0; j < 256; j += 64) {
            const float d1 = d * sc[is + 0];
            const float m1 = dmin * mn[is + 0];
            const float d2 = d * sc[is + 1];
            const float m2 = dmin * mn[is + 1];

            float dot1 = 0.0f, sum_x1 = 0.0f;
            for (int l = 0; l < 32; l++) {
                dot1  += (float)(q[l] & 0x0F) * xp[j + l];
                sum_x1 += xp[j + l];
            }
            sum += d1 * dot1 - m1 * sum_x1;

            float dot2 = 0.0f, sum_x2 = 0.0f;
            for (int l = 0; l < 32; l++) {
                dot2  += (float)(q[l] >> 4) * xp[j + 32 + l];
                sum_x2 += xp[j + 32 + l];
            }
            sum += d2 * dot2 - m2 * sum_x2;

            q += 32;
            is += 2;
        }
#endif
    }
    return sum;
}

/* Fused Q3_K dot product: 110 bytes per 256 elements
 * 3-bit = 2 low bits (qs) + 1 high bit (hmask)
 * 16 sub-blocks with 6-bit scales packed into 12 bytes */
/* ============================================================
 * Q3_K × int8 fused dot (NEON vdotq_s32 path)
 *
 * Q3_K layout (110 bytes per 256 elements):
 *   hmask[32]: 1 high bit per 3-bit value
 *   qs[64]:    2 low bits per 3-bit value, 4 per byte
 *   scales[12]: 16 × 6-bit sub-block scales (packed)
 *   d:         fp16 super-block scale
 *
 * Per-element value = ((qs >> shift) & 3) - (hmask_bit ? 0 : 4) ∈ [-4..3].
 * Same pattern as Q6_K int8 path — pre-quantize x, vdotq_s32 per 16-elem
 * sub-block. 16 vdotq_s32 per 256-element Q3_K block.
 * ============================================================ */
#if TQ_HAS_NEON
/* Shift macros — shift must be compile-time literal for vshrq_n_u8. */
#define Q3K_2BIT_S0(qv) vandq_u8((qv), vdupq_n_u8(0x03))
#define Q3K_2BIT_S2(qv) vandq_u8(vshrq_n_u8((qv), 2), vdupq_n_u8(0x03))
#define Q3K_2BIT_S4(qv) vandq_u8(vshrq_n_u8((qv), 4), vdupq_n_u8(0x03))
#define Q3K_2BIT_S6(qv) vandq_u8(vshrq_n_u8((qv), 6), vdupq_n_u8(0x03))

/* Build signed 3-bit weight vector: val = 2bit - (hm_set ? 0 : 4). */
static inline int8x16_t q3k_build16(uint8x16_t v2bit, uint8x16_t hm, uint8_t m) {
    uint8x16_t hm_test = vtstq_u8(hm, vdupq_n_u8(m));  /* 0xFF if set, 0 else */
    /* sub = 4 if hm_test==0, 0 if hm_test==0xFF  →  vbicq_u8(4, hm_test) */
    uint8x16_t sub = vbicq_u8(vdupq_n_u8(4), hm_test);
    return vsubq_s8(vreinterpretq_s8_u8(v2bit), vreinterpretq_s8_u8(sub));
}

static float fused_dot_q3_k_int8(const void* row, const int8_t* x_qs,
                                   const float* x_ds, int n)
{
    const int nb = n / 256;
    const block_q3_K* blk = (const block_q3_K*)row;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) {
            __builtin_prefetch((const uint8_t*)&blk[b+1], 0, 3);
            __builtin_prefetch((const uint8_t*)&blk[b+1] + 64, 0, 3);
        }
        const float d_all = fp16_to_fp32(blk[b].d);

        /* Decode 16 × 6-bit scales (same as scalar path) */
        memcpy(aux, blk[b].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        const uint8_t* q = blk[b].qs;
        const uint8_t* hm = blk[b].hmask;
        const int8_t* xb = x_qs + b * 256;
        const float*  xd = x_ds + b * 8;
        int is = 0;
        float block_sum = 0.0f;

        /* Each half = 128 elements. Half 0 uses qs[0..31]; half 1 uses qs[32..63].
         * Within a half: 4 shift values (0,2,4,6) × 2 sub-blocks each = 8 sub-blocks.
         * m bit mask advances by 1 each shift iteration. */
        for (int half = 0; half < 2; half++) {
            const uint8_t* qh = q + half * 32;
            uint8x16_t q_a = vld1q_u8(qh + 0);   /* qs[0..15] */
            uint8x16_t q_b = vld1q_u8(qh + 16);  /* qs[16..31] */
            /* hmask is shared across halves — m selects bit position */

            uint8x16_t hm_a = vld1q_u8(hm + 0);
            uint8x16_t hm_b = vld1q_u8(hm + 16);

            /* 8 sub-blocks per half: (shift, j) pairs. m starts at 1<<(half*4). */
            uint8_t m_base = (uint8_t)(1 << (half * 4));

            /* shift=0, m=m_base */
            int8x16_t w0a = q3k_build16(Q3K_2BIT_S0(q_a), hm_a, m_base << 0);
            int8x16_t w0b = q3k_build16(Q3K_2BIT_S0(q_b), hm_b, m_base << 0);
            /* shift=2, m=m_base<<1 */
            int8x16_t w1a = q3k_build16(Q3K_2BIT_S2(q_a), hm_a, m_base << 1);
            int8x16_t w1b = q3k_build16(Q3K_2BIT_S2(q_b), hm_b, m_base << 1);
            /* shift=4, m=m_base<<2 */
            int8x16_t w2a = q3k_build16(Q3K_2BIT_S4(q_a), hm_a, m_base << 2);
            int8x16_t w2b = q3k_build16(Q3K_2BIT_S4(q_b), hm_b, m_base << 2);
            /* shift=6, m=m_base<<3 */
            int8x16_t w3a = q3k_build16(Q3K_2BIT_S6(q_a), hm_a, m_base << 3);
            int8x16_t w3b = q3k_build16(Q3K_2BIT_S6(q_b), hm_b, m_base << 3);

            /* Load x for this half (128 elements = 8 int8x16 chunks).
             * Each x_q8 block is 32 elements, so 4 blocks per half.
             * Sub-block layout within half:
             *   pair (w0a,w0b) → x[0..31]   (x_ds[half*4+0])
             *   pair (w1a,w1b) → x[32..63]  (x_ds[half*4+1])
             *   pair (w2a,w2b) → x[64..95]  (x_ds[half*4+2])
             *   pair (w3a,w3b) → x[96..127] (x_ds[half*4+3])
             */
            const int8_t* xh = xb + half * 128;

#ifdef __ARM_FEATURE_DOTPROD
            int32_t i0a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w0a, vld1q_s8(xh +   0)));
            int32_t i0b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w0b, vld1q_s8(xh +  16)));
            int32_t i1a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w1a, vld1q_s8(xh +  32)));
            int32_t i1b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w1b, vld1q_s8(xh +  48)));
            int32_t i2a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w2a, vld1q_s8(xh +  64)));
            int32_t i2b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w2b, vld1q_s8(xh +  80)));
            int32_t i3a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w3a, vld1q_s8(xh +  96)));
            int32_t i3b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w3b, vld1q_s8(xh + 112)));
#else
            #define DOT16_Q3K(w, xp) vaddvq_s32(vpaddlq_s16(vaddq_s16( \
                vmull_s8(vget_low_s8(w), vget_low_s8(xp)), \
                vmull_s8(vget_high_s8(w), vget_high_s8(xp)))))
            int32_t i0a = DOT16_Q3K(w0a, vld1q_s8(xh +   0));
            int32_t i0b = DOT16_Q3K(w0b, vld1q_s8(xh +  16));
            int32_t i1a = DOT16_Q3K(w1a, vld1q_s8(xh +  32));
            int32_t i1b = DOT16_Q3K(w1b, vld1q_s8(xh +  48));
            int32_t i2a = DOT16_Q3K(w2a, vld1q_s8(xh +  64));
            int32_t i2b = DOT16_Q3K(w2b, vld1q_s8(xh +  80));
            int32_t i3a = DOT16_Q3K(w3a, vld1q_s8(xh +  96));
            int32_t i3b = DOT16_Q3K(w3b, vld1q_s8(xh + 112));
            #undef DOT16_Q3K
#endif
            /* Scale each sub-block (2 per x-block pair). */
            float s_xd0 = xd[half*4 + 0];
            float s_xd1 = xd[half*4 + 1];
            float s_xd2 = xd[half*4 + 2];
            float s_xd3 = xd[half*4 + 3];

            block_sum += s_xd0 * ((float)(scales[is  ] - 32) * (float)i0a + (float)(scales[is+1] - 32) * (float)i0b);
            block_sum += s_xd1 * ((float)(scales[is+2] - 32) * (float)i1a + (float)(scales[is+3] - 32) * (float)i1b);
            block_sum += s_xd2 * ((float)(scales[is+4] - 32) * (float)i2a + (float)(scales[is+5] - 32) * (float)i2b);
            block_sum += s_xd3 * ((float)(scales[is+6] - 32) * (float)i3a + (float)(scales[is+7] - 32) * (float)i3b);
            is += 8;
        }

        sumf += d_all * block_sum;
    }

    return sumf;
}

typedef struct {
    float* out; const void* weight; const int8_t* x_qs;
    const float* x_ds; size_t row_bytes; int in_dim;
    int start_row; int end_row;
} q3k_int_task_t;

static void* q3_k_int_dot_worker(void* arg) {
    q3k_int_task_t* task = (q3k_int_task_t*)arg;
    for (int d = task->start_row; d < task->end_row; d++) {
        const void* row = (const uint8_t*)task->weight + (size_t)d * task->row_bytes;
        task->out[d] = fused_dot_q3_k_int8(row, task->x_qs, task->x_ds, task->in_dim);
    }
    return NULL;
}

/* Q3_K batched int8 dot: mirrors fused_dot_iq3_xxs_int8_batched shape.
 * Amortizes 2-bit + hmask unpack + sub-scale decode across N activations. */
static void fused_dot_q3_k_int8_batched(
    float* out,              /* [N, n_rows] row-major */
    const void* weight,      /* block_q3_K array, n_rows × n_super × 110 bytes */
    size_t row_bytes,        /* = n_super × sizeof(block_q3_K) = n_super × 110 */
    int out_row_stride_n,    /* = n_rows */
    const int8_t* X_qs,      /* [N, n_super × 256] int8 */
    const float* X_ds,       /* [N, n_super × 8] fp32 scales (per-32 quant groups) */
    int start_row, int end_row,
    int n_super, int N)
{
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int d = start_row; d < end_row; d++) {
        const block_q3_K* blk = (const block_q3_K*)((const uint8_t*)weight + (size_t)d * row_bytes);

        float acc[64];
        if (N > 64) return;
        memset(acc, 0, (size_t)N * sizeof(float));

        for (int b = 0; b < n_super; b++) {
            if (b + 1 < n_super) {
                __builtin_prefetch((const uint8_t*)&blk[b+1], 0, 3);
                __builtin_prefetch((const uint8_t*)&blk[b+1] + 64, 0, 3);
            }
            const float d_all = fp16_to_fp32(blk[b].d);

            /* Decode 16 × 6-bit sub-block scales */
            memcpy(aux, blk[b].scales, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            const uint8_t* q = blk[b].qs;
            const uint8_t* hm = blk[b].hmask;
            int is = 0;

            for (int half = 0; half < 2; half++) {
                const uint8_t* qh = q + half * 32;
                uint8x16_t q_a = vld1q_u8(qh + 0);
                uint8x16_t q_b = vld1q_u8(qh + 16);
                uint8x16_t hm_a = vld1q_u8(hm + 0);
                uint8x16_t hm_b = vld1q_u8(hm + 16);
                uint8_t m_base = (uint8_t)(1 << (half * 4));

                /* Unpack 8 sub-blocks of 16 signed int8 weights each. N-invariant. */
                int8x16_t w0a = q3k_build16(Q3K_2BIT_S0(q_a), hm_a, m_base << 0);
                int8x16_t w0b = q3k_build16(Q3K_2BIT_S0(q_b), hm_b, m_base << 0);
                int8x16_t w1a = q3k_build16(Q3K_2BIT_S2(q_a), hm_a, m_base << 1);
                int8x16_t w1b = q3k_build16(Q3K_2BIT_S2(q_b), hm_b, m_base << 1);
                int8x16_t w2a = q3k_build16(Q3K_2BIT_S4(q_a), hm_a, m_base << 2);
                int8x16_t w2b = q3k_build16(Q3K_2BIT_S4(q_b), hm_b, m_base << 2);
                int8x16_t w3a = q3k_build16(Q3K_2BIT_S6(q_a), hm_a, m_base << 3);
                int8x16_t w3b = q3k_build16(Q3K_2BIT_S6(q_b), hm_b, m_base << 3);

                /* Pre-combine sub-block scales (4 pairs per half) */
                float s0 = (float)(scales[is  ] - 32);
                float s1 = (float)(scales[is+1] - 32);
                float s2 = (float)(scales[is+2] - 32);
                float s3 = (float)(scales[is+3] - 32);
                float s4 = (float)(scales[is+4] - 32);
                float s5 = (float)(scales[is+5] - 32);
                float s6 = (float)(scales[is+6] - 32);
                float s7 = (float)(scales[is+7] - 32);

                /* Inner batch loop — reuses unpacked weight vectors across N. */
                for (int n = 0; n < N; n++) {
                    const int8_t* xh = X_qs + (size_t)n * (n_super * 256)
                                             + b * 256 + half * 128;
                    const float*  xd = X_ds + (size_t)n * (n_super * 8) + b * 8;

                    int8x16_t xv0a = vld1q_s8(xh +   0);
                    int8x16_t xv0b = vld1q_s8(xh +  16);
                    int8x16_t xv1a = vld1q_s8(xh +  32);
                    int8x16_t xv1b = vld1q_s8(xh +  48);
                    int8x16_t xv2a = vld1q_s8(xh +  64);
                    int8x16_t xv2b = vld1q_s8(xh +  80);
                    int8x16_t xv3a = vld1q_s8(xh +  96);
                    int8x16_t xv3b = vld1q_s8(xh + 112);

#ifdef __ARM_FEATURE_DOTPROD
                    int32_t i0a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w0a, xv0a));
                    int32_t i0b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w0b, xv0b));
                    int32_t i1a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w1a, xv1a));
                    int32_t i1b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w1b, xv1b));
                    int32_t i2a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w2a, xv2a));
                    int32_t i2b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w2b, xv2b));
                    int32_t i3a = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w3a, xv3a));
                    int32_t i3b = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w3b, xv3b));
#else
                    #define DOT16_Q3K(w, xp) vaddvq_s32(vpaddlq_s16(vaddq_s16( \
                        vmull_s8(vget_low_s8(w), vget_low_s8(xp)), \
                        vmull_s8(vget_high_s8(w), vget_high_s8(xp)))))
                    int32_t i0a = DOT16_Q3K(w0a, xv0a);
                    int32_t i0b = DOT16_Q3K(w0b, xv0b);
                    int32_t i1a = DOT16_Q3K(w1a, xv1a);
                    int32_t i1b = DOT16_Q3K(w1b, xv1b);
                    int32_t i2a = DOT16_Q3K(w2a, xv2a);
                    int32_t i2b = DOT16_Q3K(w2b, xv2b);
                    int32_t i3a = DOT16_Q3K(w3a, xv3a);
                    int32_t i3b = DOT16_Q3K(w3b, xv3b);
                    #undef DOT16_Q3K
#endif
                    float block_sum = xd[half*4 + 0] * (s0 * (float)i0a + s1 * (float)i0b)
                                    + xd[half*4 + 1] * (s2 * (float)i1a + s3 * (float)i1b)
                                    + xd[half*4 + 2] * (s4 * (float)i2a + s5 * (float)i2b)
                                    + xd[half*4 + 3] * (s6 * (float)i3a + s7 * (float)i3b);
                    acc[n] += d_all * block_sum;
                }
                is += 8;
            }
        }

        for (int n = 0; n < N; n++) {
            out[(size_t)n * out_row_stride_n + d] = acc[n];
        }
    }
}
#endif /* TQ_HAS_NEON */

static float fused_dot_q3_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q3_K* blk = (const block_q3_K*)row;
    float sum = 0.0f;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int b = 0; b < nb; b++) {
        const float d_all = fp16_to_fp32(blk[b].d);

        const uint8_t* q  = blk[b].qs;
        const uint8_t* hm = blk[b].hmask;
        uint8_t m = 1;

        /* Decode 16 x 6-bit scales (same as dequant_q3_k) */
        memcpy(aux, blk[b].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        const float* xp = x + b * 256;
        int yi = 0;

        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                float dot = 0.0f;
                for (int l = 0; l < 16; ++l) {
                    dot += xp[yi + l] * (float)((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }
                sum += dl * dot;
                yi += 16;

                dl = d_all * (scales[is++] - 32);
                dot = 0.0f;
                for (int l = 0; l < 16; ++l) {
                    dot += xp[yi + l] * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                sum += dl * dot;
                yi += 16;

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
    return sum;
}

/* Fused Q4_0 dot product: 18 bytes per 32 elements */
static float fused_dot_q4_0(const void* row, const float* x, int n) {
    const int nb = n / 32;
    const block_q4_0* blk = (const block_q4_0*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xp = x + b * 32;

        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blk[b].qs[j];
            block_sum += (float)((int)(byte & 0x0F) - 8) * xp[j];
            block_sum += (float)((int)(byte >> 4) - 8) * xp[j + 16];
        }
        sum += d * block_sum;
    }
    return sum;
}

/* ============================================================
 * Q6_K × int8 fused dot (NEON vdotq_s32 path)
 *
 * The float-FMA fused_dot_q6_k below is pure scalar — the dominant
 * decode-time function on Q4_K_M models (which embed Q6_K for the
 * critical `attention.wo` and `ffn_down` projections). Sample on
 * Qwen3.5-4B Q4_K_M shows fused_dot_q6_k = 17725 self-time vs
 * matmul_q4_rows = 3040, leaving us at ~25% of llama.cpp CPU speed.
 *
 * Layout per 256-element block:
 *   ql[128]: low 4 bits of 256 6-bit weights
 *   qh[64]:  upper 2 bits of 256 6-bit weights
 *   sc[16]:  int8 sub-block scales (one per 16 elements)
 *   d:       fp16 super-block scale
 *
 * Per-element value = ((ql & 0xF) | ((qh & 3) << 4)) - 32, scaled.
 *
 * Strategy: pre-quantize x to int8 (Q8_0 layout, 32-elem blocks),
 * for each 16-element sub-block do one vdotq_s32 between unpacked
 * Q6_K weights (int8 [-32..31]) and x_q8.
 * ============================================================ */
#if TQ_HAS_NEON
/* Q6_K unpack helpers — QH_SHIFT must be a compile-time literal (0/2/4/6).
 * vshrq_n_u8 requires a literal shift; we open-code per shift to keep
 * everything as constants the compiler can fold. */
#define Q6K_LOW_S0(ql_v, qh_v) \
    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8( \
        vandq_u8((ql_v), vdupq_n_u8(0x0F)), \
        vshlq_n_u8(vandq_u8((qh_v), vdupq_n_u8(0x03)), 4))), \
        vdupq_n_s8(32))
#define Q6K_LOW_S2(ql_v, qh_v) \
    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8( \
        vandq_u8((ql_v), vdupq_n_u8(0x0F)), \
        vshlq_n_u8(vandq_u8(vshrq_n_u8((qh_v), 2), vdupq_n_u8(0x03)), 4))), \
        vdupq_n_s8(32))
#define Q6K_HIGH_S4(ql_v, qh_v) \
    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8( \
        vshrq_n_u8((ql_v), 4), \
        vshlq_n_u8(vandq_u8(vshrq_n_u8((qh_v), 4), vdupq_n_u8(0x03)), 4))), \
        vdupq_n_s8(32))
#define Q6K_HIGH_S6(ql_v, qh_v) \
    vsubq_s8(vreinterpretq_s8_u8(vorrq_u8( \
        vshrq_n_u8((ql_v), 4), \
        vshlq_n_u8(vandq_u8(vshrq_n_u8((qh_v), 6), vdupq_n_u8(0x03)), 4))), \
        vdupq_n_s8(32))

static float fused_dot_q6_k_int8(const void* row, const int8_t* x_qs,
                                   const float* x_ds, int n)
{
    const int nb = n / 256;
    const block_q6_K* blk = (const block_q6_K*)row;

    float sumf = 0.0f;

    for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) {
            __builtin_prefetch((const uint8_t*)&blk[b+1], 0, 3);
            __builtin_prefetch((const uint8_t*)&blk[b+1] + 128, 0, 3);
        }
        const float d = fp16_to_fp32(blk[b].d);
        const uint8_t* ql = blk[b].ql;
        const uint8_t* qh = blk[b].qh;
        const int8_t* sc = blk[b].scales;

        const int8_t* xb = x_qs + b * 256;
        const float*  xd = x_ds + b * 8;

        float sub_sum = 0.0f;

        /* Process two halves of 128 elements each */
        for (int half = 0; half < 2; half++) {
            /* Load ql (64 bytes) and qh (32 bytes) for this half */
            uint8x16_t ql_a = vld1q_u8(ql + half*64 + 0);   /* ql[0..15] */
            uint8x16_t ql_b = vld1q_u8(ql + half*64 + 16);  /* ql[16..31] */
            uint8x16_t ql_c = vld1q_u8(ql + half*64 + 32);  /* ql[32..47] */
            uint8x16_t ql_d = vld1q_u8(ql + half*64 + 48);  /* ql[48..63] */
            uint8x16_t qh_a = vld1q_u8(qh + half*32 + 0);   /* qh[0..15] */
            uint8x16_t qh_b = vld1q_u8(qh + half*32 + 16);  /* qh[16..31] */
            const int8_t* schalf = sc + half*8;

            /* Sub-blocks 0,1: x[0..31] within half ↔ qh shift 0 (low 2 bits)
             * sc[0] for x[0..15], sc[1] for x[16..31]. Both use ql_a,ql_b. */
            int8x16_t w0 = Q6K_LOW_S0(ql_a, qh_a);  /* ql[0..15] & 0xF */
            int8x16_t w1 = Q6K_LOW_S0(ql_b, qh_b);  /* ql[16..31] & 0xF */
            /* Sub-blocks 2,3: x[32..63] ↔ qh shift 2
             * sc[2] for x[32..47], sc[3] for x[48..63]. ql_c,ql_d low. */
            int8x16_t w2 = Q6K_LOW_S2(ql_c, qh_a);
            int8x16_t w3 = Q6K_LOW_S2(ql_d, qh_b);
            /* Sub-blocks 4,5: x[64..95] ↔ ql high nibble + qh shift 4
             * sc[4] for x[64..79], sc[5] for x[80..95]. */
            int8x16_t w4 = Q6K_HIGH_S4(ql_a, qh_a);
            int8x16_t w5 = Q6K_HIGH_S4(ql_b, qh_b);
            /* Sub-blocks 6,7: x[96..127] ↔ ql high + qh shift 6
             * sc[6] for x[96..111], sc[7] for x[112..127]. */
            int8x16_t w6 = Q6K_HIGH_S6(ql_c, qh_a);
            int8x16_t w7 = Q6K_HIGH_S6(ql_d, qh_b);

            /* For each sub-block: vdotq_s32 against corresponding x_q8 chunk.
             * Note: each pair of 16-elem sub-blocks shares one x_scale because
             * x is quantized in 32-elem blocks. Sub-blocks 0,1 share xd[half*4+0],
             * etc. */
            const int8_t* xh = xb + half * 128;

            int8x16_t xv0 = vld1q_s8(xh + 0);
            int8x16_t xv1 = vld1q_s8(xh + 16);
            int8x16_t xv2 = vld1q_s8(xh + 32);
            int8x16_t xv3 = vld1q_s8(xh + 48);
            int8x16_t xv4 = vld1q_s8(xh + 64);
            int8x16_t xv5 = vld1q_s8(xh + 80);
            int8x16_t xv6 = vld1q_s8(xh + 96);
            int8x16_t xv7 = vld1q_s8(xh + 112);

#ifdef __ARM_FEATURE_DOTPROD
            int32_t i0 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w0, xv0));
            int32_t i1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w1, xv1));
            int32_t i2 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w2, xv2));
            int32_t i3 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w3, xv3));
            int32_t i4 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w4, xv4));
            int32_t i5 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w5, xv5));
            int32_t i6 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w6, xv6));
            int32_t i7 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w7, xv7));
#else
            /* vmull_s8/vpadd fallback for non-DOTPROD CPUs */
            #define DOT16(w, x) vaddvq_s32(vpaddlq_s16(vaddq_s16( \
                vmull_s8(vget_low_s8(w), vget_low_s8(x)), \
                vmull_s8(vget_high_s8(w), vget_high_s8(x)))))
            int32_t i0 = DOT16(w0, xv0);
            int32_t i1 = DOT16(w1, xv1);
            int32_t i2 = DOT16(w2, xv2);
            int32_t i3 = DOT16(w3, xv3);
            int32_t i4 = DOT16(w4, xv4);
            int32_t i5 = DOT16(w5, xv5);
            int32_t i6 = DOT16(w6, xv6);
            int32_t i7 = DOT16(w7, xv7);
            #undef DOT16
#endif
            /* Combine with weight scales (int8) and x_scales (per 32-elem) */
            float xd0 = xd[half*4 + 0];
            float xd1 = xd[half*4 + 1];
            float xd2 = xd[half*4 + 2];
            float xd3 = xd[half*4 + 3];

            sub_sum += xd0 * ((float)schalf[0] * (float)i0 + (float)schalf[1] * (float)i1);
            sub_sum += xd1 * ((float)schalf[2] * (float)i2 + (float)schalf[3] * (float)i3);
            sub_sum += xd2 * ((float)schalf[4] * (float)i4 + (float)schalf[5] * (float)i5);
            sub_sum += xd3 * ((float)schalf[6] * (float)i6 + (float)schalf[7] * (float)i7);
        }

        sumf += d * sub_sum;
    }

    return sumf;
}

typedef struct {
    float* out; const void* weight; const int8_t* x_qs;
    const float* x_ds; size_t row_bytes; int in_dim;
    int start_row; int end_row;
} q6k_int_task_t;

static void* q6_k_int_dot_worker(void* arg) {
    q6k_int_task_t* task = (q6k_int_task_t*)arg;
    for (int d = task->start_row; d < task->end_row; d++) {
        const void* row = (const uint8_t*)task->weight + (size_t)d * task->row_bytes;
        task->out[d] = fused_dot_q6_k_int8(row, task->x_qs, task->x_ds, task->in_dim);
    }
    return NULL;
}
#endif /* TQ_HAS_NEON */

/* Fused Q6_K dot product: 210 bytes per 256 elements
 * Matches ggml dequantize_row_q6_K layout exactly:
 * Two 128-element halves, each with 32 iterations producing 4 elements. */
static float fused_dot_q6_k(const void* row, const float* x, int n) {
    const int nb = n / 256;
    const block_q6_K* blk = (const block_q6_K*)row;
    float sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const float d = fp16_to_fp32(blk[b].d);
        const float* xbase = x + b * 256;
        const uint8_t* ql = blk[b].ql;
        const uint8_t* qh = blk[b].qh;
        const int8_t* sc = blk[b].scales;

        for (int half = 0; half < 2; half++) {
            const float* xp = xbase + half * 128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * sc[is + 0] * q1 * xp[l +  0];
                sum += d * sc[is + 2] * q2 * xp[l + 32];
                sum += d * sc[is + 4] * q3 * xp[l + 64];
                sum += d * sc[is + 6] * q4 * xp[l + 96];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return sum;
}

/* ============================================================
 * On-the-fly dequant matmul (with fused fast paths)
 *
 * out[d] = sum_n( x[n] * dequant(W[d, n]) )
 *
 * W is stored row-major in quantized blocks.
 * Hot path for MoE expert computation.
 *
 * For supported types (IQ2_XXS, IQ2_S, Q8_0, Q4_K, Q4_0, Q6_K,
 * IQ4_NL, IQ4_XS), we use fused dequant-dot that avoids writing
 * intermediate FP32 values to memory. This eliminates ~3 GB/token
 * of temporary memory traffic for IQ2_XXS MoE models.
 * ============================================================ */

/* ============================================================
 * Multi-threaded GGUF matmul worker
 * ============================================================ */
typedef struct {
    float*       out;
    const float* x;
    const void*  weight;
    float (*fused_dot)(const void*, const float*, int);
    tq_ggml_dtype weight_type;
    size_t       row_bytes;
    int          in_dim;
    int          block_bytes;
    int          block_elems;
    int          n_blocks;
    int          start_row;
    int          end_row;
} gguf_matmul_task_t;

static void* gguf_matmul_worker(void* arg) {
    gguf_matmul_task_t* t = (gguf_matmul_task_t*)arg;

    if (t->fused_dot) {
        for (int d = t->start_row; d < t->end_row; d++) {
            const uint8_t* row = (const uint8_t*)t->weight + (size_t)d * t->row_bytes;
            t->out[d] = t->fused_dot(row, t->x, t->in_dim);
        }
        return NULL;
    }

    /* Generic fallback: dequant block -> tmp -> dot */
    for (int d = t->start_row; d < t->end_row; d++) {
        const uint8_t* row = (const uint8_t*)t->weight + (size_t)d * t->row_bytes;
        float sum = 0.0f;
        float tmp[256]; /* max block size is 256 */

        for (int b = 0; b < t->n_blocks; b++) {
            tq_dequant_row_gguf(t->weight_type,
                                row + (size_t)b * t->block_bytes,
                                tmp, t->block_elems);

            const float* xp = t->x + b * t->block_elems;

#if TQ_HAS_NEON
            float32x4_t vsum0 = vdupq_n_f32(0.0f);
            float32x4_t vsum1 = vdupq_n_f32(0.0f);
            float32x4_t vsum2 = vdupq_n_f32(0.0f);
            float32x4_t vsum3 = vdupq_n_f32(0.0f);

            int j = 0;
            for (; j + 15 < t->block_elems; j += 16) {
                float32x4_t vx0 = vld1q_f32(xp + j);
                float32x4_t vx1 = vld1q_f32(xp + j + 4);
                float32x4_t vx2 = vld1q_f32(xp + j + 8);
                float32x4_t vx3 = vld1q_f32(xp + j + 12);
                float32x4_t vt0 = vld1q_f32(tmp + j);
                float32x4_t vt1 = vld1q_f32(tmp + j + 4);
                float32x4_t vt2 = vld1q_f32(tmp + j + 8);
                float32x4_t vt3 = vld1q_f32(tmp + j + 12);
                vsum0 = vfmaq_f32(vsum0, vx0, vt0);
                vsum1 = vfmaq_f32(vsum1, vx1, vt1);
                vsum2 = vfmaq_f32(vsum2, vx2, vt2);
                vsum3 = vfmaq_f32(vsum3, vx3, vt3);
            }
            for (; j + 3 < t->block_elems; j += 4) {
                float32x4_t vx0 = vld1q_f32(xp + j);
                float32x4_t vt0 = vld1q_f32(tmp + j);
                vsum0 = vfmaq_f32(vsum0, vx0, vt0);
            }

            vsum0 = vaddq_f32(vsum0, vsum1);
            vsum2 = vaddq_f32(vsum2, vsum3);
            vsum0 = vaddq_f32(vsum0, vsum2);
            sum += vaddvq_f32(vsum0);

            for (; j < t->block_elems; j++) {
                sum += xp[j] * tmp[j];
            }
#else
            for (int j = 0; j < t->block_elems; j++) {
                sum += xp[j] * tmp[j];
            }
#endif
        }

        t->out[d] = sum;
    }
    return NULL;
}

/* Pre-quantize input vector to Q8 format for int8×int8 matmul.
 * Called once in transformer, result reused for Q/K/V/O projections.
 * Stores int8 values in qs[n], per-block scales in ds[n/32]. */
void tq_preq_input_q8(const float* x, int8_t* qs, float* ds, int n) {
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        const float* xp = x + b * 32;
        float amax = 0.0f;
#if TQ_HAS_NEON
        float32x4_t vmax = vdupq_n_f32(0.0f);
        for (int j = 0; j < 32; j += 4) {
            float32x4_t vx = vld1q_f32(xp + j);
            vmax = vmaxq_f32(vmax, vabsq_f32(vx));
        }
        amax = vmaxvq_f32(vmax);
#else
        for (int j = 0; j < 32; j++) {
            float a = xp[j] < 0 ? -xp[j] : xp[j];
            if (a > amax) amax = a;
        }
#endif
        float d = amax / 127.0f;
        ds[b] = d;
        if (d > 0.0f) {
            float id = 127.0f / amax;
#if TQ_HAS_NEON
            float32x4_t vid = vdupq_n_f32(id);
            for (int j = 0; j < 32; j += 8) {
                float32x4_t v0 = vmulq_f32(vld1q_f32(xp + j), vid);
                float32x4_t v1 = vmulq_f32(vld1q_f32(xp + j + 4), vid);
                int32x4_t i0 = vcvtnq_s32_f32(v0);
                int32x4_t i1 = vcvtnq_s32_f32(v1);
                int16x4_t s0 = vqmovn_s32(i0);
                int16x4_t s1 = vqmovn_s32(i1);
                int8x8_t b8 = vqmovn_s16(vcombine_s16(s0, s1));
                vst1_s8(qs + b * 32 + j, b8);
            }
#else
            for (int j = 0; j < 32; j++) {
                int v = (int)roundf(xp[j] * id);
                qs[b * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
            }
#endif
        } else {
            memset(qs + b * 32, 0, 32);
        }
    }
}

/* Thread-local pre-quantized input pointer (set by tq_matmul_gguf when available) */
#ifdef _MSC_VER
static __declspec(thread) const int8_t* g_preq_qs = NULL;
static __declspec(thread) const float*  g_preq_ds = NULL;
#else
static __thread const int8_t* g_preq_qs = NULL;
static __thread const float*  g_preq_ds = NULL;
#endif

void tq_set_preq(const int8_t* qs, const float* ds) {
    g_preq_qs = qs;
    g_preq_ds = ds;
}
void tq_clear_preq(void) {
    g_preq_qs = NULL;
    g_preq_ds = NULL;
}

/* Q8×Q8 integer dot worker — processes a range of output rows using int8 multiply-accumulate */
#if TQ_HAS_NEON
typedef struct {
    float* out; const void* weight; const int8_t* x_qs; const float* x_ds;
    size_t row_bytes; int n_blocks; int start_row; int end_row;
} q8_int_task_t;

/* v2: 2-block unroll + vector FMA accumulator.
 *
 * Prior kernel added to a scalar `row_sum` float each block — that's a 3-cycle
 * FMA latency chain roughly n_blocks deep. For Qwen3.6 with dim=2048 → 64
 * blocks per row → 192 cycles of latency per row on M1 P-core before the
 * result is available, so the actual throughput is latency-bound even though
 * vdotq_s32 itself can pipeline.
 *
 * Fix: keep the partial sum in a float32x4_t, accumulate via vfmaq_n_f32
 * (vector += vector × scalar). That lifts the dependency chain into a
 * 4-lane vector; lanes reduce to scalar only once at the end of the row.
 * Combined with 2-block unrolling (distinct vsum0a / vsum0b accumulators),
 * the per-row chain depth halves while issuing 2 vdotq per cycle.
 *
 * Env: TQ_Q8_V1=1 reverts to the original worker (for A/B). */
static void* q8_int_dot_worker_v2(void* arg) {
    q8_int_task_t* t = (q8_int_task_t*)arg;
    int d = t->start_row;

    for (; d + 1 < t->end_row; d += 2) {
        const block_q8_0* w0 = (const block_q8_0*)((const uint8_t*)t->weight + (size_t)d       * t->row_bytes);
        const block_q8_0* w1 = (const block_q8_0*)((const uint8_t*)t->weight + (size_t)(d + 1) * t->row_bytes);
        if (d + 2 < t->end_row) {
            const uint8_t* next = (const uint8_t*)t->weight + (size_t)(d + 2) * t->row_bytes;
            __builtin_prefetch(next +   0, 0, 0);
            __builtin_prefetch(next +  64, 0, 0);
            __builtin_prefetch(next + 128, 0, 0);
            __builtin_prefetch(next + 192, 0, 0);
        }

        /* Dual accumulators per row (block_even / block_odd) to break the
         * FMA dependency chain. Reduced to scalar at the end of the row. */
        float32x4_t vsum0a = vdupq_n_f32(0.0f);
        float32x4_t vsum0b = vdupq_n_f32(0.0f);
        float32x4_t vsum1a = vdupq_n_f32(0.0f);
        float32x4_t vsum1b = vdupq_n_f32(0.0f);

        int b = 0;
        for (; b + 1 < t->n_blocks; b += 2) {
            /* block b (even) */
            const int8_t* xqs_b = t->x_qs + b * 32;
            int8x16_t xq0_b = vld1q_s8(xqs_b +  0);
            int8x16_t xq1_b = vld1q_s8(xqs_b + 16);
#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vd0_b = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0[b].qs +  0), xq0_b);
            vd0_b = vdotq_s32(vd0_b, vld1q_s8(w0[b].qs + 16), xq1_b);
            int32x4_t vd1_b = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w1[b].qs +  0), xq0_b);
            vd1_b = vdotq_s32(vd1_b, vld1q_s8(w1[b].qs + 16), xq1_b);
#else
            int32x4_t vd0a = vdupq_n_s32(0), vd0bz = vdupq_n_s32(0);
            int32x4_t vd1a = vdupq_n_s32(0), vd1bz = vdupq_n_s32(0);
            int8x16_t vw0_a = vld1q_s8(w0[b].qs +  0);
            int8x16_t vw0_b = vld1q_s8(w0[b].qs + 16);
            int8x16_t vw1_a = vld1q_s8(w1[b].qs +  0);
            int8x16_t vw1_b = vld1q_s8(w1[b].qs + 16);
            vd0a = vpadalq_s16(vd0a, vmull_s8(vget_low_s8(vw0_a),  vget_low_s8(xq0_b)));
            vd0a = vpadalq_s16(vd0a, vmull_s8(vget_high_s8(vw0_a), vget_high_s8(xq0_b)));
            vd0bz = vpadalq_s16(vd0bz, vmull_s8(vget_low_s8(vw0_b),  vget_low_s8(xq1_b)));
            vd0bz = vpadalq_s16(vd0bz, vmull_s8(vget_high_s8(vw0_b), vget_high_s8(xq1_b)));
            vd1a = vpadalq_s16(vd1a, vmull_s8(vget_low_s8(vw1_a),  vget_low_s8(xq0_b)));
            vd1a = vpadalq_s16(vd1a, vmull_s8(vget_high_s8(vw1_a), vget_high_s8(xq0_b)));
            vd1bz = vpadalq_s16(vd1bz, vmull_s8(vget_low_s8(vw1_b),  vget_low_s8(xq1_b)));
            vd1bz = vpadalq_s16(vd1bz, vmull_s8(vget_high_s8(vw1_b), vget_high_s8(xq1_b)));
            int32x4_t vd0_b = vaddq_s32(vd0a, vd0bz);
            int32x4_t vd1_b = vaddq_s32(vd1a, vd1bz);
#endif
            float scale0_b = fp16_to_fp32(w0[b].d) * t->x_ds[b];
            float scale1_b = fp16_to_fp32(w1[b].d) * t->x_ds[b];
            vsum0a = vfmaq_n_f32(vsum0a, vcvtq_f32_s32(vd0_b), scale0_b);
            vsum1a = vfmaq_n_f32(vsum1a, vcvtq_f32_s32(vd1_b), scale1_b);

            /* block b+1 (odd) — independent accumulators */
            const int8_t* xqs_c = t->x_qs + (b + 1) * 32;
            int8x16_t xq0_c = vld1q_s8(xqs_c +  0);
            int8x16_t xq1_c = vld1q_s8(xqs_c + 16);
#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vd0_c = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0[b + 1].qs +  0), xq0_c);
            vd0_c = vdotq_s32(vd0_c, vld1q_s8(w0[b + 1].qs + 16), xq1_c);
            int32x4_t vd1_c = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w1[b + 1].qs +  0), xq0_c);
            vd1_c = vdotq_s32(vd1_c, vld1q_s8(w1[b + 1].qs + 16), xq1_c);
#else
            int32x4_t vd0ca = vdupq_n_s32(0), vd0cb = vdupq_n_s32(0);
            int32x4_t vd1ca = vdupq_n_s32(0), vd1cb = vdupq_n_s32(0);
            int8x16_t vw0c_a = vld1q_s8(w0[b + 1].qs +  0);
            int8x16_t vw0c_b = vld1q_s8(w0[b + 1].qs + 16);
            int8x16_t vw1c_a = vld1q_s8(w1[b + 1].qs +  0);
            int8x16_t vw1c_b = vld1q_s8(w1[b + 1].qs + 16);
            vd0ca = vpadalq_s16(vd0ca, vmull_s8(vget_low_s8(vw0c_a),  vget_low_s8(xq0_c)));
            vd0ca = vpadalq_s16(vd0ca, vmull_s8(vget_high_s8(vw0c_a), vget_high_s8(xq0_c)));
            vd0cb = vpadalq_s16(vd0cb, vmull_s8(vget_low_s8(vw0c_b),  vget_low_s8(xq1_c)));
            vd0cb = vpadalq_s16(vd0cb, vmull_s8(vget_high_s8(vw0c_b), vget_high_s8(xq1_c)));
            vd1ca = vpadalq_s16(vd1ca, vmull_s8(vget_low_s8(vw1c_a),  vget_low_s8(xq0_c)));
            vd1ca = vpadalq_s16(vd1ca, vmull_s8(vget_high_s8(vw1c_a), vget_high_s8(xq0_c)));
            vd1cb = vpadalq_s16(vd1cb, vmull_s8(vget_low_s8(vw1c_b),  vget_low_s8(xq1_c)));
            vd1cb = vpadalq_s16(vd1cb, vmull_s8(vget_high_s8(vw1c_b), vget_high_s8(xq1_c)));
            int32x4_t vd0_c = vaddq_s32(vd0ca, vd0cb);
            int32x4_t vd1_c = vaddq_s32(vd1ca, vd1cb);
#endif
            float scale0_c = fp16_to_fp32(w0[b + 1].d) * t->x_ds[b + 1];
            float scale1_c = fp16_to_fp32(w1[b + 1].d) * t->x_ds[b + 1];
            vsum0b = vfmaq_n_f32(vsum0b, vcvtq_f32_s32(vd0_c), scale0_c);
            vsum1b = vfmaq_n_f32(vsum1b, vcvtq_f32_s32(vd1_c), scale1_c);
        }
        /* Odd remaining block */
        for (; b < t->n_blocks; b++) {
            const int8_t* xqs = t->x_qs + b * 32;
            int8x16_t xq0 = vld1q_s8(xqs +  0);
            int8x16_t xq1 = vld1q_s8(xqs + 16);
#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vd0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w0[b].qs +  0), xq0);
            vd0 = vdotq_s32(vd0, vld1q_s8(w0[b].qs + 16), xq1);
            int32x4_t vd1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w1[b].qs +  0), xq0);
            vd1 = vdotq_s32(vd1, vld1q_s8(w1[b].qs + 16), xq1);
#else
            int32x4_t vd0a = vdupq_n_s32(0), vd0b = vdupq_n_s32(0);
            int32x4_t vd1a = vdupq_n_s32(0), vd1b = vdupq_n_s32(0);
            int8x16_t vw0_lo = vld1q_s8(w0[b].qs +  0);
            int8x16_t vw0_hi = vld1q_s8(w0[b].qs + 16);
            int8x16_t vw1_lo = vld1q_s8(w1[b].qs +  0);
            int8x16_t vw1_hi = vld1q_s8(w1[b].qs + 16);
            vd0a = vpadalq_s16(vd0a, vmull_s8(vget_low_s8(vw0_lo),  vget_low_s8(xq0)));
            vd0a = vpadalq_s16(vd0a, vmull_s8(vget_high_s8(vw0_lo), vget_high_s8(xq0)));
            vd0b = vpadalq_s16(vd0b, vmull_s8(vget_low_s8(vw0_hi),  vget_low_s8(xq1)));
            vd0b = vpadalq_s16(vd0b, vmull_s8(vget_high_s8(vw0_hi), vget_high_s8(xq1)));
            vd1a = vpadalq_s16(vd1a, vmull_s8(vget_low_s8(vw1_lo),  vget_low_s8(xq0)));
            vd1a = vpadalq_s16(vd1a, vmull_s8(vget_high_s8(vw1_lo), vget_high_s8(xq0)));
            vd1b = vpadalq_s16(vd1b, vmull_s8(vget_low_s8(vw1_hi),  vget_low_s8(xq1)));
            vd1b = vpadalq_s16(vd1b, vmull_s8(vget_high_s8(vw1_hi), vget_high_s8(xq1)));
            int32x4_t vd0 = vaddq_s32(vd0a, vd0b);
            int32x4_t vd1 = vaddq_s32(vd1a, vd1b);
#endif
            float scale0 = fp16_to_fp32(w0[b].d) * t->x_ds[b];
            float scale1 = fp16_to_fp32(w1[b].d) * t->x_ds[b];
            vsum0a = vfmaq_n_f32(vsum0a, vcvtq_f32_s32(vd0), scale0);
            vsum1a = vfmaq_n_f32(vsum1a, vcvtq_f32_s32(vd1), scale1);
        }
        /* Combine dual accumulators then horizontal reduce */
        t->out[d]     = vaddvq_f32(vaddq_f32(vsum0a, vsum0b));
        t->out[d + 1] = vaddvq_f32(vaddq_f32(vsum1a, vsum1b));
    }
    /* Tail: single row */
    for (; d < t->end_row; d++) {
        const block_q8_0* wblk = (const block_q8_0*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        float32x4_t vsum = vdupq_n_f32(0.0f);
        for (int b = 0; b < t->n_blocks; b++) {
            const int8_t* xqs = t->x_qs + b * 32;
#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vd = vdotq_s32(vdupq_n_s32(0), vld1q_s8(wblk[b].qs +  0), vld1q_s8(xqs +  0));
            vd = vdotq_s32(vd, vld1q_s8(wblk[b].qs + 16), vld1q_s8(xqs + 16));
#else
            int32x4_t vd0 = vdupq_n_s32(0), vd1 = vdupq_n_s32(0);
            for (int j = 0; j < 32; j += 16) {
                int8x16_t vw = vld1q_s8(wblk[b].qs + j);
                int8x16_t vx = vld1q_s8(xqs + j);
                vd0 = vpadalq_s16(vd0, vmull_s8(vget_low_s8(vw), vget_low_s8(vx)));
                vd1 = vpadalq_s16(vd1, vmull_s8(vget_high_s8(vw), vget_high_s8(vx)));
            }
            int32x4_t vd = vaddq_s32(vd0, vd1);
#endif
            float scale = fp16_to_fp32(wblk[b].d) * t->x_ds[b];
            vsum = vfmaq_n_f32(vsum, vcvtq_f32_s32(vd), scale);
        }
        t->out[d] = vaddvq_f32(vsum);
    }
    return NULL;
}

void* q8_int_dot_worker(void* arg) {
    /* v1 stays as the A/B fallback under TQ_Q8_V1. v2 is the new default. */
    if (!getenv("TQ_Q8_V1")) {
        return q8_int_dot_worker_v2(arg);
    }
    q8_int_task_t* t = (q8_int_task_t*)arg;
    int d = t->start_row;
    /* 2-row inner loop: pair rows share x_qs / x_ds. ILP hides weight load
     * latency. Same trick as the q4k_int worker. */
    for (; d + 1 < t->end_row; d += 2) {
        const block_q8_0* wblk0 = (const block_q8_0*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        const block_q8_0* wblk1 = (const block_q8_0*)((const uint8_t*)t->weight + (size_t)(d + 1) * t->row_bytes);
        if (d + 2 < t->end_row) {
            const uint8_t* next = (const uint8_t*)t->weight + (size_t)(d + 2) * t->row_bytes;
            __builtin_prefetch(next +   0, 0, 0);
            __builtin_prefetch(next +  64, 0, 0);
            __builtin_prefetch(next + 128, 0, 0);
            __builtin_prefetch(next + 192, 0, 0);
        }
        float row_sum0 = 0.0f, row_sum1 = 0.0f;
        for (int b = 0; b < t->n_blocks; b++) {
            const float wd0 = fp16_to_fp32(wblk0[b].d);
            const float wd1 = fp16_to_fp32(wblk1[b].d);
            const int8_t* wqs0 = wblk0[b].qs;
            const int8_t* wqs1 = wblk1[b].qs;
            const int8_t* xqs = t->x_qs + b * 32;
            int8x16_t xq0 = vld1q_s8(xqs +  0);
            int8x16_t xq1 = vld1q_s8(xqs + 16);
#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vd0 = vdupq_n_s32(0);
            int32x4_t vd1 = vdupq_n_s32(0);
            vd0 = vdotq_s32(vd0, vld1q_s8(wqs0 +  0), xq0);
            vd0 = vdotq_s32(vd0, vld1q_s8(wqs0 + 16), xq1);
            vd1 = vdotq_s32(vd1, vld1q_s8(wqs1 +  0), xq0);
            vd1 = vdotq_s32(vd1, vld1q_s8(wqs1 + 16), xq1);
            float xd = t->x_ds[b];
            row_sum0 += wd0 * xd * (float)vaddvq_s32(vd0);
            row_sum1 += wd1 * xd * (float)vaddvq_s32(vd1);
#else
            int32x4_t vd0a = vdupq_n_s32(0), vd0b = vdupq_n_s32(0);
            int32x4_t vd1a = vdupq_n_s32(0), vd1b = vdupq_n_s32(0);
            int8x16_t vw0a = vld1q_s8(wqs0);
            int8x16_t vw0b = vld1q_s8(wqs0 + 16);
            int8x16_t vw1a = vld1q_s8(wqs1);
            int8x16_t vw1b = vld1q_s8(wqs1 + 16);
            vd0a = vpadalq_s16(vd0a, vmull_s8(vget_low_s8(vw0a), vget_low_s8(xq0)));
            vd0a = vpadalq_s16(vd0a, vmull_s8(vget_high_s8(vw0a), vget_high_s8(xq0)));
            vd0b = vpadalq_s16(vd0b, vmull_s8(vget_low_s8(vw0b), vget_low_s8(xq1)));
            vd0b = vpadalq_s16(vd0b, vmull_s8(vget_high_s8(vw0b), vget_high_s8(xq1)));
            vd1a = vpadalq_s16(vd1a, vmull_s8(vget_low_s8(vw1a), vget_low_s8(xq0)));
            vd1a = vpadalq_s16(vd1a, vmull_s8(vget_high_s8(vw1a), vget_high_s8(xq0)));
            vd1b = vpadalq_s16(vd1b, vmull_s8(vget_low_s8(vw1b), vget_low_s8(xq1)));
            vd1b = vpadalq_s16(vd1b, vmull_s8(vget_high_s8(vw1b), vget_high_s8(xq1)));
            float xd = t->x_ds[b];
            row_sum0 += wd0 * xd * (float)vaddvq_s32(vaddq_s32(vd0a, vd0b));
            row_sum1 += wd1 * xd * (float)vaddvq_s32(vaddq_s32(vd1a, vd1b));
#endif
        }
        t->out[d] = row_sum0;
        t->out[d + 1] = row_sum1;
    }
    /* Tail: single row */
    for (; d < t->end_row; d++) {
        const block_q8_0* wblk = (const block_q8_0*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        float row_sum = 0.0f;
        for (int b = 0; b < t->n_blocks; b++) {
            const float wd = fp16_to_fp32(wblk[b].d);
            const int8_t* wqs = wblk[b].qs;
            const int8_t* xqs = t->x_qs + b * 32;
#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t vd = vdupq_n_s32(0);
            vd = vdotq_s32(vd, vld1q_s8(wqs +  0), vld1q_s8(xqs +  0));
            vd = vdotq_s32(vd, vld1q_s8(wqs + 16), vld1q_s8(xqs + 16));
            row_sum += wd * t->x_ds[b] * (float)vaddvq_s32(vd);
#else
            int32x4_t vd0 = vdupq_n_s32(0), vd1 = vdupq_n_s32(0);
            for (int j = 0; j < 32; j += 16) {
                int8x16_t vw = vld1q_s8(wqs + j);
                int8x16_t vx = vld1q_s8(xqs + j);
                vd0 = vpadalq_s16(vd0, vmull_s8(vget_low_s8(vw), vget_low_s8(vx)));
                vd1 = vpadalq_s16(vd1, vmull_s8(vget_high_s8(vw), vget_high_s8(vx)));
            }
            row_sum += wd * t->x_ds[b] * (float)vaddvq_s32(vaddq_s32(vd0, vd1));
#endif
        }
        t->out[d] = row_sum;
    }
    return NULL;
}

/* Q4_K int8 dot worker — same idea as q8_int but with on-the-fly nibble unpack.
 * Pre-quantized x: int8 array (x_qs), per-32-element scales (x_ds), and
 * pre-summed int sums per 32-element block (x_isums) so the dmin*mn correction
 * doesn't recompute sum(x_int8) per output row. */
typedef struct {
    float* out; const void* weight; const int8_t* x_qs; const float* x_ds;
    const int32_t* x_isums; size_t row_bytes; int nb_super; int start_row; int end_row;
} q4k_int_task_t;

/* Q5_K int8 dot worker — same pattern as Q4_K, plus 5th bit from qh.
 * The qh array has 1 bit per element across 8 sub-blocks; bit position
 * shifts by 2 per j-iteration (u1 = 1<<(2*iter), u2 = 2<<(2*iter)). */
typedef q4k_int_task_t q5k_int_task_t;

void* q5k_int_dot_worker(void* arg) {
    q5k_int_task_t* t = (q5k_int_task_t*)arg;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    for (int d = t->start_row; d < t->end_row; d++) {
        const block_q5_K* wblk = (const block_q5_K*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        if (d + 1 < t->end_row) {
            const uint8_t* next = (const uint8_t*)t->weight + (size_t)(d + 1) * t->row_bytes;
            __builtin_prefetch(next +   0, 0, 0);
            __builtin_prefetch(next +  64, 0, 0);
            __builtin_prefetch(next + 128, 0, 0);
            __builtin_prefetch(next + 192, 0, 0);
        }
        float row_sum = 0.0f;
        for (int sb = 0; sb < t->nb_super; sb++) {
            const block_q5_K* blk = wblk + sb;
            const float dW    = fp16_to_fp32(blk->d);
            const float dminW = fp16_to_fp32(blk->dmin);

            uint8_t sc[8], mn[8];
            sc[0] = blk->scales[0] & 63;
            sc[1] = blk->scales[1] & 63;
            sc[2] = blk->scales[2] & 63;
            sc[3] = blk->scales[3] & 63;
            mn[0] = blk->scales[4] & 63;
            mn[1] = blk->scales[5] & 63;
            mn[2] = blk->scales[6] & 63;
            mn[3] = blk->scales[7] & 63;
            sc[4] = (blk->scales[8]  & 0x0F) | ((blk->scales[0] >> 6) << 4);
            sc[5] = (blk->scales[9]  & 0x0F) | ((blk->scales[1] >> 6) << 4);
            sc[6] = (blk->scales[10] & 0x0F) | ((blk->scales[2] >> 6) << 4);
            sc[7] = (blk->scales[11] & 0x0F) | ((blk->scales[3] >> 6) << 4);
            mn[4] = (blk->scales[8]  >> 4) | ((blk->scales[4] >> 6) << 4);
            mn[5] = (blk->scales[9]  >> 4) | ((blk->scales[5] >> 6) << 4);
            mn[6] = (blk->scales[10] >> 4) | ((blk->scales[6] >> 6) << 4);
            mn[7] = (blk->scales[11] >> 4) | ((blk->scales[7] >> 6) << 4);

            const uint8_t* ql = blk->qs;   /* 128 bytes of low nibbles */
            const uint8_t* qh = blk->qh;   /* 32 bytes of high bits */
            int sub_base = sb * 8;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;

            for (int j = 0; j < 256; j += 64) {
                int sub_idx_a = sub_base + is;
                int sub_idx_b = sub_base + is + 1;

                /* Low 4 bits of weights */
                uint8x16_t qa = vld1q_u8(ql);
                uint8x16_t qb = vld1q_u8(ql + 16);
                uint8x16_t lo_a = vandq_u8(qa, mask_lo);
                uint8x16_t lo_b = vandq_u8(qb, mask_lo);
                uint8x16_t hi_a = vshrq_n_u8(qa, 4);
                uint8x16_t hi_b = vshrq_n_u8(qb, 4);

                /* 5th bit from qh: extract bit u1 (sub-block A) and u2 (B).
                 * qh[0..15] covers elements 0..15, qh[16..31] covers 16..31.
                 * For sub-block A (low nibbles), bit position = log2(u1).
                 * Convert "bit set" → byte value 16 by shifting to bit 4. */
                uint8x16_t qh_a = vld1q_u8(qh);
                uint8x16_t qh_b = vld1q_u8(qh + 16);
                uint8x16_t u1v = vdupq_n_u8(u1);
                uint8x16_t u2v = vdupq_n_u8(u2);
                /* Test bit, then convert to 0 or 16:
                 *   masked_a = qh & u1  →  0 or u1 (in {1,4,16,64})
                 *   want bit 4 set when u1 bit set → multiply masked_a by (16/u1)
                 * For u1 in {1,4,16,64}: 16/u1 in {16,4,1,1/4}.
                 * Easier: vceqq + select between 0 and 16. */
                uint8x16_t bit_a_lo = vceqq_u8(vandq_u8(qh_a, u1v), u1v);  /* 0xFF or 0x00 */
                uint8x16_t bit_a_hi = vceqq_u8(vandq_u8(qh_b, u1v), u1v);
                uint8x16_t bit_b_lo = vceqq_u8(vandq_u8(qh_a, u2v), u2v);
                uint8x16_t bit_b_hi = vceqq_u8(vandq_u8(qh_b, u2v), u2v);
                uint8x16_t v16 = vdupq_n_u8(16);
                /* And with 16 to get 0 or 16 */
                lo_a = vorrq_u8(lo_a, vandq_u8(bit_a_lo, v16));
                lo_b = vorrq_u8(lo_b, vandq_u8(bit_a_hi, v16));
                hi_a = vorrq_u8(hi_a, vandq_u8(bit_b_lo, v16));
                hi_b = vorrq_u8(hi_b, vandq_u8(bit_b_hi, v16));

                int8x16_t wa_lo = vreinterpretq_s8_u8(lo_a);
                int8x16_t wa_hi = vreinterpretq_s8_u8(lo_b);
                int8x16_t wb_lo = vreinterpretq_s8_u8(hi_a);
                int8x16_t wb_hi = vreinterpretq_s8_u8(hi_b);

                const int8_t* xa = t->x_qs + (size_t)sub_idx_a * 32;
                int8x16_t xa_lo = vld1q_s8(xa);
                int8x16_t xa_hi = vld1q_s8(xa + 16);
                const int8_t* xb = t->x_qs + (size_t)sub_idx_b * 32;
                int8x16_t xb_lo = vld1q_s8(xb);
                int8x16_t xb_hi = vld1q_s8(xb + 16);

#ifdef __ARM_FEATURE_DOTPROD
                int32x4_t accA = vdotq_s32(vdupq_n_s32(0), wa_lo, xa_lo);
                accA = vdotq_s32(accA, wa_hi, xa_hi);
                int32_t isumA = vaddvq_s32(accA);
                int32x4_t accB = vdotq_s32(vdupq_n_s32(0), wb_lo, xb_lo);
                accB = vdotq_s32(accB, wb_hi, xb_hi);
                int32_t isumB = vaddvq_s32(accB);
#else
                int32x4_t accA = vpadalq_s16(vdupq_n_s32(0),
                                              vmull_s8(vget_low_s8(wa_lo), vget_low_s8(xa_lo)));
                accA = vpadalq_s16(accA, vmull_s8(vget_high_s8(wa_lo), vget_high_s8(xa_lo)));
                accA = vpadalq_s16(accA, vmull_s8(vget_low_s8(wa_hi), vget_low_s8(xa_hi)));
                accA = vpadalq_s16(accA, vmull_s8(vget_high_s8(wa_hi), vget_high_s8(xa_hi)));
                int32_t isumA = vaddvq_s32(accA);
                int32x4_t accB = vpadalq_s16(vdupq_n_s32(0),
                                              vmull_s8(vget_low_s8(wb_lo), vget_low_s8(xb_lo)));
                accB = vpadalq_s16(accB, vmull_s8(vget_high_s8(wb_lo), vget_high_s8(xb_lo)));
                accB = vpadalq_s16(accB, vmull_s8(vget_low_s8(wb_hi), vget_low_s8(xb_hi)));
                accB = vpadalq_s16(accB, vmull_s8(vget_high_s8(wb_hi), vget_high_s8(xb_hi)));
                int32_t isumB = vaddvq_s32(accB);
#endif

                float xdA = t->x_ds[sub_idx_a];
                float xdB = t->x_ds[sub_idx_b];
                int32_t xisA = t->x_isums[sub_idx_a];
                int32_t xisB = t->x_isums[sub_idx_b];

                row_sum += (dW * sc[is + 0] * xdA) * (float)isumA
                         - (dminW * mn[is + 0] * xdA) * (float)xisA;
                row_sum += (dW * sc[is + 1] * xdB) * (float)isumB
                         - (dminW * mn[is + 1] * xdB) * (float)xisB;

                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        t->out[d] = row_sum;
    }
    return NULL;
}

/* 2-row inner loop helper for q4k_int worker — processes 2 output rows
 * in parallel for instruction-level parallelism. The two rows share the
 * same x_qs / x_ds / x_isums (read-only activation), but have different
 * weight rows. Parallelism hides load latency on the weight reads. */
static inline void q4k_int_dot_two_rows(
    const block_q4_K* blk0, const block_q4_K* blk1,
    const int8_t* x_qs, const float* x_ds, const int32_t* x_isums,
    int nb_super, float* out0, float* out1)
{
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    float sum0 = 0.0f, sum1 = 0.0f;
    for (int sb = 0; sb < nb_super; sb++) {
        const block_q4_K* b0 = blk0 + sb;
        const block_q4_K* b1 = blk1 + sb;
        const float dW0    = fp16_to_fp32(b0->d);
        const float dminW0 = fp16_to_fp32(b0->dmin);
        const float dW1    = fp16_to_fp32(b1->d);
        const float dminW1 = fp16_to_fp32(b1->dmin);

        uint8_t sc0[8], mn0[8], sc1[8], mn1[8];
        #define UNPACK(SC, MN, BLK)                                              \
            do {                                                                  \
                SC[0] = BLK->scales[0] & 63;                                      \
                SC[1] = BLK->scales[1] & 63;                                      \
                SC[2] = BLK->scales[2] & 63;                                      \
                SC[3] = BLK->scales[3] & 63;                                      \
                MN[0] = BLK->scales[4] & 63;                                      \
                MN[1] = BLK->scales[5] & 63;                                      \
                MN[2] = BLK->scales[6] & 63;                                      \
                MN[3] = BLK->scales[7] & 63;                                      \
                SC[4] = (BLK->scales[8]  & 0x0F) | ((BLK->scales[0] >> 6) << 4);  \
                SC[5] = (BLK->scales[9]  & 0x0F) | ((BLK->scales[1] >> 6) << 4);  \
                SC[6] = (BLK->scales[10] & 0x0F) | ((BLK->scales[2] >> 6) << 4);  \
                SC[7] = (BLK->scales[11] & 0x0F) | ((BLK->scales[3] >> 6) << 4);  \
                MN[4] = (BLK->scales[8]  >> 4) | ((BLK->scales[4] >> 6) << 4);    \
                MN[5] = (BLK->scales[9]  >> 4) | ((BLK->scales[5] >> 6) << 4);    \
                MN[6] = (BLK->scales[10] >> 4) | ((BLK->scales[6] >> 6) << 4);    \
                MN[7] = (BLK->scales[11] >> 4) | ((BLK->scales[7] >> 6) << 4);    \
            } while (0)
        UNPACK(sc0, mn0, b0);
        UNPACK(sc1, mn1, b1);
        #undef UNPACK

        const uint8_t* q0 = b0->qs;
        const uint8_t* q1 = b1->qs;
        int sub_base = sb * 8;
        int is = 0;

        for (int j = 0; j < 256; j += 64) {
            int sub_idx_a = sub_base + is;
            int sub_idx_b = sub_base + is + 1;

            /* Load both rows' nibbles in parallel */
            uint8x16_t qa0 = vld1q_u8(q0);
            uint8x16_t qb0 = vld1q_u8(q0 + 16);
            uint8x16_t qa1 = vld1q_u8(q1);
            uint8x16_t qb1 = vld1q_u8(q1 + 16);
            int8x16_t wa_lo0 = vreinterpretq_s8_u8(vandq_u8(qa0, mask_lo));
            int8x16_t wa_hi0 = vreinterpretq_s8_u8(vandq_u8(qb0, mask_lo));
            int8x16_t wb_lo0 = vreinterpretq_s8_u8(vshrq_n_u8(qa0, 4));
            int8x16_t wb_hi0 = vreinterpretq_s8_u8(vshrq_n_u8(qb0, 4));
            int8x16_t wa_lo1 = vreinterpretq_s8_u8(vandq_u8(qa1, mask_lo));
            int8x16_t wa_hi1 = vreinterpretq_s8_u8(vandq_u8(qb1, mask_lo));
            int8x16_t wb_lo1 = vreinterpretq_s8_u8(vshrq_n_u8(qa1, 4));
            int8x16_t wb_hi1 = vreinterpretq_s8_u8(vshrq_n_u8(qb1, 4));

            const int8_t* xa = x_qs + (size_t)sub_idx_a * 32;
            int8x16_t xa_lo = vld1q_s8(xa);
            int8x16_t xa_hi = vld1q_s8(xa + 16);
            const int8_t* xb = x_qs + (size_t)sub_idx_b * 32;
            int8x16_t xb_lo = vld1q_s8(xb);
            int8x16_t xb_hi = vld1q_s8(xb + 16);

#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t accA0 = vdotq_s32(vdupq_n_s32(0), wa_lo0, xa_lo);
            accA0 = vdotq_s32(accA0, wa_hi0, xa_hi);
            int32x4_t accA1 = vdotq_s32(vdupq_n_s32(0), wa_lo1, xa_lo);
            accA1 = vdotq_s32(accA1, wa_hi1, xa_hi);
            int32x4_t accB0 = vdotq_s32(vdupq_n_s32(0), wb_lo0, xb_lo);
            accB0 = vdotq_s32(accB0, wb_hi0, xb_hi);
            int32x4_t accB1 = vdotq_s32(vdupq_n_s32(0), wb_lo1, xb_lo);
            accB1 = vdotq_s32(accB1, wb_hi1, xb_hi);
            int32_t isumA0 = vaddvq_s32(accA0);
            int32_t isumA1 = vaddvq_s32(accA1);
            int32_t isumB0 = vaddvq_s32(accB0);
            int32_t isumB1 = vaddvq_s32(accB1);
#else
            /* Fallback: same as single-row but doubled; relies on compiler
             * to find ILP. Already a win on M1 if vmull/vpadalq saturate. */
            int32x4_t accA0 = vpadalq_s16(vdupq_n_s32(0), vmull_s8(vget_low_s8(wa_lo0), vget_low_s8(xa_lo)));
            accA0 = vpadalq_s16(accA0, vmull_s8(vget_high_s8(wa_lo0), vget_high_s8(xa_lo)));
            accA0 = vpadalq_s16(accA0, vmull_s8(vget_low_s8(wa_hi0),  vget_low_s8(xa_hi)));
            accA0 = vpadalq_s16(accA0, vmull_s8(vget_high_s8(wa_hi0), vget_high_s8(xa_hi)));
            int32x4_t accA1 = vpadalq_s16(vdupq_n_s32(0), vmull_s8(vget_low_s8(wa_lo1), vget_low_s8(xa_lo)));
            accA1 = vpadalq_s16(accA1, vmull_s8(vget_high_s8(wa_lo1), vget_high_s8(xa_lo)));
            accA1 = vpadalq_s16(accA1, vmull_s8(vget_low_s8(wa_hi1),  vget_low_s8(xa_hi)));
            accA1 = vpadalq_s16(accA1, vmull_s8(vget_high_s8(wa_hi1), vget_high_s8(xa_hi)));
            int32x4_t accB0 = vpadalq_s16(vdupq_n_s32(0), vmull_s8(vget_low_s8(wb_lo0), vget_low_s8(xb_lo)));
            accB0 = vpadalq_s16(accB0, vmull_s8(vget_high_s8(wb_lo0), vget_high_s8(xb_lo)));
            accB0 = vpadalq_s16(accB0, vmull_s8(vget_low_s8(wb_hi0),  vget_low_s8(xb_hi)));
            accB0 = vpadalq_s16(accB0, vmull_s8(vget_high_s8(wb_hi0), vget_high_s8(xb_hi)));
            int32x4_t accB1 = vpadalq_s16(vdupq_n_s32(0), vmull_s8(vget_low_s8(wb_lo1), vget_low_s8(xb_lo)));
            accB1 = vpadalq_s16(accB1, vmull_s8(vget_high_s8(wb_lo1), vget_high_s8(xb_lo)));
            accB1 = vpadalq_s16(accB1, vmull_s8(vget_low_s8(wb_hi1),  vget_low_s8(xb_hi)));
            accB1 = vpadalq_s16(accB1, vmull_s8(vget_high_s8(wb_hi1), vget_high_s8(xb_hi)));
            int32_t isumA0 = vaddvq_s32(accA0);
            int32_t isumA1 = vaddvq_s32(accA1);
            int32_t isumB0 = vaddvq_s32(accB0);
            int32_t isumB1 = vaddvq_s32(accB1);
#endif

            float xdA = x_ds[sub_idx_a];
            float xdB = x_ds[sub_idx_b];
            int32_t xisA = x_isums[sub_idx_a];
            int32_t xisB = x_isums[sub_idx_b];

            sum0 += (dW0 * sc0[is + 0] * xdA) * (float)isumA0
                  - (dminW0 * mn0[is + 0] * xdA) * (float)xisA;
            sum0 += (dW0 * sc0[is + 1] * xdB) * (float)isumB0
                  - (dminW0 * mn0[is + 1] * xdB) * (float)xisB;
            sum1 += (dW1 * sc1[is + 0] * xdA) * (float)isumA1
                  - (dminW1 * mn1[is + 0] * xdA) * (float)xisA;
            sum1 += (dW1 * sc1[is + 1] * xdB) * (float)isumB1
                  - (dminW1 * mn1[is + 1] * xdB) * (float)xisB;

            q0 += 32;
            q1 += 32;
            is += 2;
        }
    }
    *out0 = sum0;
    *out1 = sum1;
}

void* q4k_int_dot_worker(void* arg) {
    q4k_int_task_t* t = (q4k_int_task_t*)arg;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    int d = t->start_row;
    /* 2-row inner loop while we have pairs */
    for (; d + 1 < t->end_row; d += 2) {
        const block_q4_K* wblk0 = (const block_q4_K*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        const block_q4_K* wblk1 = (const block_q4_K*)((const uint8_t*)t->weight + (size_t)(d + 1) * t->row_bytes);
        if (d + 2 < t->end_row) {
            const uint8_t* next = (const uint8_t*)t->weight + (size_t)(d + 2) * t->row_bytes;
            __builtin_prefetch(next +   0, 0, 0);
            __builtin_prefetch(next +  64, 0, 0);
            __builtin_prefetch(next + 128, 0, 0);
            __builtin_prefetch(next + 192, 0, 0);
        }
        q4k_int_dot_two_rows(wblk0, wblk1, t->x_qs, t->x_ds, t->x_isums,
                             t->nb_super, &t->out[d], &t->out[d + 1]);
    }
    /* Tail: single row */
    for (; d < t->end_row; d++) {
        const block_q4_K* wblk = (const block_q4_K*)((const uint8_t*)t->weight + (size_t)d * t->row_bytes);
        float row_sum = 0.0f;
        for (int sb = 0; sb < t->nb_super; sb++) {
            const block_q4_K* blk = wblk + sb;
            const float dW    = fp16_to_fp32(blk->d);
            const float dminW = fp16_to_fp32(blk->dmin);

            /* 6-bit packed sub-block scales (sc) and mins (mn).
             * Layout matches fused_dot_q4_k. */
            uint8_t sc[8], mn[8];
            sc[0] = blk->scales[0] & 63;
            sc[1] = blk->scales[1] & 63;
            sc[2] = blk->scales[2] & 63;
            sc[3] = blk->scales[3] & 63;
            mn[0] = blk->scales[4] & 63;
            mn[1] = blk->scales[5] & 63;
            mn[2] = blk->scales[6] & 63;
            mn[3] = blk->scales[7] & 63;
            sc[4] = (blk->scales[8]  & 0x0F) | ((blk->scales[0] >> 6) << 4);
            sc[5] = (blk->scales[9]  & 0x0F) | ((blk->scales[1] >> 6) << 4);
            sc[6] = (blk->scales[10] & 0x0F) | ((blk->scales[2] >> 6) << 4);
            sc[7] = (blk->scales[11] & 0x0F) | ((blk->scales[3] >> 6) << 4);
            mn[4] = (blk->scales[8]  >> 4) | ((blk->scales[4] >> 6) << 4);
            mn[5] = (blk->scales[9]  >> 4) | ((blk->scales[5] >> 6) << 4);
            mn[6] = (blk->scales[10] >> 4) | ((blk->scales[6] >> 6) << 4);
            mn[7] = (blk->scales[11] >> 4) | ((blk->scales[7] >> 6) << 4);

            const uint8_t* q = blk->qs;
            int sub_base = sb * 8;
            int is = 0;

            /* 4 j-iterations × 64 elements = 256-element super-block.
             * Each iteration handles two 32-element sub-blocks (lo+hi nibbles). */
            for (int j = 0; j < 256; j += 64) {
                int sub_idx_a = sub_base + is;     /* offset j..j+31  */
                int sub_idx_b = sub_base + is + 1; /* offset j+32..j+63 */

                /* Load 32 bytes of packed nibbles */
                uint8x16_t qa = vld1q_u8(q);
                uint8x16_t qb = vld1q_u8(q + 16);
                /* lo_a: weights j..j+15, lo_b: weights j+16..j+31 */
                int8x16_t wa_lo = vreinterpretq_s8_u8(vandq_u8(qa, mask_lo));
                int8x16_t wa_hi = vreinterpretq_s8_u8(vandq_u8(qb, mask_lo));
                /* hi_a: weights j+32..j+47, hi_b: weights j+48..j+63 */
                int8x16_t wb_lo = vreinterpretq_s8_u8(vshrq_n_u8(qa, 4));
                int8x16_t wb_hi = vreinterpretq_s8_u8(vshrq_n_u8(qb, 4));

                /* x for sub-block A: 32 int8 values starting at sub_idx_a*32 */
                const int8_t* xa = t->x_qs + (size_t)sub_idx_a * 32;
                int8x16_t xa_lo = vld1q_s8(xa);
                int8x16_t xa_hi = vld1q_s8(xa + 16);
                /* x for sub-block B */
                const int8_t* xb = t->x_qs + (size_t)sub_idx_b * 32;
                int8x16_t xb_lo = vld1q_s8(xb);
                int8x16_t xb_hi = vld1q_s8(xb + 16);

#ifdef __ARM_FEATURE_DOTPROD
                /* ARMv8.2 dotprod: 16 int8 MACs per call. 2 calls = full sub-block. */
                int32x4_t accA = vdotq_s32(vdupq_n_s32(0), wa_lo, xa_lo);
                accA = vdotq_s32(accA, wa_hi, xa_hi);
                int32_t isumA = vaddvq_s32(accA);
                int32x4_t accB = vdotq_s32(vdupq_n_s32(0), wb_lo, xb_lo);
                accB = vdotq_s32(accB, wb_hi, xb_hi);
                int32_t isumB = vaddvq_s32(accB);
#else
                /* int8 dot for sub-block A: 4 widening multiplies, padalq accumulates */
                int32x4_t accA = vpadalq_s16(vdupq_n_s32(0),
                                              vmull_s8(vget_low_s8(wa_lo),  vget_low_s8(xa_lo)));
                accA = vpadalq_s16(accA, vmull_s8(vget_high_s8(wa_lo), vget_high_s8(xa_lo)));
                accA = vpadalq_s16(accA, vmull_s8(vget_low_s8(wa_hi),  vget_low_s8(xa_hi)));
                accA = vpadalq_s16(accA, vmull_s8(vget_high_s8(wa_hi), vget_high_s8(xa_hi)));
                int32_t isumA = vaddvq_s32(accA);

                int32x4_t accB = vpadalq_s16(vdupq_n_s32(0),
                                              vmull_s8(vget_low_s8(wb_lo),  vget_low_s8(xb_lo)));
                accB = vpadalq_s16(accB, vmull_s8(vget_high_s8(wb_lo), vget_high_s8(xb_lo)));
                accB = vpadalq_s16(accB, vmull_s8(vget_low_s8(wb_hi),  vget_low_s8(xb_hi)));
                accB = vpadalq_s16(accB, vmull_s8(vget_high_s8(wb_hi), vget_high_s8(xb_hi)));
                int32_t isumB = vaddvq_s32(accB);
#endif

                /* Combine: weight_i = q_i * (d*sc) - (dmin*mn)
                 *   dot = sum(w_i * x_i)
                 *       = (d*sc) * sum(q_i * x_int8_i) * x_d
                 *         - (dmin*mn) * sum(x_int8_i) * x_d
                 *   First term uses isum (just computed). Second term uses
                 *   precomputed x_isums to avoid re-summing per row. */
                float xdA = t->x_ds[sub_idx_a];
                float xdB = t->x_ds[sub_idx_b];
                int32_t xisA = t->x_isums[sub_idx_a];
                int32_t xisB = t->x_isums[sub_idx_b];

                row_sum += (dW * sc[is + 0] * xdA) * (float)isumA
                         - (dminW * mn[is + 0] * xdA) * (float)xisA;
                row_sum += (dW * sc[is + 1] * xdB) * (float)isumB
                         - (dminW * mn[is + 1] * xdB) * (float)xisB;

                q += 32;
                is += 2;
            }
        }
        t->out[d] = row_sum;
    }
    return NULL;
}
#endif

/* Force-CPU variant: skips Metal dispatch entirely. Used for fused QKV
 * matmuls (Phi-3) where the Metal buffer management has a bug with the
 * large output dimension (out_dim = 3 * hidden_dim). The CPU NEON path
 * handles it correctly. */
void tq_matmul_gguf_cpu(float* out, const float* x,
                         const void* weight, tq_ggml_dtype weight_type,
                         int out_dim, int in_dim);

/* Thread-local flag to force CPU path in tq_matmul_gguf. Set to 1
 * before calling tq_matmul_gguf to skip Metal dispatch. Used for
 * Phi-3 fused QKV/FFN matmuls where Metal has a buffer sizing bug
 * with the unusually large output dimensions. */
/* Global flag (NOT thread-local) — worker threads in the matmul thread pool
 * must see the same value set by the main thread. Prior _Thread_local version
 * silently allowed Metal dispatch from worker threads despite the main thread
 * setting the flag to 1 (observed as Phi-3.5 Q4_K_M garbage output under Metal). */
int tq_matmul_force_cpu = 0;
__thread int tq_tls_force_serial_matmul = 0;

void tq_matmul_gguf(float* out, const float* x,
                    const void* weight, tq_ggml_dtype weight_type,
                    int out_dim, int in_dim)
{
    /* Metal GPU dispatch for supported GGUF types.
     *
     * Two modes:
     *   1. Batch mode: when tq_metal_batch_begin_if_available() was called
     *      by the transformer, all matmuls encode into one command buffer.
     *      Dispatch overhead is amortized across the batch.
     *   2. Immediate mode: for large matmuls (out_dim >= 512), the GPU
     *      throughput beats CPU even with per-call dispatch overhead.
     *      Small matmuls stay on CPU where fused dot is faster.
     *
     * Returns 0 on success, -1 if type unsupported or Metal unavailable.
     * On -1, falls through to CPU path below. */
#ifdef TQ_HAS_METAL
    {
        extern int tq_metal_available(void);
        extern int tq_metal_matmul_gguf(float*, const float*, const void*,
                                        tq_ggml_dtype, int, int);
        extern int tq_metal_batch_active(void);

        if (!tq_matmul_force_cpu && tq_metal_available()) {
            /* In batch mode, always dispatch to GPU (overhead is amortized).
             * In immediate mode, only for types that have Metal pipelines
             * AND large matrices where GPU wins. */
            int has_metal_pipeline = (weight_type == TQ_GGML_TYPE_IQ2_XXS ||
                                      weight_type == TQ_GGML_TYPE_IQ2_S ||
                                      weight_type == TQ_GGML_TYPE_Q8_0 ||
                                      weight_type == TQ_GGML_TYPE_Q4_K);
            int use_gpu = has_metal_pipeline && (tq_metal_batch_active() || (out_dim >= 512));
            if (use_gpu) {
                int rc = tq_metal_matmul_gguf(out, x, weight, weight_type,
                                              out_dim, in_dim);
                if (rc == 0) return; /* GPU handled it */
                /* rc == -1: unsupported type, fall through to CPU */
            }
        }
    }
#endif

    const size_t block_bytes = tq_ggml_type_size(weight_type);
    const int    block_elems = tq_ggml_type_blck(weight_type);

    if (block_bytes == 0 || block_elems == 0) {
        static int warn_count = 0;
        if (warn_count++ < 5) {
            void* ra = __builtin_return_address(0);
            fprintf(stderr, "tq_matmul_gguf: unsupported type %d (out=%d, in=%d, w=%p, caller=%p)\n",
                    (int)weight_type, out_dim, in_dim, weight, ra);
        }
        memset(out, 0, (size_t)out_dim * sizeof(float));
        return;
    }

    const int    n_blocks  = in_dim / block_elems;
    const size_t row_bytes = (size_t)n_blocks * block_bytes;

    /* Q8×Q8 integer dot: input pre-quantized in transformer (once per layer).
     * Uses int8×int8 vmull_s8+vpadalq_s16 — ~2x faster than float fused dot. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_Q8_0 && g_preq_qs != NULL && in_dim <= 4096) {
        const int8_t* xqs = g_preq_qs;
        const float*  xds = g_preq_ds;

        /* Always use thread pool — pass preq data via task struct (TLS doesn't propagate) */
        int n_threads = tq_get_threads();
        if (tq_tls_force_serial_matmul) n_threads = 1;
        if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
        if (n_threads > out_dim) n_threads = out_dim;
        if (n_threads < 1) n_threads = 1;

        q8_int_task_t tasks[TQ_TP_MAX];
        void* ptrs[TQ_TP_MAX];
        int rows_per = out_dim / n_threads;
        for (int t = 0; t < n_threads; t++) {
            tasks[t] = (q8_int_task_t){
                .out = out, .weight = weight, .x_qs = xqs, .x_ds = xds,
                .row_bytes = row_bytes, .n_blocks = n_blocks,
                .start_row = t * rows_per,
                .end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per
            };
            ptrs[t] = &tasks[t];
        }
        if (n_threads == 1) {
            q8_int_dot_worker(ptrs[0]);
        } else {
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(q8_int_dot_worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- IQ2_XXS × int8 task definition (shared by serial and parallel paths) ---- */
#if TQ_HAS_NEON
    /* forward-declare workers: defined at file scope below */
    extern void* iq2_xxs_int_dot_worker(void* arg);
    extern void* iq2_s_int_dot_worker(void* arg);
#endif

    /* ---- IQ4_XS × int8 fast path (vqtbl1q_s8 + vdotq_s32) ----
     * Used by UD-IQ3_XXS (some routed experts) and UD-IQ4_XS.
     * 16-entry codebook fits in one NEON TBL lookup. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_IQ4_XS
        && in_dim >= 256 && in_dim <= 16384
        && (in_dim % 256 == 0) && !getenv("TQ_IQ4XS_NOINT"))
    {
        int n_blocks32 = in_dim / 32;
        int8_t x_qs_buf[16384];
        float  x_ds_buf[512];
        for (int bb = 0; bb < n_blocks32; bb++) {
            const float* xp = x + bb * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xp[j] < 0 ? -xp[j] : xp[j];
                if (a > amax) amax = a;
            }
            float dq = amax / 127.0f;
            x_ds_buf[bb] = dq;
            if (dq > 0.0f) {
                float id = 1.0f / dq;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(xp[j] * id);
                    x_qs_buf[bb * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
            } else {
                memset(x_qs_buf + bb * 32, 0, 32);
            }
        }

        size_t row_bytes_iq4xs = (size_t)in_dim / 256 * sizeof(block_iq4_xs);

        int n_threads = tq_get_threads();
        if (tq_tls_force_serial_matmul) n_threads = 1;

        if (n_threads <= 1 || out_dim <= n_threads) {
            for (int d = 0; d < out_dim; d++) {
                const void* r = (const uint8_t*)weight + (size_t)d * row_bytes_iq4xs;
                out[d] = fused_dot_iq4_xs_int8(r, x_qs_buf, x_ds_buf, in_dim);
            }
        } else {
            iq4xs_int_task_t tasks[TQ_TP_MAX];
            void* ptrs[TQ_TP_MAX];
            int rows_per = out_dim / n_threads;
            for (int t = 0; t < n_threads; t++) {
                tasks[t] = (iq4xs_int_task_t){
                    .out = out, .weight = weight,
                    .x_qs = x_qs_buf, .x_ds = x_ds_buf,
                    .row_bytes = row_bytes_iq4xs, .in_dim = in_dim,
                    .start_row = t * rows_per,
                    .end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per
                };
                ptrs[t] = &tasks[t];
            }
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(iq4_xs_int_dot_worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- Q3_K × int8 fast path (vdotq_s32) ----
     * Q3_K is the primary kernel used by UD-Q3_K_* quantizations (and
     * embedded by UD-IQ3_XXS for critical layers). `fused_dot_q3_k` was
     * pure scalar — same fix pattern as Q6_K int8. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_Q3_K
        && in_dim >= 256 && in_dim <= 16384
        && (in_dim % 256 == 0) && !getenv("TQ_Q3K_NOINT"))
    {
        int n_blocks32 = in_dim / 32;
        int8_t x_qs_buf[16384];
        float  x_ds_buf[512];
        for (int bb = 0; bb < n_blocks32; bb++) {
            const float* xp = x + bb * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xp[j] < 0 ? -xp[j] : xp[j];
                if (a > amax) amax = a;
            }
            float dq = amax / 127.0f;
            x_ds_buf[bb] = dq;
            if (dq > 0.0f) {
                float id = 1.0f / dq;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(xp[j] * id);
                    x_qs_buf[bb * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
            } else {
                memset(x_qs_buf + bb * 32, 0, 32);
            }
        }

        size_t row_bytes_q3k = (size_t)in_dim / 256 * sizeof(block_q3_K);

        int n_threads = tq_get_threads();
        if (tq_tls_force_serial_matmul) n_threads = 1;

        if (n_threads <= 1 || out_dim <= n_threads) {
            for (int d = 0; d < out_dim; d++) {
                const void* row = (const uint8_t*)weight + (size_t)d * row_bytes_q3k;
                out[d] = fused_dot_q3_k_int8(row, x_qs_buf, x_ds_buf, in_dim);
            }
        } else {
            q3k_int_task_t tasks[TQ_TP_MAX];
            void* ptrs[TQ_TP_MAX];
            int rows_per = out_dim / n_threads;
            for (int t = 0; t < n_threads; t++) {
                tasks[t] = (q3k_int_task_t){
                    .out = out, .weight = weight,
                    .x_qs = x_qs_buf, .x_ds = x_ds_buf,
                    .row_bytes = row_bytes_q3k, .in_dim = in_dim,
                    .start_row = t * rows_per,
                    .end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per
                };
                ptrs[t] = &tasks[t];
            }
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(q3_k_int_dot_worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- IQ3_XXS × int8 fast path (vdotq_s32) ----
     * UD-IQ3_XXS uses IQ3_XXS for the bulk of routed-expert weights.
     * Previous fused_dot_iq3_xxs was partially NEON (float FMA); int8
     * vdotq_s32 gives ~2× kernel throughput. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_IQ3_XXS
        && in_dim >= 256 && in_dim <= 16384
        && (in_dim % 256 == 0) && !getenv("TQ_IQ3XXS_NOINT"))
    {
        int n_blocks32 = in_dim / 32;
        int8_t x_qs_buf[16384];
        float  x_ds_buf[512];
        for (int bb = 0; bb < n_blocks32; bb++) {
            const float* xp = x + bb * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xp[j] < 0 ? -xp[j] : xp[j];
                if (a > amax) amax = a;
            }
            float dq = amax / 127.0f;
            x_ds_buf[bb] = dq;
            if (dq > 0.0f) {
                float id = 1.0f / dq;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(xp[j] * id);
                    x_qs_buf[bb * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
            } else {
                memset(x_qs_buf + bb * 32, 0, 32);
            }
        }

        size_t row_bytes_iq3x = (size_t)in_dim / 256 * 98;

        int n_threads = tq_get_threads();
        if (tq_tls_force_serial_matmul) n_threads = 1;

        if (n_threads <= 1 || out_dim <= n_threads) {
            for (int d = 0; d < out_dim; d++) {
                const void* row = (const uint8_t*)weight + (size_t)d * row_bytes_iq3x;
                out[d] = fused_dot_iq3_xxs_int8(row, x_qs_buf, x_ds_buf, in_dim);
            }
        } else {
            iq3xxs_int_task_t tasks[TQ_TP_MAX];
            void* ptrs[TQ_TP_MAX];
            int rows_per = out_dim / n_threads;
            for (int t = 0; t < n_threads; t++) {
                tasks[t] = (iq3xxs_int_task_t){
                    .out = out, .weight = weight,
                    .x_qs = x_qs_buf, .x_ds = x_ds_buf,
                    .row_bytes = row_bytes_iq3x, .in_dim = in_dim,
                    .start_row = t * rows_per,
                    .end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per
                };
                ptrs[t] = &tasks[t];
            }
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(iq3_xxs_int_dot_worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- IQ3_S × int8 fast path (vdotq_s32) ----
     * UD-IQ2_XXS quantizations embed IQ3_S for some layers (e.g., Qwen3.6
     * routed-expert critical paths). The float fused-dot is scalar; sample
     * caught it among the hot kernels alongside IQ2_XXS/IQ2_S. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_IQ3_S
        && in_dim >= 256 && in_dim <= 16384
        && (in_dim % 256 == 0) && !getenv("TQ_IQ3S_NOINT"))
    {
        int n_blocks32 = in_dim / 32;
        int8_t x_qs_buf[16384];
        float  x_ds_buf[512];
        for (int bb = 0; bb < n_blocks32; bb++) {
            const float* xp = x + bb * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xp[j] < 0 ? -xp[j] : xp[j];
                if (a > amax) amax = a;
            }
            float dq = amax / 127.0f;
            x_ds_buf[bb] = dq;
            if (dq > 0.0f) {
                float id = 1.0f / dq;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(xp[j] * id);
                    x_qs_buf[bb * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
            } else {
                memset(x_qs_buf + bb * 32, 0, 32);
            }
        }

        size_t row_bytes_iq3s = (size_t)in_dim / 256 * sizeof(block_iq3_s_t);

        int n_threads = tq_get_threads();
        if (tq_tls_force_serial_matmul) n_threads = 1;

        if (n_threads <= 1 || out_dim <= n_threads) {
            for (int d = 0; d < out_dim; d++) {
                const void* row = (const uint8_t*)weight + (size_t)d * row_bytes_iq3s;
                out[d] = fused_dot_iq3_s_int8(row, x_qs_buf, x_ds_buf, in_dim);
            }
        } else {
            iq3s_int_task_t tasks[TQ_TP_MAX];
            void* ptrs[TQ_TP_MAX];
            int rows_per = out_dim / n_threads;
            for (int t = 0; t < n_threads; t++) {
                tasks[t] = (iq3s_int_task_t){
                    .out = out, .weight = weight,
                    .x_qs = x_qs_buf, .x_ds = x_ds_buf,
                    .row_bytes = row_bytes_iq3s, .in_dim = in_dim,
                    .start_row = t * rows_per,
                    .end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per
                };
                ptrs[t] = &tasks[t];
            }
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(iq3_s_int_dot_worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- Q6_K × int8 fast path (vdotq_s32) ----
     * Q4_K_M models embed Q6_K for o_proj/down_proj. The float fused-dot is
     * scalar; sample on Qwen3.5-4B shows it dominates decode. Pre-quantize
     * x once and use vdotq_s32 over 16-elem sub-blocks for ~5× speedup. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_Q6_K
        && in_dim >= 256 && in_dim <= 16384
        && (in_dim % 256 == 0) && !getenv("TQ_Q6K_NOINT"))
    {
        int n_blocks32 = in_dim / 32;
        int8_t x_qs_buf[16384];
        float  x_ds_buf[512];
        for (int bb = 0; bb < n_blocks32; bb++) {
            const float* xp = x + bb * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xp[j] < 0 ? -xp[j] : xp[j];
                if (a > amax) amax = a;
            }
            float dq = amax / 127.0f;
            x_ds_buf[bb] = dq;
            if (dq > 0.0f) {
                float id = 1.0f / dq;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(xp[j] * id);
                    x_qs_buf[bb * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
            } else {
                memset(x_qs_buf + bb * 32, 0, 32);
            }
        }

        size_t row_bytes_q6 = (size_t)in_dim / 256 * sizeof(block_q6_K);

        int n_threads = tq_get_threads();
        if (tq_tls_force_serial_matmul) n_threads = 1;

        if (n_threads <= 1 || out_dim <= n_threads) {
            for (int d = 0; d < out_dim; d++) {
                const void* row = (const uint8_t*)weight + (size_t)d * row_bytes_q6;
                out[d] = fused_dot_q6_k_int8(row, x_qs_buf, x_ds_buf, in_dim);
            }
        } else {
            q6k_int_task_t tasks[TQ_TP_MAX];
            void* ptrs[TQ_TP_MAX];
            int rows_per = out_dim / n_threads;
            for (int t = 0; t < n_threads; t++) {
                tasks[t] = (q6k_int_task_t){
                    .out = out, .weight = weight,
                    .x_qs = x_qs_buf, .x_ds = x_ds_buf,
                    .row_bytes = row_bytes_q6, .in_dim = in_dim,
                    .start_row = t * rows_per,
                    .end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per
                };
                ptrs[t] = &tasks[t];
            }
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(q6_k_int_dot_worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- IQ2_XXS × int8 fast path (auto-quantize activation, vdotq_s32) ----
     * llama.cpp-style: pre-quantize x to Q8_0-compatible int8 blocks, then
     * feed into vdotq_s32 alongside sign-expanded grid values. Closes the
     * 35% generation-speed gap measured against llama.cpp on Qwen3.6-A3B.
     *
     * Theory: vdotq_s32 issues 16 int8×int8 ops/cycle on M1 Pro; vfmaq_f32
     * only 4 float ops/cycle. For 3B-active MoE models with 40 layers ×
     * 256 experts, this dominates decode-time throughput.
     *
     * TQ_IQ2_XXS_NOINT=1 reverts to float fused-dot (for A/B comparison). */
#if TQ_HAS_NEON
    if ((weight_type == TQ_GGML_TYPE_IQ2_XXS || weight_type == TQ_GGML_TYPE_IQ2_S)
        && in_dim >= 256 && in_dim <= 16384
        && (in_dim % 256 == 0) && !getenv("TQ_IQ2_XXS_NOINT"))
    {
        /* Quantize x per 32-element block (Q8_0 scale layout).
         * in_dim is multiple of 256, so also multiple of 32. */
        int n_blocks32 = in_dim / 32;
        int8_t x_qs_buf[16384];
        float  x_ds_buf[512];
        for (int b = 0; b < n_blocks32; b++) {
            const float* xp = x + b * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xp[j] < 0 ? -xp[j] : xp[j];
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f;
            x_ds_buf[b] = d;
            if (d > 0.0f) {
                float id = 1.0f / d;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(xp[j] * id);
                    x_qs_buf[b * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
            } else {
                memset(x_qs_buf + b * 32, 0, 32);
            }
        }

        /* Dispatch one row per output dim. Row size: 66 bytes (IQ2_XXS) or 82 (IQ2_S) per 256 weights. */
        int is_iq2s = (weight_type == TQ_GGML_TYPE_IQ2_S);
        size_t block_bytes = is_iq2s ? 82 : 66;
        size_t row_bytes = (size_t)in_dim / 256 * block_bytes;
        float (*dot_fn)(const void*, const int8_t*, const float*, int) =
            is_iq2s ? fused_dot_iq2_s_int8 : fused_dot_iq2_xxs_int8;
        void* (*worker_fn)(void*) = is_iq2s ? iq2_s_int_dot_worker : iq2_xxs_int_dot_worker;

        int n_threads = tq_get_threads();
        if (tq_tls_force_serial_matmul) n_threads = 1;

        if (n_threads <= 1 || out_dim <= n_threads) {
            for (int d = 0; d < out_dim; d++) {
                const void* row = (const uint8_t*)weight + (size_t)d * row_bytes;
                out[d] = dot_fn(row, x_qs_buf, x_ds_buf, in_dim);
            }
        } else {
            iq2_int_task_t tasks[TQ_TP_MAX];
            void* ptrs[TQ_TP_MAX];
            int rows_per = out_dim / n_threads;
            for (int t = 0; t < n_threads; t++) {
                tasks[t] = (iq2_int_task_t){
                    .out = out, .weight = weight,
                    .x_qs = x_qs_buf, .x_ds = x_ds_buf,
                    .row_bytes = row_bytes, .in_dim = in_dim,
                    .start_row = t * rows_per,
                    .end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per
                };
                ptrs[t] = &tasks[t];
            }
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(worker_fn, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- Q8_0×Q8 integer dot fast path (auto-quantize activation) ----
     * When g_preq_qs is not set (most callers), quantize the input
     * activation inline and use the NEON int8×int8 path. Cost of
     * one-time activation quantization is O(in_dim); matmul is
     * O(in_dim * out_dim), so quantization overhead is negligible
     * for typical out_dim >= 256.
     *
     * Previously disabled with false claim "per-call overhead > benefit"
     * — actually this is 3-4x faster than float fused_dot for Q8_0. */
#if TQ_HAS_NEON
    if (weight_type == TQ_GGML_TYPE_Q8_0 && in_dim >= 32 && in_dim <= 16384) {
        /* Step 1: Quantize input x[in_dim] to Q8 blocks on stack.
         * Buffers sized for max FFN dim (Llama-70B uses 28672, but most
         * models <= 16384). Stack usage: 16KB + 2KB = 18KB. */
        int8_t  x_qs[16384];
        float   x_ds[512];
        if (in_dim <= 16384) {
            for (int b = 0; b < n_blocks; b++) {
                const float* xp = x + b * 32;
                /* Find absmax for this block */
                float amax = 0.0f;
                for (int j = 0; j < 32; j++) {
                    float a = xp[j] < 0 ? -xp[j] : xp[j];
                    if (a > amax) amax = a;
                }
                float d = amax / 127.0f;
                x_ds[b] = d;
                if (d > 0.0f) {
                    float id = 1.0f / d;
                    for (int j = 0; j < 32; j++) {
                        int v = (int)roundf(xp[j] * id);
                        x_qs[b * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                    }
                } else {
                    memset(x_qs + b * 32, 0, 32);
                }
            }

            /* Step 2: Integer dot for each output row */
            int n_threads = tq_get_threads();
            if (tq_tls_force_serial_matmul) n_threads = 1;
            /* For large matmuls, use thread pool; for small ones, single thread */
            if (n_threads <= 1 || out_dim <= n_threads) {
                for (int d = 0; d < out_dim; d++) {
                    const block_q8_0* wblk = (const block_q8_0*)((const uint8_t*)weight + (size_t)d * row_bytes);
                    float row_sum = 0.0f;
                    for (int b = 0; b < n_blocks; b++) {
                        const float wd = fp16_to_fp32(wblk[b].d);
                        const int8_t* wqs = wblk[b].qs;
                        const int8_t* xqs = x_qs + b * 32;
                        /* NEON int8×int8 dot using vdotq_s32 (M1+) or widening multiply */
                        int32x4_t vdot0 = vdupq_n_s32(0);
                        int32x4_t vdot1 = vdupq_n_s32(0);
                        for (int j = 0; j < 32; j += 16) {
                            int8x16_t vw = vld1q_s8(wqs + j);
                            int8x16_t vx = vld1q_s8(xqs + j);
                            /* Widening multiply-accumulate: int8×int8 → int16 → int32 */
                            int16x8_t vprod_lo = vmull_s8(vget_low_s8(vw), vget_low_s8(vx));
                            int16x8_t vprod_hi = vmull_s8(vget_high_s8(vw), vget_high_s8(vx));
                            vdot0 = vpadalq_s16(vdot0, vprod_lo);
                            vdot1 = vpadalq_s16(vdot1, vprod_hi);
                        }
                        int32_t isum = vaddvq_s32(vaddq_s32(vdot0, vdot1));
                        row_sum += wd * x_ds[b] * (float)isum;
                    }
                    out[d] = row_sum;
                }
            } else {
                /* Multi-threaded: each thread processes a range of output rows */
                if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
                if (n_threads > out_dim) n_threads = out_dim;
                q8_int_task_t q8_tasks[TQ_TP_MAX];
                void* q8_ptrs[TQ_TP_MAX];

                int rows_per = out_dim / n_threads;
                for (int t = 0; t < n_threads; t++) {
                    q8_tasks[t].out = out;
                    q8_tasks[t].weight = weight;
                    q8_tasks[t].x_qs = x_qs;
                    q8_tasks[t].x_ds = x_ds;
                    q8_tasks[t].row_bytes = row_bytes;
                    q8_tasks[t].n_blocks = n_blocks;
                    q8_tasks[t].start_row = t * rows_per;
                    q8_tasks[t].end_row = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per;
                    q8_ptrs[t] = &q8_tasks[t];
                }
                extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
                extern void* q8_int_dot_worker(void* arg);
                tq_tp_run(q8_int_dot_worker, q8_ptrs, n_threads);
            }
            return;
        }
    }

    /* ---- Q4_K × int8 dot fast path (auto-quantize activation) ----
     * Same pattern as Q8_0: quantize x once to int8 (32-element blocks),
     * precompute per-block int sums for the dmin*mn correction, then
     * run vmull_s8 + vpadalq_s16 dots over 4-bit nibbles unpacked to int8.
     * Replaces the float fused_dot_q4_k path on Phi-3.5/Llama Q4_K_M models. */
    if ((weight_type == TQ_GGML_TYPE_Q4_K || weight_type == TQ_GGML_TYPE_Q5_K)
        && in_dim >= 256 && in_dim <= 16384 && (in_dim % 256) == 0)
    {
        /* Stack buffers: x as int8 (16KB), per-block scales (512 floats =
         * 2KB), per-block int sums (512 ints = 2KB). Total ~20KB. */
        int8_t  x_qs[16384];
        float   x_ds[512];
        int32_t x_isums[512];

        /* Step 1: Per-32-element-block quantization of x to int8. */
        const int n_blocks_x = in_dim / 32;
        for (int b = 0; b < n_blocks_x; b++) {
            const float* xp = x + b * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = xp[j] < 0 ? -xp[j] : xp[j];
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f;
            x_ds[b] = d;
            int32_t isum = 0;
            if (d > 0.0f) {
                float id = 1.0f / d;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(xp[j] * id);
                    int8_t q = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                    x_qs[b * 32 + j] = q;
                    isum += q;
                }
            } else {
                memset(x_qs + b * 32, 0, 32);
            }
            x_isums[b] = isum;
        }

        const int nb_super = in_dim / 256; /* number of 256-elem super-blocks */
        int n_threads = tq_get_threads();
        if (tq_tls_force_serial_matmul) n_threads = 1;
        if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
        if (n_threads > out_dim) n_threads = out_dim;
        if (n_threads < 1) n_threads = 1;

        q4k_int_task_t tasks[TQ_TP_MAX];
        void* ptrs[TQ_TP_MAX];
        int rows_per = out_dim / n_threads;
        for (int t = 0; t < n_threads; t++) {
            tasks[t].out       = out;
            tasks[t].weight    = weight;
            tasks[t].x_qs      = x_qs;
            tasks[t].x_ds      = x_ds;
            tasks[t].x_isums   = x_isums;
            tasks[t].row_bytes = row_bytes;
            tasks[t].nb_super  = nb_super;
            tasks[t].start_row = t * rows_per;
            tasks[t].end_row   = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per;
            ptrs[t] = &tasks[t];
        }
        void* (*worker)(void*) = (weight_type == TQ_GGML_TYPE_Q5_K)
                                  ? q5k_int_dot_worker : q4k_int_dot_worker;
        if (n_threads == 1) {
            worker(ptrs[0]);
        } else {
            extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
            tq_tp_run(worker, ptrs, n_threads);
        }
        return;
    }
#endif

    /* ---- Fused fast paths: dequant + dot in one pass, no tmp buffer ---- */

    /* Fused path function pointer: returns dot product for one row */
    float (*fused_dot)(const void*, const float*, int) = NULL;

    switch (weight_type) {
        case TQ_GGML_TYPE_IQ2_XXS:
#if TQ_HAS_NEON
            fused_dot = fused_dot_iq2_xxs_neon;
#else
            fused_dot = fused_dot_iq2_xxs;
#endif
            break;
        case TQ_GGML_TYPE_IQ2_S:
            fused_dot = fused_dot_iq2_s;
            break;
        case TQ_GGML_TYPE_Q8_0:
            fused_dot = fused_dot_q8_0;
            break;
        case TQ_GGML_TYPE_Q2_K:
            fused_dot = fused_dot_q2_k;
            break;
        case TQ_GGML_TYPE_Q8_1:
            fused_dot = fused_dot_q8_1;
            break;
        case TQ_GGML_TYPE_Q3_K:
            fused_dot = fused_dot_q3_k;
            break;
        case TQ_GGML_TYPE_Q4_K:
            fused_dot = fused_dot_q4_k;
            break;
        case TQ_GGML_TYPE_Q4_0:
            fused_dot = fused_dot_q4_0;
            break;
        case TQ_GGML_TYPE_Q6_K:
            fused_dot = fused_dot_q6_k;
            break;
        case TQ_GGML_TYPE_IQ3_XXS:
            fused_dot = fused_dot_iq3_xxs;
            break;
        case TQ_GGML_TYPE_IQ3_S:
            fused_dot = fused_dot_iq3_s;
            break;
        case TQ_GGML_TYPE_IQ4_NL:
            fused_dot = fused_dot_iq4_nl;
            break;
        case TQ_GGML_TYPE_IQ4_XS:
            fused_dot = fused_dot_iq4_xs;
            break;
        default:
            break;
    }

    /* ---- Multi-threaded dispatch ---- */
    int n_threads = tq_get_threads();

    /* Per-thread override: when caller is already inside a parallel region
     * (e.g. cross-expert MoE parallelism), force single-threaded matmul
     * to avoid nested thread-pool contention. */
    extern __thread int tq_tls_force_serial_matmul;
    if (tq_tls_force_serial_matmul) n_threads = 1;

    /* Note: single-thread for small matmuls was tested and was SLOWER
     * (538ms vs 251ms MoE). Multi-threading benefits IQ2_XXS fused dot
     * even at out_dim=512. Keep multi-threaded. */

    /* For small matrices or single-thread config, skip thread overhead */
    if (n_threads <= 1 || out_dim < n_threads) {
        /* Single-threaded path */
        if (fused_dot) {
            for (int d = 0; d < out_dim; d++) {
                const uint8_t* row = (const uint8_t*)weight + (size_t)d * row_bytes;
                out[d] = fused_dot(row, x, in_dim);
            }
        } else {
            gguf_matmul_task_t task = {
                .out = out, .x = x, .weight = weight, .fused_dot = NULL,
                .weight_type = weight_type, .row_bytes = row_bytes,
                .in_dim = in_dim, .block_bytes = (int)block_bytes,
                .block_elems = block_elems, .n_blocks = n_blocks,
                .start_row = 0, .end_row = out_dim
            };
            gguf_matmul_worker(&task);
        }
        return;
    }

    /* Cap threads */
    if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
    if (n_threads > out_dim) n_threads = out_dim;

    gguf_matmul_task_t tasks[TQ_TP_MAX];
    void* ptrs[TQ_TP_MAX];

    int rows_per_thread = out_dim / n_threads;
    for (int t = 0; t < n_threads; t++) {
        tasks[t].out         = out;
        tasks[t].x           = x;
        tasks[t].weight      = weight;
        tasks[t].fused_dot   = fused_dot;
        tasks[t].weight_type = weight_type;
        tasks[t].row_bytes   = row_bytes;
        tasks[t].in_dim      = in_dim;
        tasks[t].block_bytes = (int)block_bytes;
        tasks[t].block_elems = block_elems;
        tasks[t].n_blocks    = n_blocks;
        tasks[t].start_row   = t * rows_per_thread;
        tasks[t].end_row     = (t == n_threads - 1) ? out_dim : (t + 1) * rows_per_thread;
        ptrs[t] = &tasks[t];
    }

    tq_tp_run(gguf_matmul_worker, ptrs, n_threads);
}

void tq_matmul_gguf_cpu(float* out, const float* x,
                         const void* weight, tq_ggml_dtype weight_type,
                         int out_dim, int in_dim)
{
    /* Save-and-restore, not hard-reset. A caller (e.g., tq_forward for
     * Phi-3) may have already set the flag to 1 as an invariant for the
     * entire forward pass; hard-resetting to 0 here would let subsequent
     * matmuls in the same forward dispatch to Metal and produce garbage. */
    int prev = tq_matmul_force_cpu;
    tq_matmul_force_cpu = 1;
    tq_matmul_gguf(out, x, weight, weight_type, out_dim, in_dim);
    tq_matmul_force_cpu = prev;
}

/* ============================================================
 * Metal batch mode wrappers
 *
 * These forward to Metal batch API when available, otherwise no-op.
 * The transformer/MoE code calls these to batch consecutive matmuls
 * into a single GPU command buffer, reducing dispatch overhead.
 * ============================================================ */

void tq_metal_batch_begin_if_available(void) {
#ifdef TQ_HAS_METAL
    extern int tq_metal_available(void);
    extern void tq_metal_batch_begin(void);
    if (tq_metal_available()) {
        tq_metal_batch_begin();
    }
#endif
}

void tq_metal_batch_flush_if_available(void) {
#ifdef TQ_HAS_METAL
    extern void tq_metal_batch_flush(void);
    extern int tq_metal_batch_active(void);
    if (tq_metal_batch_active()) {
        tq_metal_batch_flush();
    }
#endif
}

void tq_metal_batch_end_if_available(void) {
#ifdef TQ_HAS_METAL
    extern void tq_metal_batch_end(void);
    extern int tq_metal_batch_active(void);
    if (tq_metal_batch_active()) {
        tq_metal_batch_end();
    }
#endif
}

/* ============================================================
 * Public batched matmul wrappers for MoE prefill (Mission A Step 3d).
 *
 * Input layout:
 *   x   [N, in_dim]    FP32 row-major (activations for N tokens)
 *   out [N, out_dim]   FP32 row-major
 *   weight = mmap'd quantized block array of out_dim rows
 *
 * Internally: pre-quantize X to int8 (per-32 Q8-style scale) ONCE, then
 * call the per-type batched NEON kernel which amortizes weight unpack
 * across the N activations.
 *
 * Returns 0 on success, -1 if not available (falls back via tq_matmul_gguf).
 * ============================================================ */

#if TQ_HAS_NEON

/* Pre-quantize X[N, in_dim] to int8 with per-32-elem scales.
 * Returns allocated buffers (caller frees). Returns 0 on success. */
static int _quantize_x_batch_i8(const float* x, int in_dim, int N,
                                 int8_t** out_qs, float** out_ds)
{
    int n_blocks = in_dim / 32;
    int8_t* X_qs = (int8_t*)malloc((size_t)N * (size_t)in_dim * sizeof(int8_t));
    float*  X_ds = (float*)malloc((size_t)N * (size_t)n_blocks * sizeof(float));
    if (!X_qs || !X_ds) { free(X_qs); free(X_ds); return -1; }

    for (int n = 0; n < N; n++) {
        const float* xp = x + (size_t)n * in_dim;
        int8_t* qs_row = X_qs + (size_t)n * in_dim;
        float*  ds_row = X_ds + (size_t)n * n_blocks;
        for (int bb = 0; bb < n_blocks; bb++) {
            const float* block = xp + bb * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float a = block[j] < 0 ? -block[j] : block[j];
                if (a > amax) amax = a;
            }
            float dq = amax / 127.0f;
            ds_row[bb] = dq;
            if (dq > 0.0f) {
                float id = 1.0f / dq;
                for (int j = 0; j < 32; j++) {
                    int v = (int)roundf(block[j] * id);
                    qs_row[bb * 32 + j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
                }
            } else {
                memset(qs_row + bb * 32, 0, 32);
            }
        }
    }
    *out_qs = X_qs; *out_ds = X_ds;
    return 0;
}
#endif

extern void tq_tp_run(void* (*fn)(void*), void** args, int n);
extern int tq_get_threads(void);

/* ---- IQ3_XXS batched wrapper ---- */
int tq_batched_matmul_iq3_xxs(float* out, const void* weight,
                               const float* x, int out_dim, int in_dim, int N)
{
#if TQ_HAS_NEON
    if (in_dim < 256 || (in_dim % 256 != 0) || N <= 0 || N > 64) return -1;
    int8_t* X_qs = NULL; float* X_ds = NULL;
    if (_quantize_x_batch_i8(x, in_dim, N, &X_qs, &X_ds) != 0) return -1;
    int n_super = in_dim / 256;
    size_t row_bytes = (size_t)n_super * 98;

    /* Threading across rows */
    int n_threads = tq_get_threads();
    if (n_threads > out_dim) n_threads = out_dim;
    if (n_threads < 1) n_threads = 1;

    if (n_threads <= 1) {
        fused_dot_iq3_xxs_int8_batched(out, weight, row_bytes, out_dim,
                                       X_qs, X_ds, 0, out_dim, n_super, N);
    } else {
        /* Simple row split: each thread handles a contiguous slice */
        typedef struct {
            float* out; const void* weight; size_t row_bytes; int out_row_stride;
            const int8_t* X_qs; const float* X_ds; int start; int end;
            int n_super; int N;
        } task_t;
        /* Split manually, dispatch via tq_tp_run */
        #define MOE_TP_MAX 16
        task_t tasks[MOE_TP_MAX];
        void* ptrs[MOE_TP_MAX];
        if (n_threads > MOE_TP_MAX) n_threads = MOE_TP_MAX;
        int per = out_dim / n_threads;
        for (int t = 0; t < n_threads; t++) {
            tasks[t].out = out;
            tasks[t].weight = weight;
            tasks[t].row_bytes = row_bytes;
            tasks[t].out_row_stride = out_dim;
            tasks[t].X_qs = X_qs;
            tasks[t].X_ds = X_ds;
            tasks[t].start = t * per;
            tasks[t].end = (t == n_threads - 1) ? out_dim : (t + 1) * per;
            tasks[t].n_super = n_super;
            tasks[t].N = N;
            ptrs[t] = &tasks[t];
        }
        /* Worker stub: use inline static — but function ptr can't capture.
         * Fall back to serial here to keep this patch small; threading is done
         * at a higher level in the MoE loop. */
        (void)ptrs;
        fused_dot_iq3_xxs_int8_batched(out, weight, row_bytes, out_dim,
                                       X_qs, X_ds, 0, out_dim, n_super, N);
    }
    free(X_qs); free(X_ds);
    return 0;
#else
    (void)out; (void)weight; (void)x; (void)out_dim; (void)in_dim; (void)N;
    return -1;
#endif
}

/* ---- IQ3_S batched wrapper ---- */
int tq_batched_matmul_iq3_s(float* out, const void* weight,
                             const float* x, int out_dim, int in_dim, int N)
{
#if TQ_HAS_NEON
    if (in_dim < 256 || (in_dim % 256 != 0) || N <= 0 || N > 64) return -1;
    int8_t* X_qs = NULL; float* X_ds = NULL;
    if (_quantize_x_batch_i8(x, in_dim, N, &X_qs, &X_ds) != 0) return -1;
    int n_super = in_dim / 256;
    size_t row_bytes = (size_t)n_super * sizeof(block_iq3_s_t);

    fused_dot_iq3_s_int8_batched(out, weight, row_bytes, out_dim,
                                 X_qs, X_ds, 0, out_dim, n_super, N);
    free(X_qs); free(X_ds);
    return 0;
#else
    (void)out; (void)weight; (void)x; (void)out_dim; (void)in_dim; (void)N;
    return -1;
#endif
}

/* ---- IQ4_XS batched wrapper ---- */
int tq_batched_matmul_iq4_xs(float* out, const void* weight,
                              const float* x, int out_dim, int in_dim, int N)
{
#if TQ_HAS_NEON
    if (in_dim < 256 || (in_dim % 256 != 0) || N <= 0 || N > 64) return -1;
    int8_t* X_qs = NULL; float* X_ds = NULL;
    if (_quantize_x_batch_i8(x, in_dim, N, &X_qs, &X_ds) != 0) return -1;
    int n_super = in_dim / 256;
    size_t row_bytes = (size_t)n_super * sizeof(block_iq4_xs);

    fused_dot_iq4_xs_int8_batched(out, weight, row_bytes, out_dim,
                                  X_qs, X_ds, 0, out_dim, n_super, N);
    free(X_qs); free(X_ds);
    return 0;
#else
    (void)out; (void)weight; (void)x; (void)out_dim; (void)in_dim; (void)N;
    return -1;
#endif
}

/* ---- Q3_K batched wrapper ----
 * Exposed via moe_batched_dispatch in tq_moe.c for Q3_K_S routed experts.
 * Uses per-32 int8 activation quantization (same scheme as IQ3/IQ4 batched). */
int tq_batched_matmul_q3_k(float* out, const void* weight,
                            const float* x, int out_dim, int in_dim, int N)
{
#if TQ_HAS_NEON
    if (in_dim < 256 || (in_dim % 256 != 0) || N <= 0 || N > 64) return -1;
    int8_t* X_qs = NULL; float* X_ds = NULL;
    if (_quantize_x_batch_i8(x, in_dim, N, &X_qs, &X_ds) != 0) return -1;
    int n_super = in_dim / 256;
    size_t row_bytes = (size_t)n_super * sizeof(block_q3_K);

    fused_dot_q3_k_int8_batched(out, weight, row_bytes, out_dim,
                                 X_qs, X_ds, 0, out_dim, n_super, N);
    free(X_qs); free(X_ds);
    return 0;
#else
    (void)out; (void)weight; (void)x; (void)out_dim; (void)in_dim; (void)N;
    return -1;
#endif
}
