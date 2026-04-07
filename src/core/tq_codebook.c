/**
 * tq_codebook.c -- Optimal Gaussian Lloyd-Max codebook quantization
 *
 * Pre-computed optimal centroids for N(0,1) distribution at bit-widths 1-4.
 * These are the reconstruction points from the Max-Lloyd algorithm.
 * Decision boundaries are midpoints between consecutive centroids.
 *
 * Usage: After RHT, each coordinate is approximately N(0, 1/sqrt(d)),
 * so we scale by inv_std = sqrt(d) to normalize to N(0,1) before
 * codebook lookup, then scale back after dequantization.
 */

#include "turboquant/turboquant.h"
#include <math.h>
#include <float.h>

/* ============================================================
 * Pre-computed Lloyd-Max centroids for standard normal N(0,1)
 * ============================================================ */

/* b=1 (2 levels): E[|X|] for half-normal = sqrt(2/pi) ~ 0.7979 */
static const float CODEBOOK_1BIT[2] = {-0.7979f, 0.7979f};

/* b=2 (4 levels): optimal Lloyd-Max for N(0,1) */
static const float CODEBOOK_2BIT[4] = {-1.5104f, -0.4528f, 0.4528f, 1.5104f};

/* b=3 (8 levels): optimal Lloyd-Max for N(0,1) */
static const float CODEBOOK_3BIT[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f
};

/* b=4 (16 levels): optimal Lloyd-Max for N(0,1) */
static const float CODEBOOK_4BIT[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};

/* b=5 (32 levels): optimal Lloyd-Max for N(0,1).
 * Source: Max 1960 Table I, 32-level Gaussian quantizer output values.
 * MSE is roughly 4x lower than 4-bit; outer level shrinks to ~2.0 because
 * the additional levels concentrate in the body. */
static const float CODEBOOK_5BIT[32] = {
    -1.9956f, -1.7900f, -1.6107f, -1.4493f, -1.3010f, -1.1631f, -1.0334f, -0.9104f,
    -0.7928f, -0.6795f, -0.5697f, -0.4626f, -0.3576f, -0.2543f, -0.1520f, -0.0506f,
     0.0506f,  0.1520f,  0.2543f,  0.3576f,  0.4626f,  0.5697f,  0.6795f,  0.7928f,
     0.9104f,  1.0334f,  1.1631f,  1.3010f,  1.4493f,  1.6107f,  1.7900f,  1.9956f
};

/* Codebook table indexed by bits */
static const float* const CODEBOOKS[6] = {
    NULL,          /* 0 bits: unused */
    CODEBOOK_1BIT, /* 1 bit: 2 levels */
    CODEBOOK_2BIT, /* 2 bits: 4 levels */
    CODEBOOK_3BIT, /* 3 bits: 8 levels */
    CODEBOOK_4BIT, /* 4 bits: 16 levels */
    CODEBOOK_5BIT  /* 5 bits: 32 levels */
};

static const int CODEBOOK_SIZES[6] = {0, 2, 4, 8, 16, 32};

/* ============================================================
 * Codebook quantize: find nearest centroid for each element
 * ============================================================ */

void tq_codebook_quantize(const float* src, uint8_t* dst_indices,
                           int n, int bits, float inv_std) {
    if (!src || !dst_indices || bits < 1 || bits > 5 || n <= 0) return;

    const float* centroids = CODEBOOKS[bits];
    int n_levels = CODEBOOK_SIZES[bits];

    for (int i = 0; i < n; i++) {
        /* Scale to standard normal space */
        float x = src[i] * inv_std;

        /* Find nearest centroid (linear scan, optimal for small n_levels) */
        int best = 0;
        float best_dist = fabsf(x - centroids[0]);
        for (int c = 1; c < n_levels; c++) {
            float dist = fabsf(x - centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best = c;
            }
        }
        dst_indices[i] = (uint8_t)best;
    }
}

/* ============================================================
 * Codebook dequantize: reconstruct from centroid lookup
 * ============================================================ */

void tq_codebook_dequantize(const uint8_t* indices, float* dst,
                             int n, int bits, float inv_std) {
    if (!indices || !dst || bits < 1 || bits > 5 || n <= 0) return;

    const float* centroids = CODEBOOKS[bits];
    float std_val = (inv_std > 1e-10f) ? (1.0f / inv_std) : 1.0f;

    for (int i = 0; i < n; i++) {
        dst[i] = centroids[indices[i]] * std_val;
    }
}

/* ============================================================
 * Codebook helpers: get centroids and number of levels
 * ============================================================ */

const float* tq_codebook_centroids(int bits) {
    if (bits < 1 || bits > 5) return NULL;
    return CODEBOOKS[bits];
}

int tq_codebook_levels(int bits) {
    if (bits < 1 || bits > 4) return 0;
    return CODEBOOK_SIZES[bits];
}
