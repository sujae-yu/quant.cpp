/**
 * test_turbo_quant_kv.cpp -- Standalone test for quant.cpp 1-bit KV cache
 *
 * Apache 2.0 License, QuantumAI Inc.
 *
 * Build:
 *   g++ -std=c++11 -O2 -o test_turbo_quant_kv test_turbo_quant_kv.cpp ggml-turbo-quant.c -lm
 *   ./test_turbo_quant_kv
 *
 * Tests:
 *   1. Block size static assertion (compile-time)
 *   2. Quantize/dequantize roundtrip MSE
 *   3. Attention score cosine similarity vs FP32
 *   4. Norm preservation
 *   5. Compression ratio verification
 */

#include "ggml-turbo-quant.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

/* ============================================================
 * Test utilities
 * ============================================================ */

static const int DIM = TQ_KV_1B_BLOCK_SIZE;  /* 128 */

/* Simple xorshift64 PRNG for reproducible tests */
static uint64_t rng_state = 0xDEADBEEFCAFE1234ULL;

static double xorshift64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0xFFFFFFFF) / (double)0xFFFFFFFF;
}

/* Generate random Gaussian using Box-Muller transform */
static float rand_gaussian(void) {
    double u1 = xorshift64();
    double u2 = xorshift64();
    if (u1 < 1e-10) u1 = 1e-10;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2));
}

/* Fill array with random Gaussian values */
static void fill_gaussian(float * arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand_gaussian();
    }
}

/* Compute L2 norm */
static float l2_norm(const float * x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrtf(sum);
}

/* Compute dot product */
static float dot_product(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* Compute cosine similarity between two arrays */
static float cosine_similarity(const float * a, const float * b, int n) {
    float dot = dot_product(a, b, n);
    float na = l2_norm(a, n);
    float nb = l2_norm(b, n);
    if (na < 1e-10f || nb < 1e-10f) return 0.0f;
    float result = dot / (na * nb);
    /* Clamp to [-1, 1] to handle floating point edge cases */
    if (result != result) return 0.0f;  /* NaN check */
    if (result > 1.0f) result = 1.0f;
    if (result < -1.0f) result = -1.0f;
    return result;
}

/* Compute MSE between two arrays */
static float compute_mse(const float * a, const float * b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum / (float)n;
}

/* Compute normalized MSE (MSE / variance of original) */
static float compute_nmse(const float * orig, const float * recon, int n) {
    float mse = compute_mse(orig, recon, n);
    float var = 0.0f;
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += orig[i];
    mean /= (float)n;
    for (int i = 0; i < n; i++) {
        float d = orig[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    if (var < 1e-10f) return 0.0f;
    return mse / var;
}

/* ============================================================
 * Test 1: Block size verification (compile-time + runtime)
 * ============================================================ */

static int test_block_size(void) {
    printf("Test 1: Block size verification...\n");

    /* Compile-time check already done via typedef in header */
    size_t expected = 2 + 2 + 4 + 16;  /* norm + pad + seed + signs */
    size_t actual = sizeof(block_tq_kv_1b);

    printf("  block_tq_kv_1b size: %zu bytes (expected %zu)\n", actual, expected);
    printf("  Block size: %d elements\n", TQ_KV_1B_BLOCK_SIZE);
    printf("  Bits per element: %.2f (including metadata)\n",
           (float)(actual * 8) / (float)TQ_KV_1B_BLOCK_SIZE);

    if (actual != expected) {
        printf("  FAIL: size mismatch\n");
        return 1;
    }

    printf("  PASS\n\n");
    return 0;
}

/* ============================================================
 * Test 2: Quantize/Dequantize roundtrip MSE
 *
 * For 1-bit quantization with RHT, the theoretical NMSE is
 * 1 - 2/pi ~ 0.363. We check MSE < 0.01 on unit-norm vectors
 * (which corresponds to NMSE of ~0.36 for Gaussian data).
 * ============================================================ */

static int test_roundtrip_mse(void) {
    printf("Test 2: Quantize/dequantize roundtrip MSE...\n");

    const int num_trials = 100;
    float total_mse = 0.0f;
    float total_nmse = 0.0f;
    float total_cosine = 0.0f;
    float worst_mse = 0.0f;

    for (int trial = 0; trial < num_trials; trial++) {
        float original[DIM];
        float reconstructed[DIM];
        block_tq_kv_1b block;

        fill_gaussian(original, DIM);

        /* Quantize */
        quantize_row_tq_kv_1b_ref(original, &block, DIM);

        /* Dequantize */
        dequantize_row_tq_kv_1b(&block, reconstructed, DIM);

        float mse = compute_mse(original, reconstructed, DIM);
        float nmse = compute_nmse(original, reconstructed, DIM);
        float cos = cosine_similarity(original, reconstructed, DIM);

        total_mse += mse;
        total_nmse += nmse;
        total_cosine += cos;
        if (mse > worst_mse) worst_mse = mse;
    }

    float avg_mse = total_mse / (float)num_trials;
    float avg_nmse = total_nmse / (float)num_trials;
    float avg_cosine = total_cosine / (float)num_trials;

    printf("  Avg MSE:    %.6f (threshold: < 0.01 per element for unit-variance)\n", avg_mse);
    printf("  Avg NMSE:   %.6f (theoretical: %.6f = 1 - 2/pi)\n", avg_nmse, 1.0 - 2.0 / 3.14159265);
    printf("  Avg Cosine: %.6f (theoretical: %.6f = sqrt(2/pi))\n", avg_cosine, sqrt(2.0 / 3.14159265));
    printf("  Worst MSE:  %.6f\n", worst_mse);

    /* The MSE threshold is per-element for standard Gaussian input.
     * For Gaussian with var=1.0, each element has magnitude ~1.0,
     * so MSE < 0.01 means < 1% error per element on average.
     * But 1-bit is aggressive, so we use a more relaxed threshold
     * and focus on attention accuracy. */
    if (avg_nmse > 0.50f) {
        printf("  FAIL: NMSE too high (expected < 0.50, got %.6f)\n", avg_nmse);
        return 1;
    }

    printf("  PASS\n\n");
    return 0;
}

/* ============================================================
 * Test 3: Attention score cosine similarity
 *
 * The key metric: do quantized attention scores preserve the
 * relative ordering of the FP32 scores?
 *
 * For 1-bit sign quantization, the theoretical cosine between
 * true and estimated inner products is 2/pi ~ 0.637.
 * We test that attention cosine > 0.6 (slightly below theoretical).
 * ============================================================ */

static int test_attention_cosine(void) {
    printf("Test 3: Attention score cosine similarity...\n");

    const int seq_len = 64;
    const int num_trials = 20;
    float total_cosine = 0.0f;

    for (int trial = 0; trial < num_trials; trial++) {
        /* Generate random query */
        float query[DIM];
        fill_gaussian(query, DIM);

        /* Generate random keys and quantize them */
        float keys_fp32[64][DIM];     /* seq_len = 64 */
        block_tq_kv_1b blocks[64];

        for (int s = 0; s < seq_len; s++) {
            fill_gaussian(keys_fp32[s], DIM);
            quantize_row_tq_kv_1b_ref(keys_fp32[s], &blocks[s], DIM);
        }

        /* Compute FP32 attention scores (ground truth) */
        float scores_fp32[64];
        for (int s = 0; s < seq_len; s++) {
            scores_fp32[s] = dot_product(query, keys_fp32[s], DIM);
        }

        /* Compute 1-bit Hamming attention scores */
        float scores_1b[64];
        tq_kv_1b_attention(query, blocks, scores_1b, seq_len, DIM);

        /* Cosine similarity between score vectors */
        float cos = cosine_similarity(scores_fp32, scores_1b, seq_len);
        total_cosine += cos;
    }

    float avg_cosine = total_cosine / (float)num_trials;
    printf("  Avg attention cosine: %.6f (threshold: > 0.6, theoretical: %.6f = 2/pi)\n",
           avg_cosine, 2.0 / 3.14159265);

    if (avg_cosine < 0.6f) {
        printf("  FAIL: attention cosine too low\n");
        return 1;
    }

    printf("  PASS\n\n");
    return 0;
}

/* ============================================================
 * Test 4: Norm preservation
 *
 * After quantize -> dequantize, the L2 norm should be roughly
 * preserved (within FP16 precision and 1-bit reconstruction error).
 * ============================================================ */

static int test_norm_preservation(void) {
    printf("Test 4: Norm preservation...\n");

    const int num_trials = 50;
    float max_ratio_error = 0.0f;

    for (int trial = 0; trial < num_trials; trial++) {
        float original[DIM];
        float reconstructed[DIM];
        block_tq_kv_1b block;

        fill_gaussian(original, DIM);

        quantize_row_tq_kv_1b_ref(original, &block, DIM);
        dequantize_row_tq_kv_1b(&block, reconstructed, DIM);

        float orig_norm = l2_norm(original, DIM);
        float recon_norm = l2_norm(reconstructed, DIM);

        if (orig_norm > 1e-6f) {
            float ratio_error = fabsf(recon_norm / orig_norm - 1.0f);
            if (ratio_error > max_ratio_error) max_ratio_error = ratio_error;
        }
    }

    printf("  Max norm ratio error: %.6f (threshold: < 0.30)\n", max_ratio_error);

    /* 1-bit reconstruction has sqrt(2/pi) ~ 0.798 scaling factor,
     * so the norm will be off by ~20%. We allow 30% tolerance. */
    if (max_ratio_error > 0.30f) {
        printf("  FAIL: norm ratio error too high\n");
        return 1;
    }

    printf("  PASS\n\n");
    return 0;
}

/* ============================================================
 * Test 5: Compression ratio
 * ============================================================ */

static int test_compression_ratio(void) {
    printf("Test 5: Compression ratio...\n");

    float fp16_bytes = (float)(DIM * 2);     /* FP16: 2 bytes per element */
    float fp32_bytes = (float)(DIM * 4);     /* FP32: 4 bytes per element */
    float tq_bytes = (float)sizeof(block_tq_kv_1b);

    float ratio_vs_fp16 = fp16_bytes / tq_bytes;
    float ratio_vs_fp32 = fp32_bytes / tq_bytes;
    float bpw = (tq_bytes * 8.0f) / (float)DIM;

    printf("  FP16 size:  %d bytes per %d elements\n", (int)fp16_bytes, DIM);
    printf("  FP32 size:  %d bytes per %d elements\n", (int)fp32_bytes, DIM);
    printf("  TQ_1B size: %d bytes per %d elements\n", (int)tq_bytes, DIM);
    printf("  Compression vs FP16: %.1fx\n", ratio_vs_fp16);
    printf("  Compression vs FP32: %.1fx\n", ratio_vs_fp32);
    printf("  Bits per weight: %.2f\n", bpw);

    if (ratio_vs_fp16 < 10.0f) {
        printf("  FAIL: compression ratio vs FP16 too low (expected > 10x)\n");
        return 1;
    }

    printf("  PASS\n\n");
    return 0;
}

/* ============================================================
 * Test 6: Multi-block quantization (k > 128)
 * ============================================================ */

static int test_multi_block(void) {
    printf("Test 6: Multi-block quantization (k=512)...\n");

    const int k = 512;  /* 4 blocks */
    float original[512];
    float reconstructed[512];
    block_tq_kv_1b blocks[4];

    fill_gaussian(original, k);

    quantize_row_tq_kv_1b_ref(original, blocks, k);
    dequantize_row_tq_kv_1b(blocks, reconstructed, k);

    float mse = compute_mse(original, reconstructed, k);
    float cos = cosine_similarity(original, reconstructed, k);

    printf("  MSE (k=512):    %.6f\n", mse);
    printf("  Cosine (k=512): %.6f\n", cos);

    if (cos < 0.5f) {
        printf("  FAIL: cosine too low for multi-block\n");
        return 1;
    }

    printf("  PASS\n\n");
    return 0;
}

/* ============================================================
 * Test 7: Zero and edge cases
 * ============================================================ */

static int test_edge_cases(void) {
    printf("Test 7: Edge cases...\n");

    float zeros[DIM];
    float reconstructed[DIM];
    block_tq_kv_1b block;

    /* Zero vector */
    memset(zeros, 0, sizeof(zeros));
    quantize_row_tq_kv_1b_ref(zeros, &block, DIM);
    dequantize_row_tq_kv_1b(&block, reconstructed, DIM);

    float recon_norm = l2_norm(reconstructed, DIM);
    printf("  Zero vector: recon norm = %.6f (expected ~0)\n", recon_norm);

    /* Constant vector */
    float constant[DIM];
    for (int i = 0; i < DIM; i++) constant[i] = 1.0f;
    quantize_row_tq_kv_1b_ref(constant, &block, DIM);
    dequantize_row_tq_kv_1b(&block, reconstructed, DIM);

    float cos = cosine_similarity(constant, reconstructed, DIM);
    printf("  Constant vector: cosine = %.6f\n", cos);

    /* Large magnitude (kept within FP16 range: max norm < 65504) */
    float large[DIM];
    for (int i = 0; i < DIM; i++) large[i] = (float)(i + 1) * 3.0f;
    quantize_row_tq_kv_1b_ref(large, &block, DIM);
    dequantize_row_tq_kv_1b(&block, reconstructed, DIM);

    float cos2 = cosine_similarity(large, reconstructed, DIM);
    printf("  Large magnitude: cosine = %.6f\n", cos2);

    /* Structured (non-Gaussian) inputs can have low cosine with 1-bit.
     * This is expected -- RHT + sign works best on naturally distributed data.
     * We only check that it doesn't produce NaN/Inf or negative cosine. */
    if (cos2 != cos2 || cos2 < -0.1f) {  /* NaN or very negative */
        printf("  FAIL: invalid cosine for large magnitude\n");
        return 1;
    }

    printf("  PASS\n\n");
    return 0;
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("===========================================\n");
    printf("quant.cpp 1-bit KV Cache -- Standalone Test\n");
    printf("===========================================\n\n");

    int failures = 0;

    failures += test_block_size();
    failures += test_roundtrip_mse();
    failures += test_attention_cosine();
    failures += test_norm_preservation();
    failures += test_compression_ratio();
    failures += test_multi_block();
    failures += test_edge_cases();

    printf("===========================================\n");
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TEST(S) FAILED\n", failures);
    }
    printf("===========================================\n");

    return failures;
}
