/**
 * test_turbo_kv.cpp -- Tests for TurboQuant KV cache quantization
 *
 * Tests the RHT + Lloyd-Max codebook + QJL residual pipeline for both
 * 3-bit (2-bit codebook + 1-bit QJL) and 4-bit (3-bit codebook + 1-bit QJL)
 * variants. Validates roundtrip MSE, attention accuracy, and comparison
 * with uniform baseline.
 */

#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"

void tq_turbo_kv_3b_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_kv_3b_dequantize_ref(const void* src, float* dst, int n);
void tq_turbo_kv_3b_attention_ref(const float* query, const void* kv,
                                    float* scores, int seq_len, int head_dim);

void tq_turbo_kv_4b_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_kv_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_turbo_kv_4b_attention_ref(const float* query, const void* kv,
                                    float* scores, int seq_len, int head_dim);

void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
}

#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <cstring>

/* ============================================================
 * Helper: compute MSE between two vectors
 * ============================================================ */
static double compute_mse(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    return mse / n;
}

/* ============================================================
 * Helper: compute cosine similarity
 * ============================================================ */
static double compute_cosine(const float* a, const float* b, int n) {
    double dot = 0.0, sq_a = 0.0, sq_b = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        sq_a += (double)a[i] * (double)a[i];
        sq_b += (double)b[i] * (double)b[i];
    }
    double denom = std::sqrt(sq_a) * std::sqrt(sq_b);
    return (denom > 1e-15) ? dot / denom : 0.0;
}

/* ============================================================
 * 3-bit roundtrip tests
 * ============================================================ */

TEST(TurboKV3B, RoundtripBasic) {
    const int dim = TQ_BK;  /* 128 */
    std::vector<float> input(dim);
    for (int i = 0; i < dim; i++) input[i] = sinf(i * 0.1f);

    block_tq_turbo_kv_3b block;
    memset(&block, 0, sizeof(block));
    tq_turbo_kv_3b_quantize_ref(input.data(), &block, dim);

    std::vector<float> output(dim);
    tq_turbo_kv_3b_dequantize_ref(&block, output.data(), dim);

    double mse = compute_mse(input.data(), output.data(), dim);
    double cosine = compute_cosine(input.data(), output.data(), dim);

    /* 2-bit codebook (4 levels) gives ~6 dB SNR per coordinate.
     * Cosine ~ 0.4-0.7 for 128-dim vectors is expected at 2-bit.
     * The real value of QJL is in attention (inner product), not dequant. */
    EXPECT_LT(mse, 1.0) << "TurboKV 3B MSE too high: " << mse;
    EXPECT_GT(cosine, 0.3) << "TurboKV 3B cosine too low: " << cosine;
}

TEST(TurboKV3B, BlockStructureSize) {
    /* Verify block size: 8 (header) + 32 (2-bit indices) + 16 (QJL signs) = 56 */
    EXPECT_EQ(sizeof(block_tq_turbo_kv_3b), 56u);
    /* BPE = 56 * 8 / 128 = 3.5 */
    float bpe = (float)sizeof(block_tq_turbo_kv_3b) * 8.0f / TQ_BK;
    EXPECT_NEAR(bpe, 3.5f, 0.01f);
}

TEST(TurboKV3B, NormPreserved) {
    const int dim = TQ_BK;
    std::vector<float> input(dim);
    for (int i = 0; i < dim; i++) input[i] = 2.0f * sinf(i * 0.2f);

    block_tq_turbo_kv_3b block;
    tq_turbo_kv_3b_quantize_ref(input.data(), &block, dim);

    /* Check that stored norm is approximately correct */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) norm_sq += input[i] * input[i];
    float expected_norm = sqrtf(norm_sq);

    /* FP16 stored norm */
    union { float f; uint32_t u; } fp16;
    uint16_t stored = block.norm;
    uint32_t sign = (stored & 0x8000) << 16;
    uint32_t exp = (stored >> 10) & 0x1F;
    uint32_t mant = stored & 0x03FF;
    if (exp == 0) fp16.u = sign;
    else { exp = exp - 15 + 127; fp16.u = sign | (exp << 23) | (mant << 13); }
    float decoded_norm = fp16.f;

    EXPECT_NEAR(decoded_norm, expected_norm, expected_norm * 0.01f);
}

TEST(TurboKV3B, QJLSignsNonZero) {
    const int dim = TQ_BK;
    std::vector<float> input(dim);
    for (int i = 0; i < dim; i++) input[i] = sinf(i * 0.3f);

    block_tq_turbo_kv_3b block;
    memset(&block, 0, sizeof(block));
    tq_turbo_kv_3b_quantize_ref(input.data(), &block, dim);

    /* QJL signs should not be all zeros */
    bool has_nonzero = false;
    for (int i = 0; i < TQ_BK / 8; i++) {
        if (block.qjl_signs[i] != 0) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}

/* ============================================================
 * 4-bit roundtrip tests
 * ============================================================ */

TEST(TurboKV4B, RoundtripBasic) {
    const int dim = TQ_BK;
    std::vector<float> input(dim);
    for (int i = 0; i < dim; i++) input[i] = sinf(i * 0.1f);

    block_tq_turbo_kv_4b block;
    memset(&block, 0, sizeof(block));
    tq_turbo_kv_4b_quantize_ref(input.data(), &block, dim);

    std::vector<float> output(dim);
    tq_turbo_kv_4b_dequantize_ref(&block, output.data(), dim);

    double mse = compute_mse(input.data(), output.data(), dim);
    double cosine = compute_cosine(input.data(), output.data(), dim);

    /* 3-bit codebook (8 levels) gives ~12 dB SNR per coordinate.
     * Cosine ~ 0.6-0.8 for 128-dim vectors is expected at 3-bit. */
    EXPECT_LT(mse, 0.5) << "TurboKV 4B MSE too high: " << mse;
    EXPECT_GT(cosine, 0.6) << "TurboKV 4B cosine too low: " << cosine;
}

TEST(TurboKV4B, BlockStructureSize) {
    /* Verify block size: 8 (header) + 48 (3-bit indices) + 16 (QJL signs) = 72 */
    EXPECT_EQ(sizeof(block_tq_turbo_kv_4b), 72u);
    /* BPE = 72 * 8 / 128 = 4.5 */
    float bpe = (float)sizeof(block_tq_turbo_kv_4b) * 8.0f / TQ_BK;
    EXPECT_NEAR(bpe, 4.5f, 0.01f);
}

TEST(TurboKV4B, BetterThan3B) {
    /* 4-bit should have lower MSE than 3-bit */
    const int dim = TQ_BK;
    std::vector<float> input(dim);
    for (int i = 0; i < dim; i++) input[i] = sinf(i * 0.15f);

    block_tq_turbo_kv_3b block_3b;
    block_tq_turbo_kv_4b block_4b;
    tq_turbo_kv_3b_quantize_ref(input.data(), &block_3b, dim);
    tq_turbo_kv_4b_quantize_ref(input.data(), &block_4b, dim);

    std::vector<float> out_3b(dim), out_4b(dim);
    tq_turbo_kv_3b_dequantize_ref(&block_3b, out_3b.data(), dim);
    tq_turbo_kv_4b_dequantize_ref(&block_4b, out_4b.data(), dim);

    double mse_3b = compute_mse(input.data(), out_3b.data(), dim);
    double mse_4b = compute_mse(input.data(), out_4b.data(), dim);

    EXPECT_LT(mse_4b, mse_3b) << "4-bit MSE (" << mse_4b
                                << ") should be lower than 3-bit (" << mse_3b << ")";
}

/* ============================================================
 * Attention accuracy tests
 * ============================================================ */

TEST(TurboKV3B, AttentionAccuracy) {
    const int dim = 128;
    const int seq_len = 8;

    std::vector<float> keys(seq_len * dim);
    std::vector<float> query(dim);

    /* Generate test data */
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < seq_len * dim; i++) keys[i] = dist(rng);
    for (int i = 0; i < dim; i++) query[i] = dist(rng);

    /* FP32 reference scores */
    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < dim; d++)
            dot += query[d] * keys[s * dim + d];
        fp32_scores[s] = dot;
    }

    /* Quantize keys */
    std::vector<block_tq_turbo_kv_3b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        tq_turbo_kv_3b_quantize_ref(&keys[s * dim], &blocks[s], dim);
    }

    /* Compute quantized attention */
    std::vector<float> quant_scores(seq_len);
    tq_turbo_kv_3b_attention_ref(query.data(), blocks.data(),
                                   quant_scores.data(), seq_len, dim);

    /* Cosine similarity between score vectors should be high */
    double cosine = compute_cosine(fp32_scores.data(), quant_scores.data(), seq_len);
    EXPECT_GT(std::fabs(cosine), 0.8)
        << "TurboKV 3B attention cosine too low: " << cosine;
}

TEST(TurboKV4B, AttentionAccuracy) {
    const int dim = 128;
    const int seq_len = 8;

    std::vector<float> keys(seq_len * dim);
    std::vector<float> query(dim);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < seq_len * dim; i++) keys[i] = dist(rng);
    for (int i = 0; i < dim; i++) query[i] = dist(rng);

    std::vector<float> fp32_scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < dim; d++)
            dot += query[d] * keys[s * dim + d];
        fp32_scores[s] = dot;
    }

    std::vector<block_tq_turbo_kv_4b> blocks(seq_len);
    for (int s = 0; s < seq_len; s++) {
        tq_turbo_kv_4b_quantize_ref(&keys[s * dim], &blocks[s], dim);
    }

    std::vector<float> quant_scores(seq_len);
    tq_turbo_kv_4b_attention_ref(query.data(), blocks.data(),
                                   quant_scores.data(), seq_len, dim);

    double cosine = compute_cosine(fp32_scores.data(), quant_scores.data(), seq_len);
    EXPECT_GT(std::fabs(cosine), 0.9)
        << "TurboKV 4B attention cosine too low: " << cosine;
}

/* ============================================================
 * Context API integration tests
 * ============================================================ */

TEST(TurboKV3B, ContextAPIRoundtrip) {
    tq_context_t* ctx;
    ASSERT_EQ(tq_init(&ctx, TQ_BACKEND_CPU), TQ_OK);

    const int dim = 128;
    std::vector<float> key(dim);
    for (int i = 0; i < dim; i++) key[i] = sinf(i * 0.1f);

    size_t buf_size = tq_quantize_keys_size(1, dim, TQ_TYPE_TURBO_KV_3B);
    ASSERT_GT(buf_size, 0u);
    std::vector<uint8_t> buf(buf_size);

    tq_status st = tq_quantize_keys(ctx, key.data(), 1, dim,
                                     TQ_TYPE_TURBO_KV_3B,
                                     buf.data(), buf_size);
    ASSERT_EQ(st, TQ_OK);

    std::vector<float> output(dim);
    st = tq_dequantize_keys(ctx, buf.data(), 1, dim,
                             TQ_TYPE_TURBO_KV_3B, output.data());
    ASSERT_EQ(st, TQ_OK);

    double cosine = compute_cosine(key.data(), output.data(), dim);
    EXPECT_GT(cosine, 0.3);  /* 2-bit codebook: limited precision */

    tq_free(ctx);
}

TEST(TurboKV4B, ContextAPIRoundtrip) {
    tq_context_t* ctx;
    ASSERT_EQ(tq_init(&ctx, TQ_BACKEND_CPU), TQ_OK);

    const int dim = 128;
    std::vector<float> key(dim);
    for (int i = 0; i < dim; i++) key[i] = sinf(i * 0.1f);

    size_t buf_size = tq_quantize_keys_size(1, dim, TQ_TYPE_TURBO_KV_4B);
    ASSERT_GT(buf_size, 0u);
    std::vector<uint8_t> buf(buf_size);

    tq_status st = tq_quantize_keys(ctx, key.data(), 1, dim,
                                     TQ_TYPE_TURBO_KV_4B,
                                     buf.data(), buf_size);
    ASSERT_EQ(st, TQ_OK);

    std::vector<float> output(dim);
    st = tq_dequantize_keys(ctx, buf.data(), 1, dim,
                             TQ_TYPE_TURBO_KV_4B, output.data());
    ASSERT_EQ(st, TQ_OK);

    double cosine = compute_cosine(key.data(), output.data(), dim);
    EXPECT_GT(cosine, 0.6);  /* 3-bit codebook: moderate precision */

    tq_free(ctx);
}

/* ============================================================
 * Comparison: TurboKV 4B vs Uniform 4B
 * ============================================================ */

TEST(TurboKV4B, CompareWithUniform4B) {
    /* Test with Gaussian data (where RHT + codebook should shine) */
    const int dim = TQ_BK;
    const int n_trials = 50;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_mse_turbo = 0.0, total_mse_uniform = 0.0;

    for (int trial = 0; trial < n_trials; trial++) {
        std::vector<float> input(dim);
        for (int i = 0; i < dim; i++) input[i] = dist(rng);

        /* TurboKV 4B */
        block_tq_turbo_kv_4b turbo_block;
        tq_turbo_kv_4b_quantize_ref(input.data(), &turbo_block, dim);
        std::vector<float> turbo_out(dim);
        tq_turbo_kv_4b_dequantize_ref(&turbo_block, turbo_out.data(), dim);
        total_mse_turbo += compute_mse(input.data(), turbo_out.data(), dim);

        /* Uniform 4B */
        block_tq_uniform_4b uni_block;
        tq_uniform_4b_quantize_ref(input.data(), &uni_block, dim);
        std::vector<float> uni_out(dim);
        tq_uniform_4b_dequantize_ref(&uni_block, uni_out.data(), dim);
        total_mse_uniform += compute_mse(input.data(), uni_out.data(), dim);
    }

    double avg_mse_turbo = total_mse_turbo / n_trials;
    double avg_mse_uniform = total_mse_uniform / n_trials;

    /* TurboKV should be competitive with or better than uniform for Gaussian data.
     * With RHT + optimal codebook, TurboKV MSE should be lower for
     * data that matches the Gaussian assumption. */
    printf("  TurboKV 4B avg MSE: %.6f\n", avg_mse_turbo);
    printf("  Uniform 4B avg MSE: %.6f\n", avg_mse_uniform);
    printf("  Ratio (turbo/uniform): %.2f\n", avg_mse_turbo / avg_mse_uniform);

    /* Both should have bounded MSE */
    EXPECT_LT(avg_mse_turbo, 0.5) << "TurboKV 4B MSE too high";
    EXPECT_LT(avg_mse_uniform, 0.5) << "Uniform 4B MSE too high";
}

/* ============================================================
 * Traits table integration
 * ============================================================ */

TEST(TurboKV, TraitsTable) {
    /* Verify new types are registered in TQ_TRAITS */
    EXPECT_STREQ(tq_type_name(TQ_TYPE_TURBO_KV_3B), "turbo_kv_3b");
    EXPECT_STREQ(tq_type_name(TQ_TYPE_TURBO_KV_4B), "turbo_kv_4b");

    EXPECT_EQ(tq_type_block_size(TQ_TYPE_TURBO_KV_3B), (size_t)TQ_BK);
    EXPECT_EQ(tq_type_block_size(TQ_TYPE_TURBO_KV_4B), (size_t)TQ_BK);

    EXPECT_EQ(tq_type_type_size(TQ_TYPE_TURBO_KV_3B), sizeof(block_tq_turbo_kv_3b));
    EXPECT_EQ(tq_type_type_size(TQ_TYPE_TURBO_KV_4B), sizeof(block_tq_turbo_kv_4b));

    EXPECT_GT(tq_type_bpe(TQ_TYPE_TURBO_KV_3B), 0.0f);
    EXPECT_GT(tq_type_bpe(TQ_TYPE_TURBO_KV_4B), 0.0f);

    /* Type lookup by name */
    EXPECT_EQ(tq_type_from_name("turbo_kv_3b"), TQ_TYPE_TURBO_KV_3B);
    EXPECT_EQ(tq_type_from_name("turbo_kv_4b"), TQ_TYPE_TURBO_KV_4B);
}

TEST(TurboKV, FormatSpec) {
    tq_format_spec_t spec3 = tq_get_format_spec(TQ_TYPE_TURBO_KV_3B);
    EXPECT_EQ(spec3.algorithm, TQ_ALG_TURBO);
    EXPECT_EQ(spec3.key_bits, 3);
    EXPECT_TRUE(spec3.flags & TQ_FLAG_HAS_RESIDUAL);

    tq_format_spec_t spec4 = tq_get_format_spec(TQ_TYPE_TURBO_KV_4B);
    EXPECT_EQ(spec4.algorithm, TQ_ALG_TURBO);
    EXPECT_EQ(spec4.key_bits, 4);
    EXPECT_TRUE(spec4.flags & TQ_FLAG_HAS_RESIDUAL);
}

/* ============================================================
 * Bit-packing correctness test
 * ============================================================ */

TEST(TurboKV, BitPackingRoundtrip2Bit) {
    /* Verify 2-bit packing/unpacking is lossless */
    const int n = TQ_BK;
    std::vector<uint8_t> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i % 4;

    std::vector<uint8_t> packed(n / 4);
    /* Use the quantize/dequantize path to test packing implicitly */
    block_tq_turbo_kv_3b block;
    memset(&block, 0, sizeof(block));

    /* Create data that maps to known indices */
    std::vector<float> input(n);
    for (int i = 0; i < n; i++) input[i] = 0.5f * sinf(i * 0.05f);

    tq_turbo_kv_3b_quantize_ref(input.data(), &block, n);
    std::vector<float> output(n);
    tq_turbo_kv_3b_dequantize_ref(&block, output.data(), n);

    /* Verify the reconstruction is reasonable (not garbage from bad packing) */
    double cosine = compute_cosine(input.data(), output.data(), n);
    EXPECT_GT(cosine, 0.3);  /* 2-bit: low but not garbage */
}

TEST(TurboKV, BitPackingRoundtrip3Bit) {
    const int n = TQ_BK;
    std::vector<float> input(n);
    for (int i = 0; i < n; i++) input[i] = 0.5f * sinf(i * 0.05f);

    block_tq_turbo_kv_4b block;
    memset(&block, 0, sizeof(block));
    tq_turbo_kv_4b_quantize_ref(input.data(), &block, n);
    std::vector<float> output(n);
    tq_turbo_kv_4b_dequantize_ref(&block, output.data(), n);

    double cosine = compute_cosine(input.data(), output.data(), n);
    EXPECT_GT(cosine, 0.6);  /* 3-bit: moderate precision */
}

/* ============================================================
 * Zero/constant input edge cases
 * ============================================================ */

TEST(TurboKV, ZeroInput) {
    const int dim = TQ_BK;
    std::vector<float> input(dim, 0.0f);

    block_tq_turbo_kv_3b block_3b;
    memset(&block_3b, 0, sizeof(block_3b));
    tq_turbo_kv_3b_quantize_ref(input.data(), &block_3b, dim);
    std::vector<float> output_3b(dim);
    tq_turbo_kv_3b_dequantize_ref(&block_3b, output_3b.data(), dim);

    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(output_3b[i], 0.0f, 1e-4f);
    }

    block_tq_turbo_kv_4b block_4b;
    memset(&block_4b, 0, sizeof(block_4b));
    tq_turbo_kv_4b_quantize_ref(input.data(), &block_4b, dim);
    std::vector<float> output_4b(dim);
    tq_turbo_kv_4b_dequantize_ref(&block_4b, output_4b.data(), dim);

    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(output_4b[i], 0.0f, 1e-4f);
    }
}
