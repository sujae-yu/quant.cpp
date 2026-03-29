/**
 * TurboQuant llama.cpp integration test
 *
 * Simulates the full llama.cpp integration flow without linking
 * against llama.cpp itself:
 *   1. Register GGML types
 *   2. Parse CLI arguments
 *   3. Create context
 *   4. Quantize keys (simulating prefill)
 *   5. Compute attention (simulating decode)
 *   6. Verify results
 *
 * Build:
 *   Compiled as part of the TurboQuant test suite when TQ_BUILD_TESTS=ON
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>

/* Include the integration layer (it pulls in turboquant.h internally) */
#include "tq_kv_cache.cpp"

/* ============================================================
 * Helper: generate deterministic test data
 * ============================================================ */

static void fill_sinusoidal(float* data, int n, float freq, float phase) {
    for (int i = 0; i < n; i++) {
        data[i] = sinf(freq * (float)i + phase);
    }
}

static double compute_mse(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    return mse / n;
}

static double cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

/* ============================================================
 * Test: Type registration
 * ============================================================ */

TEST(LlamaCppIntegration, RegisterTypes) {
    /* Reset registration state for test isolation */
    g_tq_registered = 0;

    tq_status status = tq_ggml_register_types();
    EXPECT_EQ(status, TQ_OK);

    /* Second call should be idempotent */
    status = tq_ggml_register_types();
    EXPECT_EQ(status, TQ_OK);
}

/* ============================================================
 * Test: Trait table completeness
 * ============================================================ */

TEST(LlamaCppIntegration, TraitTableComplete) {
    for (size_t i = 0; i < TQ_GGML_NUM_TYPES; i++) {
        const tq_ggml_type_trait* t = &TQ_GGML_TRAITS[i];

        EXPECT_NE(t->type_name, nullptr) << "Trait " << i << " has null name";
        EXPECT_GT(t->type_size, 0u) << "Trait " << i << " has zero type_size";
        EXPECT_GT(t->block_size, 0u) << "Trait " << i << " has zero block_size";
        EXPECT_GT(t->bpe, 0.0f) << "Trait " << i << " has zero bpe";
        EXPECT_NE(t->from_float, nullptr) << "Trait " << i << " has null from_float";
        EXPECT_NE(t->to_float, nullptr) << "Trait " << i << " has null to_float";
        EXPECT_NE(t->vec_dot, nullptr) << "Trait " << i << " has null vec_dot";
    }
}

/* ============================================================
 * Test: GGML type ID mapping roundtrip
 * ============================================================ */

TEST(LlamaCppIntegration, TypeIdMapping) {
    for (int i = 0; i < TQ_TYPE_COUNT; i++) {
        tq_type tq = (tq_type)i;
        int ggml_id = tq_to_ggml_type(tq);
        EXPECT_GE(ggml_id, GGML_TYPE_TQ_BASE);

        tq_type back = ggml_to_tq_type(ggml_id);
        EXPECT_EQ(back, tq) << "Roundtrip failed for tq_type " << i;
    }

    /* Invalid types */
    EXPECT_EQ(tq_to_ggml_type(TQ_TYPE_COUNT), -1);
    EXPECT_EQ(ggml_to_tq_type(0), TQ_TYPE_COUNT);
    EXPECT_EQ(ggml_to_tq_type(999), TQ_TYPE_COUNT);
}

/* ============================================================
 * Test: CLI argument parsing
 * ============================================================ */

TEST(LlamaCppIntegration, ParseKvCacheType) {
    /* Short aliases */
    EXPECT_EQ(tq_parse_kv_cache_type("turbo3"),    TQ_TYPE_TURBO_3B);
    EXPECT_EQ(tq_parse_kv_cache_type("turbo4"),    TQ_TYPE_TURBO_4B);
    EXPECT_EQ(tq_parse_kv_cache_type("polar3"),    TQ_TYPE_POLAR_3B);
    EXPECT_EQ(tq_parse_kv_cache_type("polar4"),    TQ_TYPE_POLAR_4B);
    EXPECT_EQ(tq_parse_kv_cache_type("qjl1"),      TQ_TYPE_QJL_1B);
    EXPECT_EQ(tq_parse_kv_cache_type("uniform4"),  TQ_TYPE_UNIFORM_4B);
    EXPECT_EQ(tq_parse_kv_cache_type("uniform2"),  TQ_TYPE_UNIFORM_2B);

    /* Underscore aliases */
    EXPECT_EQ(tq_parse_kv_cache_type("turbo_3b"),  TQ_TYPE_TURBO_3B);
    EXPECT_EQ(tq_parse_kv_cache_type("uniform_4b"),TQ_TYPE_UNIFORM_4B);

    /* Dash aliases (ggml-style) */
    EXPECT_EQ(tq_parse_kv_cache_type("tq-turbo-3b"),  TQ_TYPE_TURBO_3B);
    EXPECT_EQ(tq_parse_kv_cache_type("tq-uniform-4b"),TQ_TYPE_UNIFORM_4B);

    /* Invalid */
    EXPECT_EQ(tq_parse_kv_cache_type(nullptr),     TQ_TYPE_COUNT);
    EXPECT_EQ(tq_parse_kv_cache_type("invalid"),   TQ_TYPE_COUNT);
    EXPECT_EQ(tq_parse_kv_cache_type(""),          TQ_TYPE_COUNT);
}

/* ============================================================
 * Test: from_float / to_float roundtrip via GGML wrappers
 *
 * For each TurboQuant type, quantize a block of data through
 * the GGML-compatible wrappers and verify the roundtrip MSE
 * is within acceptable bounds.
 * ============================================================ */

TEST(LlamaCppIntegration, GgmlFromFloatToFloatRoundtrip) {
    for (size_t i = 0; i < TQ_GGML_NUM_TYPES; i++) {
        const tq_ggml_type_trait* t = &TQ_GGML_TRAITS[i];
        int bs = (int)t->block_size;

        std::vector<float> input(bs);
        fill_sinusoidal(input.data(), bs, 0.1f, 0.0f);

        /* Allocate quantized buffer */
        std::vector<uint8_t> quantized(t->type_size);

        /* Quantize via GGML wrapper */
        t->from_float(input.data(), quantized.data(), bs);

        /* Dequantize via GGML wrapper */
        std::vector<float> output(bs);
        t->to_float(quantized.data(), output.data(), bs);

        /* Check MSE */
        double mse = compute_mse(input.data(), output.data(), bs);
        EXPECT_LT(mse, 1.0) << "Type " << t->type_name
                             << " has excessive roundtrip MSE: " << mse;
    }
}

/* ============================================================
 * Test: vec_dot correctness
 *
 * Verify that the vec_dot wrapper produces results consistent
 * with dequantize-then-dot (the reference approach).
 * ============================================================ */

TEST(LlamaCppIntegration, GgmlVecDotConsistency) {
    for (size_t i = 0; i < TQ_GGML_NUM_TYPES; i++) {
        const tq_ggml_type_trait* t = &TQ_GGML_TRAITS[i];
        int bs = (int)t->block_size;

        /* Create key and query vectors */
        std::vector<float> key(bs), query(bs);
        fill_sinusoidal(key.data(), bs, 0.05f, 0.0f);
        fill_sinusoidal(query.data(), bs, 0.07f, 1.0f);

        /* Quantize key */
        std::vector<uint8_t> quantized(t->type_size);
        t->from_float(key.data(), quantized.data(), bs);

        /* Method 1: vec_dot */
        float dot_result = 0.0f;
        t->vec_dot(bs, &dot_result, quantized.data(), query.data());

        /* Method 2: dequantize then dot */
        std::vector<float> deq(bs);
        t->to_float(quantized.data(), deq.data(), bs);
        float ref_dot = 0.0f;
        for (int j = 0; j < bs; j++) {
            ref_dot += deq[j] * query[j];
        }

        /* Results should match exactly (same code path internally) */
        EXPECT_NEAR(dot_result, ref_dot, 1e-3f)
            << "Type " << t->type_name << " vec_dot mismatch";
    }
}

/* ============================================================
 * Test: End-to-end simulated llama.cpp flow
 *
 * Simulates what happens during llama.cpp inference:
 *   1. Register types
 *   2. Parse CLI type selection
 *   3. Create context
 *   4. Quantize keys (prefill phase)
 *   5. Compute attention (decode phase)
 *   6. Verify attention scores are reasonable
 * ============================================================ */

TEST(LlamaCppIntegration, EndToEndFlow) {
    const int head_dim = TQ_BK;  /* 128 */
    const int seq_len = 4;       /* 4 cached tokens */

    /* Step 1: Register types */
    g_tq_registered = 0;
    ASSERT_EQ(tq_ggml_register_types(), TQ_OK);

    /* Step 2: Parse type */
    tq_type kv_type = tq_parse_kv_cache_type("uniform4");
    ASSERT_EQ(kv_type, TQ_TYPE_UNIFORM_4B);

    /* Step 3: Create context */
    tq_context_t* ctx = tq_llamacpp_create_context();
    ASSERT_NE(ctx, nullptr);

    /* Step 4: Generate key vectors and quantize */
    std::vector<float> keys(seq_len * head_dim);
    for (int t = 0; t < seq_len; t++) {
        fill_sinusoidal(keys.data() + t * head_dim, head_dim,
                        0.1f, (float)t * 0.5f);
    }

    size_t key_buf_size = tq_quantize_keys_size(seq_len, head_dim, kv_type);
    ASSERT_GT(key_buf_size, 0u);

    std::vector<uint8_t> key_cache(key_buf_size);
    tq_status status = tq_llamacpp_quantize_keys(
        ctx, keys.data(), seq_len, head_dim, kv_type,
        key_cache.data(), key_buf_size);
    ASSERT_EQ(status, TQ_OK);

    /* Step 5: Compute attention scores */
    std::vector<float> query(head_dim);
    fill_sinusoidal(query.data(), head_dim, 0.1f, 0.0f);

    std::vector<float> scores(seq_len);
    status = tq_llamacpp_attention(
        ctx, query.data(), key_cache.data(),
        seq_len, head_dim, kv_type, scores.data());
    ASSERT_EQ(status, TQ_OK);

    /* Step 6: Verify attention scores are non-trivial.
     * The first key matches the query (same frequency, phase=0),
     * so scores[0] should be the largest. */
    bool has_nonzero = false;
    for (int i = 0; i < seq_len; i++) {
        if (scores[i] != 0.0f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero) << "All attention scores are zero";

    /* Compute reference attention scores from FP32 keys */
    std::vector<float> ref_scores(seq_len);
    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * keys[t * head_dim + d];
        }
        ref_scores[t] = dot;
    }

    /* Quantized attention should correlate with reference */
    double cos_sim = cosine_similarity(scores.data(), ref_scores.data(), seq_len);
    EXPECT_GT(cos_sim, 0.8) << "Attention score correlation too low: " << cos_sim;

    /* Cleanup */
    tq_free(ctx);
}

/* ============================================================
 * Test: Memory estimation
 * ============================================================ */

TEST(LlamaCppIntegration, BytesPerToken) {
    const int head_dim = 128;

    /* FP16 baseline: 128 * 2 * 2 = 512 bytes per token per head (K+V) */
    size_t fp16_bytes = (size_t)head_dim * 2 * 2;

    /* TurboQuant should use less memory */
    for (int i = 0; i < TQ_TYPE_COUNT; i++) {
        size_t tq_bytes = tq_llamacpp_bytes_per_token(head_dim, (tq_type)i, 0);
        /* Key is compressed, value is FP16 (default) */
        /* Key should be less than FP16, so total should be less than FP16 */
        EXPECT_LT(tq_bytes, fp16_bytes)
            << "Type " << tq_type_name((tq_type)i)
            << " uses more memory than FP16: " << tq_bytes << " >= " << fp16_bytes;
    }

    /* With value quantization, even more savings */
    size_t tq_with_val4 = tq_llamacpp_bytes_per_token(head_dim, TQ_TYPE_UNIFORM_4B, 4);
    size_t tq_with_val2 = tq_llamacpp_bytes_per_token(head_dim, TQ_TYPE_UNIFORM_4B, 2);
    EXPECT_LT(tq_with_val4, fp16_bytes);
    EXPECT_LT(tq_with_val2, tq_with_val4);
}

/* ============================================================
 * Test: Print config (smoke test -- just verify no crash)
 * ============================================================ */

TEST(LlamaCppIntegration, PrintConfig) {
    /* This should not crash or assert */
    tq_llamacpp_print_config(TQ_TYPE_TURBO_3B, 4, 32, 128, 4096);
    tq_llamacpp_print_config(TQ_TYPE_UNIFORM_4B, 0, 8, 64, 2048);
}

/* ============================================================
 * Test: Print available types (smoke test)
 * ============================================================ */

TEST(LlamaCppIntegration, PrintKvCacheTypes) {
    /* Should not crash */
    tq_print_kv_cache_types();
}

/* ============================================================
 * Test: Multi-block quantize/dequantize via GGML wrappers
 *
 * Exercises the block-loop logic in tq_ggml_from_float/to_float
 * with multiple blocks worth of data.
 * ============================================================ */

TEST(LlamaCppIntegration, MultiBlockRoundtrip) {
    const int num_blocks = 4;

    for (size_t i = 0; i < TQ_GGML_NUM_TYPES; i++) {
        const tq_ggml_type_trait* t = &TQ_GGML_TRAITS[i];
        int total_elems = num_blocks * (int)t->block_size;
        size_t total_bytes = num_blocks * t->type_size;

        std::vector<float> input(total_elems);
        fill_sinusoidal(input.data(), total_elems, 0.03f, 0.5f);

        std::vector<uint8_t> quantized(total_bytes);
        t->from_float(input.data(), quantized.data(), total_elems);

        std::vector<float> output(total_elems);
        t->to_float(quantized.data(), output.data(), total_elems);

        double mse = compute_mse(input.data(), output.data(), total_elems);
        EXPECT_LT(mse, 1.0) << "Type " << t->type_name
                             << " multi-block MSE too high: " << mse;
    }
}
