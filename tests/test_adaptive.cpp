/**
 * test_adaptive.cpp -- Tests for adaptive compression features
 *
 * Tests:
 *   1. Per-layer bit recommendation (kurtosis-based)
 *   2. Attention entropy computation
 *   3. V highres window (state allocation and field setting)
 *   4. Online Lloyd-Max codebook calibration
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <vector>
#include <numeric>
#include <algorithm>

extern "C" {
#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"
}

/* ============================================================
 * Feature 1: Per-Layer Bit Recommendation
 * ============================================================ */

TEST(AdaptiveLayerBits, HighKurtosisGets3Bit) {
    /* Layers with kurtosis > 6.0 should get 3-bit recommendation */
    float kurtosis[] = {7.5f, 8.0f, 6.5f, 3.5f, 4.0f, 5.5f};
    int n_layers = 6;
    int recommended[6] = {0};
    float avg = 0.0f;

    tq_recommend_layer_bits(kurtosis, n_layers, recommended, &avg);

    EXPECT_EQ(recommended[0], 3);  /* 7.5 > 6.0 */
    EXPECT_EQ(recommended[1], 3);  /* 8.0 > 6.0 */
    EXPECT_EQ(recommended[2], 3);  /* 6.5 > 6.0 */
    EXPECT_EQ(recommended[3], 1);  /* 3.5 <= 6.0 */
    EXPECT_EQ(recommended[4], 1);  /* 4.0 <= 6.0 */
    EXPECT_EQ(recommended[5], 1);  /* 5.5 <= 6.0 */

    /* Average: (3+3+3+1+1+1)/6 = 2.0 */
    EXPECT_NEAR(avg, 2.0f, 0.01f);
}

TEST(AdaptiveLayerBits, AllLowKurtosis) {
    float kurtosis[] = {3.0f, 3.5f, 4.0f, 5.0f};
    int recommended[4] = {0};
    float avg = 0.0f;

    tq_recommend_layer_bits(kurtosis, 4, recommended, &avg);

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(recommended[i], 1);
    }
    EXPECT_NEAR(avg, 1.0f, 0.01f);
}

TEST(AdaptiveLayerBits, AllHighKurtosis) {
    float kurtosis[] = {7.0f, 8.0f, 9.0f};
    int recommended[3] = {0};
    float avg = 0.0f;

    tq_recommend_layer_bits(kurtosis, 3, recommended, &avg);

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(recommended[i], 3);
    }
    EXPECT_NEAR(avg, 3.0f, 0.01f);
}

TEST(AdaptiveLayerBits, NullSafety) {
    /* Should not crash with NULL inputs */
    int recommended[4];
    float avg = 0.0f;
    tq_recommend_layer_bits(nullptr, 4, recommended, &avg);
    tq_recommend_layer_bits(nullptr, 0, nullptr, nullptr);
}

TEST(AdaptiveLayerBits, MemorySavingsEstimate) {
    /* Simulate a 28-layer model where 10 layers have high kurtosis */
    const int n = 28;
    float kurtosis[n];
    for (int i = 0; i < n; i++) {
        kurtosis[i] = (i < 10) ? 7.0f : 4.0f;
    }
    int recommended[n];
    float avg = 0.0f;

    tq_recommend_layer_bits(kurtosis, n, recommended, &avg);

    /* 10 layers @ 3-bit + 18 layers @ 1-bit = 48 bits total, avg = 48/28 */
    float expected_avg = (10.0f * 3 + 18.0f * 1) / 28.0f;
    EXPECT_NEAR(avg, expected_avg, 0.01f);

    float savings = (1.0f - avg / 3.0f) * 100.0f;
    EXPECT_GT(savings, 0.0f);
    EXPECT_LT(savings, 100.0f);
}

/* ============================================================
 * Feature 2: Attention Entropy
 * ============================================================ */

TEST(AttentionEntropy, UniformDistribution) {
    /* Uniform distribution over N tokens: H = log2(N) */
    const int N = 64;
    float probs[N];
    for (int i = 0; i < N; i++) {
        probs[i] = 1.0f / N;
    }

    float h = tq_attention_entropy(probs, N);
    float expected = log2f((float)N);
    EXPECT_NEAR(h, expected, 0.01f);
}

TEST(AttentionEntropy, DeltaDistribution) {
    /* All attention on one token: H = 0 */
    const int N = 128;
    float probs[N];
    memset(probs, 0, sizeof(probs));
    probs[0] = 1.0f;

    float h = tq_attention_entropy(probs, N);
    EXPECT_NEAR(h, 0.0f, 0.001f);
}

TEST(AttentionEntropy, TwoTokenEqual) {
    /* Equal attention on 2 tokens: H = 1.0 bit */
    const int N = 10;
    float probs[N];
    memset(probs, 0, sizeof(probs));
    probs[0] = 0.5f;
    probs[1] = 0.5f;

    float h = tq_attention_entropy(probs, N);
    EXPECT_NEAR(h, 1.0f, 0.001f);
}

TEST(AttentionEntropy, NullSafety) {
    float h = tq_attention_entropy(nullptr, 10);
    EXPECT_NEAR(h, 0.0f, 0.001f);

    h = tq_attention_entropy(nullptr, 0);
    EXPECT_NEAR(h, 0.0f, 0.001f);
}

TEST(AttentionEntropy, MonotonicWithSpread) {
    /* More spread = higher entropy */
    const int N = 32;

    /* Sharp: most weight on first token */
    float sharp[N];
    memset(sharp, 0, sizeof(sharp));
    sharp[0] = 0.9f;
    sharp[1] = 0.1f;

    /* Diffuse: spread across many tokens */
    float diffuse[N];
    for (int i = 0; i < N; i++) {
        diffuse[i] = 1.0f / N;
    }

    float h_sharp = tq_attention_entropy(sharp, N);
    float h_diffuse = tq_attention_entropy(diffuse, N);

    EXPECT_LT(h_sharp, h_diffuse);
}

/* ============================================================
 * Feature 3: V Highres Window State
 * ============================================================ */

TEST(VHighresWindow, StateFieldExists) {
    /* Verify v_highres_window field exists and defaults to 0 */
    tq_state_t state;
    memset(&state, 0, sizeof(state));
    EXPECT_EQ(state.v_highres_window, 0);
    EXPECT_EQ(state.value_highres_fp16, nullptr);
}

TEST(VHighresWindow, GenConfigFieldExists) {
    /* Verify v_highres_window field in gen config */
    tq_gen_config_t config = tq_default_gen_config();
    EXPECT_EQ(config.v_highres_window, 0);
}

/* ============================================================
 * Feature 4: Online Lloyd-Max Codebook Calibration
 * ============================================================ */

TEST(CodebookCalibration, GaussianData) {
    /* Generate standard normal-like data and calibrate 4-level codebook */
    const int N = 10000;
    std::vector<float> data(N);

    /* Simple Box-Muller for pseudo-normal data */
    unsigned int seed = 42;
    for (int i = 0; i < N; i += 2) {
        float u1 = ((float)(seed = seed * 1103515245 + 12345) / (float)UINT_MAX);
        float u2 = ((float)(seed = seed * 1103515245 + 12345) / (float)UINT_MAX);
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * (float)M_PI * u2);
        data[i] = z0;
        if (i + 1 < N) data[i + 1] = z1;
    }

    float centroids[4] = {0};
    float boundaries[3] = {0};
    float mse = tq_calibrate_codebook(data.data(), N, 4, 20, centroids, boundaries);

    /* MSE should be reasonable (< 0.5 for normal data with 4 levels) */
    EXPECT_GT(mse, 0.0f);
    EXPECT_LT(mse, 0.5f);

    /* Centroids should be sorted (ascending) */
    for (int i = 0; i < 3; i++) {
        EXPECT_LT(centroids[i], centroids[i + 1]);
    }

    /* Centroids should be roughly symmetric around 0 */
    EXPECT_NEAR(centroids[0] + centroids[3], 0.0f, 0.3f);
    EXPECT_NEAR(centroids[1] + centroids[2], 0.0f, 0.3f);

    /* Should be close to optimal Lloyd-Max centroids for N(0,1):
     * [-1.510, -0.453, 0.453, 1.510] */
    EXPECT_NEAR(centroids[0], -1.510f, 0.3f);
    EXPECT_NEAR(centroids[1], -0.453f, 0.3f);
    EXPECT_NEAR(centroids[2],  0.453f, 0.3f);
    EXPECT_NEAR(centroids[3],  1.510f, 0.3f);
}

TEST(CodebookCalibration, EightLevels) {
    /* 8-level codebook (3-bit) should have lower MSE than 4-level */
    const int N = 5000;
    std::vector<float> data(N);

    unsigned int seed = 123;
    for (int i = 0; i < N; i += 2) {
        float u1 = ((float)(seed = seed * 1103515245 + 12345) / (float)UINT_MAX);
        float u2 = ((float)(seed = seed * 1103515245 + 12345) / (float)UINT_MAX);
        if (u1 < 1e-10f) u1 = 1e-10f;
        data[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        if (i + 1 < N) data[i + 1] = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * (float)M_PI * u2);
    }

    float centroids_4[4], centroids_8[8];
    float mse_4 = tq_calibrate_codebook(data.data(), N, 4, 20, centroids_4, nullptr);
    float mse_8 = tq_calibrate_codebook(data.data(), N, 8, 20, centroids_8, nullptr);

    /* 8-level should always have lower or equal MSE than 4-level */
    EXPECT_LE(mse_8, mse_4);
}

TEST(CodebookCalibration, CalibrationBetterThanDefault) {
    /* Calibrated codebook should have <= MSE compared to default N(0,1) codebook
     * when data is NOT exactly N(0,1) (e.g., heavier tails) */
    const int N = 5000;
    std::vector<float> data(N);

    /* Generate data with heavier tails (like post-RHT activations) */
    unsigned int seed = 99;
    for (int i = 0; i < N; i++) {
        float u1 = ((float)(seed = seed * 1103515245 + 12345) / (float)UINT_MAX);
        float u2 = ((float)(seed = seed * 1103515245 + 12345) / (float)UINT_MAX);
        if (u1 < 1e-10f) u1 = 1e-10f;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        /* Make heavier tails by cubing: preserves sign, increases kurtosis */
        data[i] = z * (1.0f + 0.3f * z * z);
    }

    /* Normalize */
    double sum = 0, sum_sq = 0;
    for (int i = 0; i < N; i++) {
        sum += data[i];
        sum_sq += (double)data[i] * data[i];
    }
    float mean = (float)(sum / N);
    float std_val = sqrtf((float)(sum_sq / N - (double)mean * mean));
    for (int i = 0; i < N; i++) {
        data[i] = (data[i] - mean) / std_val;
    }

    /* Calibrated MSE */
    float centroids_cal[4];
    float mse_cal = tq_calibrate_codebook(data.data(), N, 4, 20, centroids_cal, nullptr);

    /* Default N(0,1) MSE */
    float default_c[4] = {-1.510f, -0.453f, 0.453f, 1.510f};
    double mse_default = 0.0;
    for (int i = 0; i < N; i++) {
        float best = fabsf(data[i] - default_c[0]);
        for (int c = 1; c < 4; c++) {
            float d = fabsf(data[i] - default_c[c]);
            if (d < best) best = d;
        }
        mse_default += (double)(best * best);
    }
    mse_default /= N;

    /* Calibrated should be better or equal */
    EXPECT_LE((double)mse_cal, mse_default + 0.001);
}

TEST(CodebookCalibration, NullSafety) {
    float centroids[4];
    float result = tq_calibrate_codebook(nullptr, 100, 4, 10, centroids, nullptr);
    EXPECT_LT(result, 0.0f);

    float data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    result = tq_calibrate_codebook(data, 10, 4, 10, nullptr, nullptr);
    EXPECT_LT(result, 0.0f);
}

TEST(CodebookCalibration, ConvergenceImprovesMSE) {
    /* More iterations should give equal or better MSE */
    const int N = 1000;
    std::vector<float> data(N);

    unsigned int seed = 77;
    for (int i = 0; i < N; i++) {
        float u = ((float)(seed = seed * 1103515245 + 12345) / (float)UINT_MAX);
        data[i] = u * 6.0f - 3.0f;  /* Uniform[-3, 3] */
    }

    float c1[4], c10[4];
    float mse_1 = tq_calibrate_codebook(data.data(), N, 4, 1, c1, nullptr);
    float mse_10 = tq_calibrate_codebook(data.data(), N, 4, 10, c10, nullptr);

    EXPECT_LE(mse_10, mse_1 + 0.001f);
}
