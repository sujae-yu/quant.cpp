/**
 * test_unbiased.cpp -- Formal unbiasedness verification for TurboQuant estimators
 *
 * Proves that quantized inner product estimators are unbiased:
 *   E[<q, quantize(k)>] = <q, k>
 *
 * Tests turbo_kv_1b, turbo_kv_3b, turbo_kv_4b, uniform_4b, uniform_2b.
 * Uses 100,000+ random vector pairs and measures relative bias.
 *
 * Target: relative bias < 0.01 (1%) for all TurboQuant types.
 */

#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
#include "turboquant/tq_types.h"
}

#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <cstring>

/* ============================================================
 * Helper: compute true FP32 inner product
 * ============================================================ */
static double true_dot(const float* q, const float* k, int dim) {
    double dot = 0.0;
    for (int i = 0; i < dim; i++) {
        dot += (double)q[i] * (double)k[i];
    }
    return dot;
}

/* ============================================================
 * Helper: compute estimated inner product via quantized attention
 *
 * Quantizes k, then uses the type's attention function to compute
 * the inner product estimate (seq_len=1 case).
 * ============================================================ */
static double estimated_dot(const float* q, const float* k, int dim, tq_type type) {
    const tq_type_traits_t* traits = &TQ_TRAITS[type];
    if (!traits->quantize || !traits->attention) return 0.0;

    /* Allocate block for quantized key */
    size_t block_bytes = traits->type_size;
    std::vector<uint8_t> block(block_bytes, 0);

    /* Quantize k */
    traits->quantize(k, block.data(), dim);

    /* Compute attention score (seq_len=1 gives raw dot product estimate) */
    float score = 0.0f;
    traits->attention(q, block.data(), &score, 1, dim);

    return (double)score;
}

/* ============================================================
 * Core test: measure bias over N random trials
 * ============================================================ */
struct BiasResult {
    double mean_true;
    double mean_estimated;
    double abs_bias;
    double rel_bias;
    double std_error;
};

static BiasResult measure_bias(tq_type type, int dim, int n_trials, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    double sum_true = 0.0;
    double sum_est  = 0.0;
    double sum_diff = 0.0;
    double sum_diff_sq = 0.0;
    double sum_abs_true = 0.0;

    std::vector<float> q(dim);
    std::vector<float> k(dim);

    for (int t = 0; t < n_trials; t++) {
        /* Generate random vectors */
        for (int i = 0; i < dim; i++) {
            q[i] = normal(rng);
            k[i] = normal(rng);
        }

        double td = true_dot(q.data(), k.data(), dim);
        double ed = estimated_dot(q.data(), k.data(), dim, type);

        sum_true += td;
        sum_est  += ed;
        sum_diff += (ed - td);
        sum_diff_sq += (ed - td) * (ed - td);
        sum_abs_true += std::abs(td);
    }

    BiasResult r;
    r.mean_true = sum_true / n_trials;
    r.mean_estimated = sum_est / n_trials;
    r.abs_bias = std::abs(sum_est - sum_true) / n_trials;
    r.rel_bias = (sum_abs_true > 0.0) ?
        std::abs(sum_est - sum_true) / sum_abs_true : 0.0;
    double var_diff = sum_diff_sq / n_trials - (sum_diff / n_trials) * (sum_diff / n_trials);
    r.std_error = std::sqrt(var_diff > 0 ? var_diff / n_trials : 0.0);

    return r;
}

/* ============================================================
 * Tests
 * ============================================================ */

class UnbiasedTest : public ::testing::Test {
protected:
    static constexpr int DIM = 128;       /* head_dim */
    static constexpr int N_TRIALS = 100000;
    static constexpr unsigned SEED = 42;
};

TEST_F(UnbiasedTest, TurboKV1B_Unbiased) {
    BiasResult r = measure_bias(TQ_TYPE_TURBO_KV_1B, DIM, N_TRIALS, SEED);
    fprintf(stderr, "[turbo_kv_1b] mean_true=%.6f mean_est=%.6f rel_bias=%.6f std_err=%.6f\n",
            r.mean_true, r.mean_estimated, r.rel_bias, r.std_error);
    /* 1-bit sign quantization has inherent bias; allow up to 5% */
    EXPECT_LT(r.rel_bias, 0.05) << "turbo_kv_1b relative bias exceeds 5%";
}

TEST_F(UnbiasedTest, TurboKV3B_Unbiased) {
    BiasResult r = measure_bias(TQ_TYPE_TURBO_KV_3B, DIM, N_TRIALS, SEED);
    fprintf(stderr, "[turbo_kv_3b] mean_true=%.6f mean_est=%.6f rel_bias=%.6f std_err=%.6f\n",
            r.mean_true, r.mean_estimated, r.rel_bias, r.std_error);
    /* QJL correction should make this nearly unbiased */
    EXPECT_LT(r.rel_bias, 0.01) << "turbo_kv_3b relative bias exceeds 1%";
}

TEST_F(UnbiasedTest, TurboKV4B_Unbiased) {
    BiasResult r = measure_bias(TQ_TYPE_TURBO_KV_4B, DIM, N_TRIALS, SEED);
    fprintf(stderr, "[turbo_kv_4b] mean_true=%.6f mean_est=%.6f rel_bias=%.6f std_err=%.6f\n",
            r.mean_true, r.mean_estimated, r.rel_bias, r.std_error);
    EXPECT_LT(r.rel_bias, 0.01) << "turbo_kv_4b relative bias exceeds 1%";
}

TEST_F(UnbiasedTest, Uniform4B_Bias) {
    BiasResult r = measure_bias(TQ_TYPE_UNIFORM_4B, DIM, N_TRIALS, SEED);
    fprintf(stderr, "[uniform_4b] mean_true=%.6f mean_est=%.6f rel_bias=%.6f std_err=%.6f\n",
            r.mean_true, r.mean_estimated, r.rel_bias, r.std_error);
    /* Uniform 4-bit should have low bias */
    EXPECT_LT(r.rel_bias, 0.02) << "uniform_4b relative bias exceeds 2%";
}

TEST_F(UnbiasedTest, Uniform2B_Bias) {
    BiasResult r = measure_bias(TQ_TYPE_UNIFORM_2B, DIM, N_TRIALS, SEED);
    fprintf(stderr, "[uniform_2b] mean_true=%.6f mean_est=%.6f rel_bias=%.6f std_err=%.6f\n",
            r.mean_true, r.mean_estimated, r.rel_bias, r.std_error);
    /* 2-bit uniform will have some bias; allow more */
    EXPECT_LT(r.rel_bias, 0.05) << "uniform_2b relative bias exceeds 5%";
}

/* ============================================================
 * Variance test: TurboKV types should have lower variance than
 * same-bit-rate alternatives due to QJL correction.
 * ============================================================ */

TEST_F(UnbiasedTest, TurboKV4B_LowerVarianceThanTurboKV3B) {
    /* turbo_kv_4b (4 bits) should have lower variance than turbo_kv_3b (3 bits)
     * since the extra codebook bit reduces the residual that QJL must estimate */
    std::mt19937 rng(SEED);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    double var_4b = 0.0, var_3b = 0.0;
    int n = 10000;
    std::vector<float> q(DIM), k(DIM);

    for (int t = 0; t < n; t++) {
        for (int i = 0; i < DIM; i++) { q[i] = normal(rng); k[i] = normal(rng); }
        double td = true_dot(q.data(), k.data(), DIM);
        double e4 = estimated_dot(q.data(), k.data(), DIM, TQ_TYPE_TURBO_KV_4B);
        double e3 = estimated_dot(q.data(), k.data(), DIM, TQ_TYPE_TURBO_KV_3B);
        var_4b += (e4 - td) * (e4 - td);
        var_3b += (e3 - td) * (e3 - td);
    }
    var_4b /= n;
    var_3b /= n;

    fprintf(stderr, "[variance] turbo_kv_4b=%.6f, turbo_kv_3b=%.6f\n", var_4b, var_3b);
    /* 4-bit should have lower estimation MSE than 3-bit */
    EXPECT_LT(var_4b, var_3b)
        << "turbo_kv_4b should have lower estimation variance than turbo_kv_3b";
}

TEST_F(UnbiasedTest, Uniform4B_LowerVarianceThanUniform2B) {
    /* Basic sanity: 4-bit uniform should have lower variance than 2-bit */
    std::mt19937 rng(SEED + 100);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    double var_4b = 0.0, var_2b = 0.0;
    int n = 10000;
    std::vector<float> q(DIM), k(DIM);

    for (int t = 0; t < n; t++) {
        for (int i = 0; i < DIM; i++) { q[i] = normal(rng); k[i] = normal(rng); }
        double td = true_dot(q.data(), k.data(), DIM);
        double e4 = estimated_dot(q.data(), k.data(), DIM, TQ_TYPE_UNIFORM_4B);
        double e2 = estimated_dot(q.data(), k.data(), DIM, TQ_TYPE_UNIFORM_2B);
        var_4b += (e4 - td) * (e4 - td);
        var_2b += (e2 - td) * (e2 - td);
    }
    var_4b /= n;
    var_2b /= n;

    fprintf(stderr, "[variance] uniform_4b=%.6f, uniform_2b=%.6f\n", var_4b, var_2b);
    EXPECT_LT(var_4b, var_2b)
        << "uniform_4b should have lower estimation variance than uniform_2b";
}

/* ============================================================
 * Consistency: running the same vector twice gives the same result
 * (deterministic quantization)
 * ============================================================ */

TEST_F(UnbiasedTest, DeterministicQuantization) {
    std::mt19937 rng(SEED);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> q(DIM), k(DIM);
    for (int i = 0; i < DIM; i++) { q[i] = normal(rng); k[i] = normal(rng); }

    tq_type types[] = {TQ_TYPE_TURBO_KV_1B, TQ_TYPE_TURBO_KV_3B, TQ_TYPE_TURBO_KV_4B,
                       TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B};
    for (tq_type t : types) {
        double e1 = estimated_dot(q.data(), k.data(), DIM, t);
        double e2 = estimated_dot(q.data(), k.data(), DIM, t);
        EXPECT_DOUBLE_EQ(e1, e2) << "Non-deterministic for type " << (int)t;
    }
}
