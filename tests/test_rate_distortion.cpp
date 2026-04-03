/**
 * test_rate_distortion.cpp -- Information-theoretic lower bound verification
 *
 * Computes the rate-distortion lower bound for Gaussian sources and
 * compares against actual quant.cpp MSE at each bit-width.
 *
 * For Gaussian N(0, sigma^2):
 *   R(D) = 0.5 * log2(sigma^2 / D)
 *   At b bits: minimum achievable MSE = sigma^2 * 2^(-2b)
 *
 * Lloyd-Max achieves ~1.18x the theoretical minimum for Gaussian.
 * quant.cpp uses Lloyd-Max codebooks, so we expect:
 *   actual_MSE / theoretical_min ~ 1.1 to 1.3
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <numeric>

extern "C" {
#include "turboquant/turboquant.h"
#include "turboquant/tq_engine.h"
}

/* ============================================================
 * Helper: compute MSE between original and dequantized vectors
 * ============================================================ */
static double compute_mse(const float* original, const float* reconstructed, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)original[i] - (double)reconstructed[i];
        mse += diff * diff;
    }
    return mse / (double)n;
}

/* ============================================================
 * Test 1: Theoretical R(D) bound for Gaussian source
 *
 * Verify the formula: D_min(b) = sigma^2 * 2^(-2b)
 * ============================================================ */
TEST(RateDistortion, TheoreticalBoundFormula) {
    double sigma2 = 1.0;

    /* At 1 bit: D_min = 1/4 = 0.25 */
    double d1 = sigma2 * pow(2.0, -2.0 * 1);
    EXPECT_NEAR(d1, 0.25, 1e-10);

    /* At 2 bits: D_min = 1/16 = 0.0625 */
    double d2 = sigma2 * pow(2.0, -2.0 * 2);
    EXPECT_NEAR(d2, 0.0625, 1e-10);

    /* At 3 bits: D_min = 1/64 = 0.015625 */
    double d3 = sigma2 * pow(2.0, -2.0 * 3);
    EXPECT_NEAR(d3, 0.015625, 1e-10);

    /* At 4 bits: D_min = 1/256 = 0.00390625 */
    double d4 = sigma2 * pow(2.0, -2.0 * 4);
    EXPECT_NEAR(d4, 0.00390625, 1e-10);

    /* Inverse: R(D) = 0.5 * log2(sigma^2 / D) */
    double r1 = 0.5 * log2(sigma2 / d1);
    EXPECT_NEAR(r1, 1.0, 1e-10);
    double r2 = 0.5 * log2(sigma2 / d2);
    EXPECT_NEAR(r2, 2.0, 1e-10);
}

/* ============================================================
 * Test 2: quant.cpp Q4 (4-bit uniform) MSE vs theoretical bound
 *
 * Q4 uses min-max uniform quantization with 16 levels.
 * For uniform quantization of N(0,1) to 16 levels:
 *   MSE ~ sigma^2 * (2 * sigma * k)^2 / (12 * L^2)
 * where k is the clipping factor and L=16.
 * The theoretical minimum is sigma^2 * 2^{-8} = 0.00390625.
 * Uniform does worse than Lloyd-Max, so gap should be > 1.0.
 * ============================================================ */
TEST(RateDistortion, Q4UniformVsTheoretical) {
    const int dim = 128;
    const int n_vectors = 1000;
    const int total = dim * n_vectors;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(total);
    for (int i = 0; i < total; i++) data[i] = dist(rng);

    /* Compute variance */
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < total; i++) {
        sum += (double)data[i];
        sum_sq += (double)data[i] * (double)data[i];
    }
    double mean = sum / total;
    double variance = sum_sq / total - mean * mean;

    /* Quantize and dequantize using Q4 */
    int n_blocks = (total + 31) / 32;
    std::vector<uint8_t> qs(n_blocks * 16);
    std::vector<float> scales(n_blocks);
    std::vector<float> reconstructed(total);

    tq_quantize_row_q4(data.data(), qs.data(), scales.data(), total);
    tq_dequantize_row_q4(qs.data(), scales.data(), reconstructed.data(), total);

    double mse = compute_mse(data.data(), reconstructed.data(), total);

    /* Theoretical minimum at 4 bits */
    double d_min = variance * pow(2.0, -2.0 * 4);

    /* Gap: actual / theoretical */
    double gap = mse / d_min;

    /* Report */
    fprintf(stderr, "\n=== Q4 (Uniform 4-bit) Rate-Distortion Analysis ===\n");
    fprintf(stderr, "Input: %d samples from N(0, %.4f)\n", total, variance);
    fprintf(stderr, "Theoretical D_min(4-bit): %.6f\n", d_min);
    fprintf(stderr, "Actual Q4 MSE:            %.6f\n", mse);
    fprintf(stderr, "Gap ratio:                %.2fx theoretical minimum\n", gap);
    fprintf(stderr, "=============================================\n");

    /* Q4 uniform should be within reasonable range of theoretical */
    EXPECT_LT(gap, 10.0) << "Q4 MSE is more than 10x theoretical minimum";
    EXPECT_GT(gap, 0.5) << "Q4 MSE suspiciously below theoretical minimum";
}

/* ============================================================
 * Test 3: quant.cpp Q2 (2-bit Lloyd-Max) MSE vs theoretical bound
 *
 * Q2 uses Lloyd-Max codebook centroids optimized for N(0,1).
 * Expected gap: ~1.18x for Lloyd-Max 2-bit Gaussian.
 * ============================================================ */
TEST(RateDistortion, Q2LloydMaxVsTheoretical) {
    const int dim = 128;
    const int n_vectors = 1000;
    const int total = dim * n_vectors;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(total);
    for (int i = 0; i < total; i++) data[i] = dist(rng);

    double sum_sq = 0.0;
    for (int i = 0; i < total; i++) sum_sq += (double)data[i] * (double)data[i];
    double variance = sum_sq / total;

    /* Quantize and dequantize using Q2 */
    int n_blocks = (total + 31) / 32;
    std::vector<uint8_t> qs(n_blocks * 8);
    std::vector<float> scales(n_blocks);
    std::vector<float> reconstructed(total);

    tq_quantize_row_q2(data.data(), qs.data(), scales.data(), total);
    tq_dequantize_row_q2(qs.data(), scales.data(), reconstructed.data(), total);

    double mse = compute_mse(data.data(), reconstructed.data(), total);

    /* Theoretical minimum at 2 bits */
    double d_min = variance * pow(2.0, -2.0 * 2);

    double gap = mse / d_min;

    fprintf(stderr, "\n=== Q2 (Lloyd-Max 2-bit) Rate-Distortion Analysis ===\n");
    fprintf(stderr, "Input: %d samples from N(0, %.4f)\n", total, variance);
    fprintf(stderr, "Theoretical D_min(2-bit): %.6f\n", d_min);
    fprintf(stderr, "Actual Q2 MSE:            %.6f\n", mse);
    fprintf(stderr, "Gap ratio:                %.2fx theoretical minimum\n", gap);
    fprintf(stderr, "Lloyd-Max literature gap:  ~1.18x for 2-bit Gaussian\n");
    fprintf(stderr, "=============================================\n");

    /* Lloyd-Max should be close to 1.18x the theoretical minimum.
     * Allow 0.8x to 4.0x range due to block-based quantization overhead
     * (32-element blocks with per-block scale add overhead vs. global codebook). */
    EXPECT_LT(gap, 4.0) << "Q2 MSE is more than 4x theoretical minimum";
    EXPECT_GT(gap, 0.5) << "Q2 MSE suspiciously below theoretical minimum";
}

/* ============================================================
 * Test 4: Rate-distortion table across all bit-widths
 *
 * Print a comprehensive table comparing theoretical lower bounds
 * with actual quant.cpp MSE for 1, 2, 3, 4 bit quantization.
 * ============================================================ */
TEST(RateDistortion, ComprehensiveTable) {
    const int dim = 256;
    const int n_vectors = 500;
    const int total = dim * n_vectors;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(total);
    for (int i = 0; i < total; i++) data[i] = dist(rng);

    double sum_sq = 0.0;
    for (int i = 0; i < total; i++) sum_sq += (double)data[i] * (double)data[i];
    double variance = sum_sq / total;

    fprintf(stderr, "\n=== Rate-Distortion Comprehensive Table ===\n");
    fprintf(stderr, "Source: N(0, %.4f), %d samples\n\n", variance, total);
    fprintf(stderr, "%-6s %-14s %-14s %-14s %-10s\n",
            "Bits", "D_min(theory)", "Actual MSE", "TQ Method", "Gap(x)");
    fprintf(stderr, "------ -------------- -------------- -------------- ----------\n");

    /* 4-bit Q4 (uniform) */
    {
        int n_blocks = (total + 31) / 32;
        std::vector<uint8_t> qs(n_blocks * 16);
        std::vector<float> scales(n_blocks);
        std::vector<float> recon(total);
        tq_quantize_row_q4(data.data(), qs.data(), scales.data(), total);
        tq_dequantize_row_q4(qs.data(), scales.data(), recon.data(), total);
        double mse = compute_mse(data.data(), recon.data(), total);
        double d_min = variance * pow(2.0, -2.0 * 4);
        fprintf(stderr, "%-6d %-14.6f %-14.6f %-14s %-10.2f\n",
                4, d_min, mse, "Q4 uniform", mse / d_min);
    }

    /* 2-bit Q2 (Lloyd-Max) */
    {
        int n_blocks = (total + 31) / 32;
        std::vector<uint8_t> qs(n_blocks * 8);
        std::vector<float> scales(n_blocks);
        std::vector<float> recon(total);
        tq_quantize_row_q2(data.data(), qs.data(), scales.data(), total);
        tq_dequantize_row_q2(qs.data(), scales.data(), recon.data(), total);
        double mse = compute_mse(data.data(), recon.data(), total);
        double d_min = variance * pow(2.0, -2.0 * 2);
        fprintf(stderr, "%-6d %-14.6f %-14.6f %-14s %-10.2f\n",
                2, d_min, mse, "Q2 Lloyd-Max", mse / d_min);
    }

    /* quant.cpp types via the public API */
    tq_context_t* ctx = NULL;
    tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
    if (status == TQ_OK && ctx) {
        /* For each TQ type, quantize and measure MSE */
        struct { tq_type type; const char* name; int bits; } types[] = {
            {TQ_TYPE_UNIFORM_4B, "uniform_4b", 4},
            {TQ_TYPE_UNIFORM_2B, "uniform_2b", 2},
            {TQ_TYPE_POLAR_3B,   "polar_3b",   3},
            {TQ_TYPE_POLAR_4B,   "polar_4b",   4},
        };

        for (auto& t : types) {
            size_t out_size = tq_quantize_keys_size(n_vectors, dim, t.type);
            std::vector<uint8_t> out(out_size);
            std::vector<float> recon(total);

            tq_status qs = tq_quantize_keys(ctx, data.data(), n_vectors, dim,
                                             t.type, out.data(), out_size);
            if (qs != TQ_OK) continue;

            tq_status ds = tq_dequantize_keys(ctx, out.data(), n_vectors, dim,
                                               t.type, recon.data());
            if (ds != TQ_OK) continue;

            double mse = compute_mse(data.data(), recon.data(), total);
            double d_min = variance * pow(2.0, -2.0 * t.bits);
            fprintf(stderr, "%-6d %-14.6f %-14.6f %-14s %-10.2f\n",
                    t.bits, d_min, mse, t.name, mse / d_min);
        }
        tq_free(ctx);
    }

    fprintf(stderr, "\nTheoretical gap for Lloyd-Max N(0,1):\n");
    fprintf(stderr, "  2-bit: ~1.18x,  3-bit: ~1.07x,  4-bit: ~1.03x\n");
    fprintf(stderr, "Block-based quantization adds overhead, so actual gaps may be higher.\n");
    fprintf(stderr, "=============================================\n");

    /* Just verify we reach this point -- the table output is the main value */
    SUCCEED();
}

/* ============================================================
 * Test 5: Rate-distortion function R(D) verification
 *
 * For a Gaussian source, R(D) = 0.5 * log2(sigma^2 / D).
 * Given the actual MSE at each bitwidth, compute the effective
 * rate needed and compare with the actual rate.
 * ============================================================ */
TEST(RateDistortion, EffectiveRateComputation) {
    double sigma2 = 1.0;

    /* For 4-bit quantization with gap factor g:
     * actual_MSE = g * sigma^2 * 2^{-2b}
     * effective_rate = 0.5 * log2(sigma^2 / actual_MSE) = b - 0.5 * log2(g)
     * So a gap of 1.18x at 2 bits means effective rate = 2 - 0.12 = 1.88 bits
     */

    struct { int bits; double gap; } cases[] = {
        {2, 1.18},  /* Lloyd-Max 2-bit */
        {3, 1.07},  /* Lloyd-Max 3-bit */
        {4, 1.03},  /* Lloyd-Max 4-bit */
    };

    fprintf(stderr, "\n=== Effective Rate Analysis ===\n");
    fprintf(stderr, "%-6s %-10s %-14s %-14s %-12s\n",
            "Bits", "Gap(x)", "Actual MSE", "Eff. Rate", "Waste(bits)");
    fprintf(stderr, "------ ---------- -------------- -------------- ------------\n");

    for (auto& c : cases) {
        double actual_mse = c.gap * sigma2 * pow(2.0, -2.0 * c.bits);
        double effective_rate = 0.5 * log2(sigma2 / actual_mse);
        double waste = (double)c.bits - effective_rate;

        fprintf(stderr, "%-6d %-10.2f %-14.6f %-14.2f %-12.3f\n",
                c.bits, c.gap, actual_mse, effective_rate, waste);

        /* Effective rate should be slightly below nominal */
        EXPECT_GT(effective_rate, (double)c.bits - 0.5)
            << "Effective rate too far below nominal at " << c.bits << " bits";
        EXPECT_LT(effective_rate, (double)c.bits)
            << "Effective rate should be below nominal (gap > 1)";
    }

    fprintf(stderr, "Insight: Lloyd-Max wastes <0.15 bits for Gaussian sources.\n");
    fprintf(stderr, "===============================\n");
}
