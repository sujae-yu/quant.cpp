/**
 * test_metal_moe.cpp — Minimal Metal IQ2_XXS matmul kernel test
 *
 * Isolates the Metal matmul_iq2_xxs shader from the fused MoE dispatch
 * to determine whether a hang originates in the shader itself or in
 * the MoE dispatch logic.
 *
 * If this test hangs: the IQ2_XXS Metal shader is broken.
 * If this test passes: the fused MoE dispatch has the bug.
 */
#include <gtest/gtest.h>

#ifndef TQ_HAS_METAL

TEST(MetalMatmul, SkipNoMetal) {
    GTEST_SKIP() << "Metal backend not compiled (TQ_HAS_METAL not defined)";
}

#else /* TQ_HAS_METAL */

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>

/* Relative tolerance: allow 1e-5 relative error or 0.01 absolute (whichever is larger) */
static void expect_near_rel(float cpu, float gpu, int row) {
    float tol = std::max(0.01f, std::fabs(cpu) * 1e-5f);
    EXPECT_NEAR(cpu, gpu, tol)
        << "Row " << row << ": CPU=" << cpu << " GPU=" << gpu
        << " tol=" << tol;
}

extern "C" {
#include "turboquant/tq_gguf.h"

/* Internal Metal dispatch functions (not in public header) */
int tq_metal_available(void);
int tq_metal_matmul_gguf(float* out, const float* x, const void* weight,
                          tq_ggml_dtype weight_type, int out_dim, int in_dim);
}

/**
 * Zero-weight smoke test: all-zero IQ2_XXS blocks should produce zero output.
 *
 * IQ2_XXS format: 66 bytes per 256-element block
 *   - 2 bytes: FP16 scale (d)
 *   - 64 bytes: 8 sub-blocks of 8 bytes each (2B grid index + 6B signs/scale)
 *
 * With d=0 (zero scale), any grid values * 0 = 0, so output must be zero.
 */
TEST(MetalMatmul, IQ2_XXS_ZeroWeights) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    const int out_dim = 4;
    const int in_dim  = 256;

    /* IQ2_XXS: 66 bytes per 256 elements */
    const size_t block_size = 66;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    /* Allocate page-aligned weight buffer (Metal may require alignment) */
    uint8_t* weight = (uint8_t*)calloc(weight_bytes, 1);
    ASSERT_NE(weight, nullptr);
    memset(weight, 0, weight_bytes);

    /* Uniform input vector */
    float input[256];
    for (int i = 0; i < in_dim; i++) input[i] = 1.0f;

    float output_gpu[4] = {999.0f, 999.0f, 999.0f, 999.0f};
    float output_cpu[4] = {999.0f, 999.0f, 999.0f, 999.0f};

    /* GPU path */
    int rc = tq_metal_matmul_gguf(output_gpu, input, weight,
                                   TQ_GGML_TYPE_IQ2_XXS,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal matmul dispatch failed (returned " << rc << ")";

    /* CPU reference */
    tq_matmul_gguf(output_cpu, input, weight,
                   TQ_GGML_TYPE_IQ2_XXS, out_dim, in_dim);

    /* Both should be zero (or at least match) */
    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

/**
 * Small non-trivial test: 4 rows x 256 cols with known-pattern weights.
 *
 * We fill IQ2_XXS blocks with a simple repeating pattern and verify
 * CPU and GPU produce the same results. We do not need exact numerical
 * correctness — just GPU/CPU agreement.
 */
TEST(MetalMatmul, IQ2_XXS_SmallMatrix) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    const int out_dim = 4;
    const int in_dim  = 256;
    const size_t block_size = 66;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    /* Fill with deterministic non-zero pattern */
    uint8_t* weight = (uint8_t*)malloc(weight_bytes);
    ASSERT_NE(weight, nullptr);
    for (size_t i = 0; i < weight_bytes; i++) {
        weight[i] = (uint8_t)((i * 37 + 13) & 0xFF);
    }

    /* Random-ish input vector */
    float input[256];
    for (int i = 0; i < in_dim; i++) {
        input[i] = sinf((float)i * 0.1f);
    }

    float output_gpu[4] = {0};
    float output_cpu[4] = {0};

    /* CPU reference first (known to work) */
    tq_matmul_gguf(output_cpu, input, weight,
                   TQ_GGML_TYPE_IQ2_XXS, out_dim, in_dim);

    /* GPU path */
    int rc = tq_metal_matmul_gguf(output_gpu, input, weight,
                                   TQ_GGML_TYPE_IQ2_XXS,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal matmul dispatch failed";

    /* Compare GPU vs CPU */
    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

/**
 * Slightly larger matrix: 8 rows x 256 cols.
 * Tests that multi-row dispatch works correctly.
 */
TEST(MetalMatmul, IQ2_XXS_8Rows) {
    if (!tq_metal_available()) {
        GTEST_SKIP() << "Metal device not available on this machine";
    }

    const int out_dim = 8;
    const int in_dim  = 256;
    const size_t block_size = 66;
    const size_t weight_bytes = (size_t)out_dim * block_size;

    uint8_t* weight = (uint8_t*)malloc(weight_bytes);
    ASSERT_NE(weight, nullptr);
    for (size_t i = 0; i < weight_bytes; i++) {
        weight[i] = (uint8_t)((i * 53 + 7) & 0xFF);
    }

    float input[256];
    for (int i = 0; i < in_dim; i++) {
        input[i] = cosf((float)i * 0.05f) * 0.5f;
    }

    float output_gpu[8] = {0};
    float output_cpu[8] = {0};

    tq_matmul_gguf(output_cpu, input, weight,
                   TQ_GGML_TYPE_IQ2_XXS, out_dim, in_dim);

    int rc = tq_metal_matmul_gguf(output_gpu, input, weight,
                                   TQ_GGML_TYPE_IQ2_XXS,
                                   out_dim, in_dim);
    ASSERT_EQ(0, rc) << "Metal matmul dispatch failed";

    for (int i = 0; i < out_dim; i++) {
        expect_near_rel(output_cpu[i], output_gpu[i], i);
    }

    free(weight);
}

#endif /* TQ_HAS_METAL */
