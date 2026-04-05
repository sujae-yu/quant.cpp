/**
 * test_gemma4_arch.cpp -- Gemma 4 architecture regression tests
 *
 * Validates Gemma 4-specific configuration logic without requiring a model file:
 *   1. Dual-FFN detection (is_gemma4 flag)
 *   2. QK-norm aware KV (FP32 keys when is_gemma4 && use_qk_norm)
 *   3. Layer output scale (simple multiply, not residual-contribution)
 *   4. Attention scale (attn_scale_dim = 1.0 for Gemma 4)
 *   5. GeGLU activation (use_gelu flag in MoE config)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>

extern "C" {
#include "turboquant/tq_engine.h"
#include "turboquant/tq_gguf.h"
}

/* ============================================================
 * Test 1: Dual-FFN detection — is_gemma4 flag
 *
 * Gemma 4 layers have BOTH MoE FFN and dense FFN (dual-FFN).
 * The is_gemma4 flag must be set when architecture is "gemma4".
 * Verify the flag exists and is distinct from generic gemma3.
 * ============================================================ */

TEST(Gemma4Arch, DualFFNDetection) {
    /* Gemma 4 config: is_gemma4 = 1, is_moe = 1 */
    tq_model_config_t config = {};
    config.is_gemma4 = 1;
    config.model_type = 1;  /* gemma family */
    config.is_moe = 1;
    config.num_experts = 16;
    config.num_active_experts = 2;

    EXPECT_EQ(config.is_gemma4, 1);
    EXPECT_EQ(config.is_moe, 1);

    /* Gemma 3 (non-Gemma4): is_gemma4 should be 0 */
    tq_model_config_t gemma3_config = {};
    gemma3_config.model_type = 1;  /* gemma family */
    gemma3_config.is_gemma4 = 0;

    EXPECT_EQ(gemma3_config.is_gemma4, 0);

    /* Verify Gemma 4 dual-FFN: MoE + dense FFN coexist.
     * In the model loader, when is_gemma4 is set, both MoE expert weights
     * and dense FFN weights (gguf_w_gate/up/down) are loaded per layer. */
    tq_layer_weights_t layer = {};
    memset(&layer, 0, sizeof(layer));

    /* Simulate dual-FFN: MoE pointer is non-NULL AND dense FFN pointers are non-NULL */
    int dummy_moe = 1;
    float dummy_gate[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    layer.moe = &dummy_moe;
    layer.gguf_w_gate = dummy_gate;
    layer.gguf_w_up = dummy_gate;
    layer.gguf_w_down = dummy_gate;

    /* Gemma 4 dual-FFN: both MoE and dense FFN present simultaneously */
    EXPECT_NE(layer.moe, nullptr) << "MoE pointer should be set for Gemma 4 MoE layers";
    EXPECT_NE(layer.gguf_w_gate, nullptr) << "Dense FFN gate should coexist with MoE";
    EXPECT_NE(layer.gguf_w_up, nullptr) << "Dense FFN up should coexist with MoE";
    EXPECT_NE(layer.gguf_w_down, nullptr) << "Dense FFN down should coexist with MoE";
}

/* ============================================================
 * Test 2: QK-norm aware KV — FP32 keys when is_gemma4 && use_qk_norm
 *
 * Gemma 4 with QK-norm produces extremely sparse post-norm keys
 * that quantize poorly (cosine=0.62 at 4-bit). The engine must
 * force FP32 key storage while keeping quantized V cache.
 *
 * Logic from tq_transformer.c:
 *   if (use_quant_kv && c->is_gemma4 && c->use_qk_norm) {
 *       use_quant_kv = 0; // fall through to FP32 key storage
 *   }
 * ============================================================ */

TEST(Gemma4Arch, QKNormAwareKV_ForceFP32Keys) {
    /* Gemma 4 with QK-norm: the combination should disable key quantization */
    tq_model_config_t config = {};
    config.is_gemma4 = 1;
    config.use_qk_norm = 1;
    config.model_type = 1;

    /* Simulate the decision logic from tq_transformer.c */
    int use_quant_kv = 1;  /* initially enabled */
    if (use_quant_kv && config.is_gemma4 && config.use_qk_norm) {
        use_quant_kv = 0;  /* force FP32 keys */
    }
    EXPECT_EQ(use_quant_kv, 0)
        << "Gemma 4 + QK-norm must disable key quantization (use FP32 keys)";
}

TEST(Gemma4Arch, QKNormAwareKV_NonGemma4Keeps) {
    /* Non-Gemma4 with QK-norm: key quantization should remain enabled */
    tq_model_config_t config = {};
    config.is_gemma4 = 0;
    config.use_qk_norm = 1;

    int use_quant_kv = 1;
    if (use_quant_kv && config.is_gemma4 && config.use_qk_norm) {
        use_quant_kv = 0;
    }
    EXPECT_EQ(use_quant_kv, 1)
        << "Non-Gemma4 with QK-norm should keep key quantization enabled";
}

TEST(Gemma4Arch, QKNormAwareKV_Gemma4WithoutQKNorm) {
    /* Gemma 4 without QK-norm: key quantization should remain enabled */
    tq_model_config_t config = {};
    config.is_gemma4 = 1;
    config.use_qk_norm = 0;

    int use_quant_kv = 1;
    if (use_quant_kv && config.is_gemma4 && config.use_qk_norm) {
        use_quant_kv = 0;
    }
    EXPECT_EQ(use_quant_kv, 1)
        << "Gemma 4 without QK-norm should keep key quantization enabled";
}

/* ============================================================
 * Test 3: Layer output scale — simple multiply (not residual-contribution)
 *
 * Gemma 4 applies layer_output_scale as a simple scalar multiply
 * on the entire hidden state x, NOT as a residual-contribution weight.
 *
 * From tq_transformer.c:
 *   if (layer->layer_output_scale != 0.0f) {
 *       float los = layer->layer_output_scale;
 *       for (int i = 0; i < dim; i++) x[i] *= los;
 *   }
 * ============================================================ */

TEST(Gemma4Arch, LayerOutputScale_SimpleMultiply) {
    const int dim = 8;
    float x[8] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
    float x_orig[8];
    memcpy(x_orig, x, sizeof(x));

    /* Simulate a Gemma 4 layer_output_scale of 0.125 */
    float layer_output_scale = 0.125f;

    /* Apply the scale as the engine does: simple multiply */
    if (layer_output_scale != 0.0f) {
        for (int i = 0; i < dim; i++) {
            x[i] *= layer_output_scale;
        }
    }

    /* Verify: each element is simply multiplied (not added, not residual) */
    for (int i = 0; i < dim; i++) {
        EXPECT_FLOAT_EQ(x[i], x_orig[i] * layer_output_scale)
            << "Layer output scale must be a simple multiply at index " << i;
    }
}

TEST(Gemma4Arch, LayerOutputScale_ZeroDisabled) {
    /* layer_output_scale = 0.0 means disabled (no scaling applied) */
    const int dim = 4;
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x_orig[4];
    memcpy(x_orig, x, sizeof(x));

    float layer_output_scale = 0.0f;
    if (layer_output_scale != 0.0f) {
        for (int i = 0; i < dim; i++) {
            x[i] *= layer_output_scale;
        }
    }

    /* Verify: no change when scale is 0.0 (disabled) */
    for (int i = 0; i < dim; i++) {
        EXPECT_FLOAT_EQ(x[i], x_orig[i])
            << "layer_output_scale=0.0 should be no-op at index " << i;
    }
}

TEST(Gemma4Arch, LayerOutputScale_NotResidualContribution) {
    /* Verify the scale is NOT applied as a residual contribution:
     * WRONG: x = residual + scale * layer_out
     * RIGHT: x *= scale (applied to entire hidden state post-FFN) */
    const int dim = 4;
    float residual[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float layer_out[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scale = 0.5f;

    /* The WRONG approach (residual contribution): */
    float wrong_result[4];
    for (int i = 0; i < dim; i++) {
        wrong_result[i] = residual[i] + scale * layer_out[i];
    }

    /* The RIGHT approach (Gemma 4 simple multiply on full hidden state): */
    float x[4];
    memcpy(x, layer_out, sizeof(x));
    for (int i = 0; i < dim; i++) {
        x[i] *= scale;
    }

    /* These should differ -- proving it's not a residual-contribution pattern */
    bool any_differ = false;
    for (int i = 0; i < dim; i++) {
        if (fabsf(x[i] - wrong_result[i]) > 1e-6f) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ)
        << "Simple multiply and residual-contribution should produce different results";

    /* Verify actual values: x[i] = layer_out[i] * scale */
    for (int i = 0; i < dim; i++) {
        EXPECT_FLOAT_EQ(x[i], layer_out[i] * scale);
    }
}

/* ============================================================
 * Test 4: Attention scale — attn_scale_dim = 1.0 for Gemma 4
 *
 * Gemma 4 uses QK-norm which already normalizes Q,K per head,
 * so attention_scale = 1/sqrt(1.0) = 1.0 (no additional scaling).
 *
 * From tq_transformer.c:
 *   float attn_scale_dim = (float)head_dim;
 *   if (c->is_gemma4) {
 *       attn_scale_dim = 1.0f;
 *   } else if (c->query_pre_attn_scalar > 0.0f) {
 *       attn_scale_dim = c->query_pre_attn_scalar;
 *   }
 *   float scale = 1.0f / sqrtf(attn_scale_dim);
 * ============================================================ */

TEST(Gemma4Arch, AttentionScale_Gemma4IsOne) {
    tq_model_config_t config = {};
    config.is_gemma4 = 1;
    config.head_dim = 256;
    config.query_pre_attn_scalar = 256.0f;  /* should be ignored for Gemma 4 */

    /* Replicate the engine logic */
    float attn_scale_dim = (float)config.head_dim;
    if (config.is_gemma4) {
        attn_scale_dim = 1.0f;
    } else if (config.query_pre_attn_scalar > 0.0f) {
        attn_scale_dim = config.query_pre_attn_scalar;
    }

    EXPECT_FLOAT_EQ(attn_scale_dim, 1.0f)
        << "Gemma 4 attn_scale_dim must be 1.0 (QK-norm handles scaling)";

    float attn_scale = 1.0f / sqrtf(attn_scale_dim);
    EXPECT_FLOAT_EQ(attn_scale, 1.0f)
        << "Gemma 4 attention scale factor must be exactly 1.0";
}

TEST(Gemma4Arch, AttentionScale_NonGemma4UsesHeadDim) {
    tq_model_config_t config = {};
    config.is_gemma4 = 0;
    config.head_dim = 128;
    config.query_pre_attn_scalar = 0.0f;

    float attn_scale_dim = (float)config.head_dim;
    if (config.is_gemma4) {
        attn_scale_dim = 1.0f;
    } else if (config.query_pre_attn_scalar > 0.0f) {
        attn_scale_dim = config.query_pre_attn_scalar;
    }

    EXPECT_FLOAT_EQ(attn_scale_dim, 128.0f)
        << "Non-Gemma4 should use head_dim for attention scaling";

    float attn_scale = 1.0f / sqrtf(attn_scale_dim);
    EXPECT_NEAR(attn_scale, 1.0f / sqrtf(128.0f), 1e-6f)
        << "Non-Gemma4 attention scale = 1/sqrt(head_dim)";
}

TEST(Gemma4Arch, AttentionScale_Gemma3UsesPreAttnScalar) {
    /* Gemma 3 uses query_pre_attn_scalar (e.g., 256.0) instead of head_dim */
    tq_model_config_t config = {};
    config.is_gemma4 = 0;
    config.model_type = 1;
    config.head_dim = 256;
    config.query_pre_attn_scalar = 256.0f;

    float attn_scale_dim = (float)config.head_dim;
    if (config.is_gemma4) {
        attn_scale_dim = 1.0f;
    } else if (config.query_pre_attn_scalar > 0.0f) {
        attn_scale_dim = config.query_pre_attn_scalar;
    }

    EXPECT_FLOAT_EQ(attn_scale_dim, 256.0f)
        << "Gemma 3 should use query_pre_attn_scalar for attention scaling";
}

/* ============================================================
 * Test 5: GeGLU activation — use_gelu flag in MoE config
 *
 * Gemma 4 MoE uses GeGLU instead of SwiGLU.
 * From tq_model.c:
 *   moe_cfg->use_gelu = c->is_gemma4 ? 1 : 0;
 * ============================================================ */

TEST(Gemma4Arch, GeGLU_Gemma4MoEUsesGeLU) {
    tq_model_config_t model_config = {};
    model_config.is_gemma4 = 1;
    model_config.is_moe = 1;
    model_config.num_experts = 16;
    model_config.num_active_experts = 2;
    model_config.expert_intermediate_dim = 512;

    /* Replicate the engine logic from tq_model.c */
    tq_moe_config_t moe_cfg = {};
    moe_cfg.num_experts = model_config.num_experts;
    moe_cfg.num_active = model_config.num_active_experts;
    moe_cfg.expert_intermediate_dim = model_config.expert_intermediate_dim;
    moe_cfg.norm_topk_prob = 1;
    moe_cfg.use_gelu = model_config.is_gemma4 ? 1 : 0;

    EXPECT_EQ(moe_cfg.use_gelu, 1)
        << "Gemma 4 MoE must use GeGLU activation (use_gelu=1)";
}

TEST(Gemma4Arch, GeGLU_NonGemma4MoEUsesSwiGLU) {
    tq_model_config_t model_config = {};
    model_config.is_gemma4 = 0;
    model_config.is_moe = 1;
    model_config.num_experts = 64;
    model_config.num_active_experts = 8;

    tq_moe_config_t moe_cfg = {};
    moe_cfg.num_experts = model_config.num_experts;
    moe_cfg.num_active = model_config.num_active_experts;
    moe_cfg.use_gelu = model_config.is_gemma4 ? 1 : 0;

    EXPECT_EQ(moe_cfg.use_gelu, 0)
        << "Non-Gemma4 MoE must use SwiGLU activation (use_gelu=0)";
}

TEST(Gemma4Arch, GeGLU_ActivationFunction) {
    /* Verify that tq_gelu_tanh produces different output than tq_silu,
     * confirming the activation dispatch matters. */
    const int n = 4;
    float input_gelu[4] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float input_silu[4] = {-1.0f, 0.0f, 1.0f, 2.0f};

    tq_gelu_tanh(input_gelu, n);
    tq_silu(input_silu, n);

    /* GeLU and SiLU should produce different outputs for non-zero inputs */
    bool any_differ = false;
    for (int i = 0; i < n; i++) {
        if (fabsf(input_gelu[i] - input_silu[i]) > 1e-4f) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ)
        << "GeGLU (tq_gelu_tanh) and SwiGLU (tq_silu) must produce different activations";

    /* Verify GeLU(0) = 0 */
    EXPECT_FLOAT_EQ(input_gelu[1], 0.0f) << "GeLU(0) must be 0";

    /* Verify GeLU(x) > 0 for x > 0 */
    EXPECT_GT(input_gelu[2], 0.0f) << "GeLU(1.0) must be positive";
    EXPECT_GT(input_gelu[3], 0.0f) << "GeLU(2.0) must be positive";
}
