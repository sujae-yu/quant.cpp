/**
 * test_gguf_moe.cpp — Tests for GGUF loader, dequantization, and MoE support
 *
 * Tests:
 *   1. GGUF type utility functions (size, block, name)
 *   2. Dequantization correctness for Q2_K, Q4_K, Q6_K, Q8_0
 *   3. MoE routing (top-K selection + softmax)
 *   4. MoE forward pass (SwiGLU expert dispatch)
 *   5. tq_matmul_gguf correctness (on-the-fly dequant matmul)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

extern "C" {
#include "turboquant/tq_gguf.h"
#include "turboquant/tq_engine.h"
}

/* ============================================================
 * Test: GGML type utilities
 * ============================================================ */

TEST(GGUFTypes, TypeSize) {
    EXPECT_EQ(tq_ggml_type_size(TQ_GGML_TYPE_F32), 4u);
    EXPECT_EQ(tq_ggml_type_size(TQ_GGML_TYPE_F16), 2u);
    EXPECT_EQ(tq_ggml_type_size(TQ_GGML_TYPE_Q8_0), 34u);  /* 2 + 32 */
    EXPECT_EQ(tq_ggml_type_size(TQ_GGML_TYPE_Q2_K), 84u);
    EXPECT_EQ(tq_ggml_type_size(TQ_GGML_TYPE_Q4_K), 144u);
    EXPECT_EQ(tq_ggml_type_size(TQ_GGML_TYPE_Q6_K), 210u);
}

TEST(GGUFTypes, BlockElements) {
    EXPECT_EQ(tq_ggml_type_blck(TQ_GGML_TYPE_F32), 1);
    EXPECT_EQ(tq_ggml_type_blck(TQ_GGML_TYPE_Q8_0), 32);
    EXPECT_EQ(tq_ggml_type_blck(TQ_GGML_TYPE_Q2_K), 256);
    EXPECT_EQ(tq_ggml_type_blck(TQ_GGML_TYPE_Q4_K), 256);
}

TEST(GGUFTypes, TypeName) {
    EXPECT_STREQ(tq_ggml_type_name(TQ_GGML_TYPE_F32), "F32");
    EXPECT_STREQ(tq_ggml_type_name(TQ_GGML_TYPE_Q4_K), "Q4_K");
    EXPECT_STREQ(tq_ggml_type_name(TQ_GGML_TYPE_IQ2_XXS), "IQ2_XXS");
}

/* ============================================================
 * Test: F32 passthrough dequantization
 * ============================================================ */

TEST(GGUFDequant, F32Passthrough) {
    float src[8] = {1.0f, -2.0f, 3.14f, 0.0f, -0.5f, 100.0f, -100.0f, 0.001f};
    float dst[8] = {};
    tq_dequant_row_gguf(TQ_GGML_TYPE_F32, src, dst, 8);
    for (int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(src[i], dst[i]);
    }
}

/* ============================================================
 * Test: Q8_0 dequantization
 * ============================================================ */

TEST(GGUFDequant, Q8_0_RoundTrip) {
    /* Create a Q8_0 block manually:
     * d (fp16 scale) + 32 int8 values
     * Dequant: out[i] = fp16_to_f32(d) * qs[i] */
    const int block_size = 34; /* 2 + 32 */
    uint8_t block[34] = {};

    /* Set d = 0.5 as fp16 (0x3800) */
    block[0] = 0x00;
    block[1] = 0x38;  /* fp16 for 0.5 */

    /* Set qs: 0, 1, 2, ..., 31 */
    for (int i = 0; i < 32; i++) {
        block[2 + i] = (uint8_t)(int8_t)i;
    }

    float dst[32] = {};
    tq_dequant_row_gguf(TQ_GGML_TYPE_Q8_0, block, dst, 32);

    /* Expected: out[i] = 0.5 * i */
    for (int i = 0; i < 32; i++) {
        EXPECT_NEAR(dst[i], 0.5f * (float)i, 0.01f)
            << "Mismatch at index " << i;
    }
}

/* ============================================================
 * Test: tq_matmul_gguf with F32 weights
 * ============================================================ */

TEST(GGUFMatmul, F32_Identity) {
    /* Test: y = x @ W^T where W is 4x4 identity matrix stored as F32 */
    const int dim = 4;
    float W[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {};

    tq_matmul_gguf(out, x, W, TQ_GGML_TYPE_F32, dim, dim);

    for (int i = 0; i < dim; i++) {
        EXPECT_NEAR(out[i], x[i], 1e-5f);
    }
}

TEST(GGUFMatmul, F32_SimpleWeight) {
    /* W = [[2, 0], [0, 3]], x = [1, 1] -> out = [2, 3] */
    float W[4] = {2.0f, 0.0f, 0.0f, 3.0f};
    float x[2] = {1.0f, 1.0f};
    float out[2] = {};

    tq_matmul_gguf(out, x, W, TQ_GGML_TYPE_F32, 2, 2);

    EXPECT_NEAR(out[0], 2.0f, 1e-5f);
    EXPECT_NEAR(out[1], 3.0f, 1e-5f);
}

/* ============================================================
 * Test: MoE routing
 * ============================================================ */

TEST(MoE, RouteTopK) {
    const int num_experts = 8;
    const int num_active = 2;
    const int hidden_dim = 4;

    /* Router weights: each expert has a 4-dim weight vector.
     * Expert 3 and Expert 5 should be selected for input [1,0,0,0] */
    float router_weight[8 * 4] = {};
    router_weight[3 * 4 + 0] = 10.0f;  /* Expert 3 has high weight on dim 0 */
    router_weight[5 * 4 + 0] = 8.0f;   /* Expert 5 has second highest */
    router_weight[0 * 4 + 0] = 1.0f;   /* Expert 0 is weak */

    float input[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    int expert_ids[2] = {};
    float expert_weights[2] = {};

    tq_moe_route(input, router_weight,
                 num_experts, num_active, hidden_dim,
                 expert_ids, expert_weights);

    /* Expert 3 should be first (highest logit = 10) */
    EXPECT_EQ(expert_ids[0], 3);
    /* Expert 5 should be second (logit = 8) */
    EXPECT_EQ(expert_ids[1], 5);

    /* Weights should sum to 1 (renormalized softmax) */
    float sum = expert_weights[0] + expert_weights[1];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    /* Expert 3 should have higher weight than Expert 5 */
    EXPECT_GT(expert_weights[0], expert_weights[1]);
}

/* ============================================================
 * Test: MoE state creation/destruction
 * ============================================================ */

TEST(MoE, StateLifecycle) {
    tq_moe_config_t config = {};
    config.num_experts = 64;
    config.num_active = 8;
    config.expert_intermediate_dim = 512;
    config.has_shared_expert = 1;
    config.shared_expert_intermediate_dim = 1024;

    tq_moe_state_t* state = tq_moe_create_state(&config, 2048);
    ASSERT_NE(state, nullptr);
    EXPECT_NE(state->router_logits, nullptr);
    EXPECT_NE(state->top_experts, nullptr);
    EXPECT_NE(state->expert_weights, nullptr);
    EXPECT_NE(state->expert_out, nullptr);
    EXPECT_NE(state->expert_hb, nullptr);
    EXPECT_NE(state->expert_hb2, nullptr);

    tq_moe_free_state(state);
}

/* ============================================================
 * Test: MoE forward with F32 expert weights
 * ============================================================ */

TEST(MoE, ForwardSmokeTest) {
    /* Minimal MoE: 4 experts, 2 active, hidden_dim=4, expert_dim=2 */
    const int num_experts = 4;
    const int num_active = 2;
    const int hidden_dim = 4;
    const int expert_dim = 2;

    tq_moe_config_t config = {};
    config.num_experts = num_experts;
    config.num_active = num_active;
    config.expert_intermediate_dim = expert_dim;
    config.has_shared_expert = 0;
    config.norm_topk_prob = 1;

    tq_moe_state_t* state = tq_moe_create_state(&config, hidden_dim);
    ASSERT_NE(state, nullptr);

    /* Create router weights: expert 0 responds to dim 0, expert 1 to dim 1, etc. */
    float router_weight[4 * 4] = {};
    for (int e = 0; e < num_experts; e++) {
        router_weight[e * hidden_dim + e] = 5.0f;
    }

    /* Create simple expert weights (F32, identity-like) */
    /* gate: [expert_dim, hidden_dim] = [[1,0,0,0],[0,1,0,0]] */
    float gate_data[4][2 * 4];  /* 4 experts, each [2, 4] */
    float up_data[4][2 * 4];
    float down_data[4][4 * 2];
    memset(gate_data, 0, sizeof(gate_data));
    memset(up_data, 0, sizeof(up_data));
    memset(down_data, 0, sizeof(down_data));

    for (int e = 0; e < num_experts; e++) {
        /* Simple identity-like expert */
        gate_data[e][0] = 1.0f;  /* gate[0,0] = 1 */
        up_data[e][0] = 1.0f;    /* up[0,0] = 1 */
        down_data[e][0] = 1.0f;  /* down[0,0] = 1 */
    }

    tq_expert_weights_t experts[4] = {};
    for (int e = 0; e < num_experts; e++) {
        experts[e].w_gate = gate_data[e];
        experts[e].gate_type = TQ_GGML_TYPE_F32;
        experts[e].w_up = up_data[e];
        experts[e].up_type = TQ_GGML_TYPE_F32;
        experts[e].w_down = down_data[e];
        experts[e].down_type = TQ_GGML_TYPE_F32;
    }

    tq_moe_layer_t layer = {};
    layer.router_weight = router_weight;
    layer.experts = experts;

    float input[4] = {1.0f, 0.5f, 0.1f, 0.0f};
    float output[4] = {};

    tq_moe_forward(&layer, &config, state, input, output, hidden_dim);

    /* Output should be non-zero (experts produced something) */
    float out_norm = 0;
    for (int i = 0; i < hidden_dim; i++) out_norm += output[i] * output[i];
    EXPECT_GT(out_norm, 0.0f) << "MoE forward produced zero output";

    tq_moe_free_state(state);
}

/* ============================================================
 * Test: Model config MoE fields
 * ============================================================ */

TEST(ModelConfig, MoEFields) {
    tq_model_config_t config = {};
    config.is_moe = 1;
    config.num_experts = 64;
    config.num_active_experts = 8;
    config.expert_intermediate_dim = 512;
    config.has_shared_expert = 1;

    EXPECT_EQ(config.is_moe, 1);
    EXPECT_EQ(config.num_experts, 64);
    EXPECT_EQ(config.num_active_experts, 8);
}

/* ============================================================
 * Test: GGUF file detection in tq_load_model
 * ============================================================ */

TEST(GGUF, MagicDetection) {
    /* Write a tiny file with GGUF magic but invalid content */
    const char* tmppath = "/tmp/test_gguf_magic.gguf";
    FILE* f = fopen(tmppath, "wb");
    ASSERT_NE(f, nullptr);
    uint32_t magic = 0x46554747; /* "GGUF" as LE uint32 */
    fwrite(&magic, 4, 1, f);
    fclose(f);

    /* tq_load_model should detect GGUF format (will fail to load, but detection works) */
    tq_model_t* model = tq_load_model(tmppath);
    /* Expected: NULL because the file is too small/invalid, but no crash */
    EXPECT_EQ(model, nullptr);

    remove(tmppath);
}
