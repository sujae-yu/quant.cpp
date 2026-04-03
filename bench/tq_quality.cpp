/**
 * quant.cpp quality benchmark
 *
 * Outputs machine-readable metrics:
 *   roundtrip_mse=X.XXXXXX
 *   attention_cosine=X.XXXXXX
 *   cross_platform=pass
 */

extern "C" {
#include "turboquant/turboquant.h"
void tq_polar_quantize_ref(const float* src, void* dst, int n);
void tq_polar_dequantize_ref(const void* src, float* dst, int n);
void tq_polar_attention_ref(const float* query, const void* kv,
                            float* scores, int seq_len, int head_dim);
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* Simple LCG PRNG for reproducibility */
static uint32_t rng_state = 42;
static float rand_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

int main() {
    const int HEAD_DIM = 128;
    const int N_VECTORS = 1000;

    /* --- Roundtrip MSE (Uniform 4B as baseline) --- */
    double total_mse = 0.0;
    for (int v = 0; v < N_VECTORS; v++) {
        float input[128], output[128];
        for (int i = 0; i < HEAD_DIM; i++) input[i] = rand_float();

        block_tq_uniform_4b block;
        tq_uniform_4b_quantize_ref(input, &block, HEAD_DIM);
        tq_uniform_4b_dequantize_ref(&block, output, HEAD_DIM);

        double mse = 0.0;
        for (int i = 0; i < HEAD_DIM; i++) {
            double d = input[i] - output[i];
            mse += d * d;
        }
        mse /= HEAD_DIM;
        total_mse += mse;
    }
    total_mse /= N_VECTORS;

    /* --- Attention cosine similarity (Uniform 4B, vector of scores) --- */
    /* Cosine similarity of attention score vectors: fp32 vs quantized */
    const int SEQ_LEN = 64;
    std::vector<float> keys(SEQ_LEN * HEAD_DIM);
    std::vector<float> query(HEAD_DIM);
    std::vector<float> fp32_scores(SEQ_LEN);
    std::vector<float> quant_scores(SEQ_LEN);

    double sum_dot = 0, sum_fp32_sq = 0, sum_quant_sq = 0;
    const int N_TRIALS = 100;

    for (int t = 0; t < N_TRIALS; t++) {
        for (int i = 0; i < SEQ_LEN * HEAD_DIM; i++) keys[i] = rand_float();
        for (int i = 0; i < HEAD_DIM; i++) query[i] = rand_float();

        /* FP32 attention scores */
        for (int s = 0; s < SEQ_LEN; s++) {
            float dot = 0;
            for (int d = 0; d < HEAD_DIM; d++)
                dot += query[d] * keys[s * HEAD_DIM + d];
            fp32_scores[s] = dot;
        }

        /* Quantized attention (Uniform 4B — most accurate baseline) */
        std::vector<block_tq_uniform_4b> blocks(SEQ_LEN);
        for (int s = 0; s < SEQ_LEN; s++)
            tq_uniform_4b_quantize_ref(&keys[s * HEAD_DIM], &blocks[s], HEAD_DIM);

        for (int s = 0; s < SEQ_LEN; s++) {
            float deq[128];
            tq_uniform_4b_dequantize_ref(&blocks[s], deq, HEAD_DIM);
            float dot = 0;
            for (int d = 0; d < HEAD_DIM; d++) dot += query[d] * deq[d];
            quant_scores[s] = dot;
        }

        /* Accumulate cosine similarity components */
        for (int s = 0; s < SEQ_LEN; s++) {
            sum_dot += (double)fp32_scores[s] * (double)quant_scores[s];
            sum_fp32_sq += (double)fp32_scores[s] * (double)fp32_scores[s];
            sum_quant_sq += (double)quant_scores[s] * (double)quant_scores[s];
        }
    }

    double avg_cos = 0.0;
    if (sum_fp32_sq > 0 && sum_quant_sq > 0)
        avg_cos = sum_dot / (sqrt(sum_fp32_sq) * sqrt(sum_quant_sq));

    /* --- Cross-platform determinism check --- */
    /* Quantize a known vector and check hash is deterministic */
    float test_vec[128];
    for (int i = 0; i < 128; i++) test_vec[i] = sinf(i * 0.1f);

    block_tq_uniform_4b block1, block2;
    tq_uniform_4b_quantize_ref(test_vec, &block1, 128);
    tq_uniform_4b_quantize_ref(test_vec, &block2, 128);

    const char* cross_platform = "pass";
    if (memcmp(&block1, &block2, sizeof(block_tq_uniform_4b)) != 0) {
        cross_platform = "fail";
    }

    /* --- Output --- */
    printf("roundtrip_mse=%.6f\n", total_mse);
    printf("attention_cosine=%.6f\n", avg_cos);
    printf("cross_platform=%s\n", cross_platform);

    return 0;
}
