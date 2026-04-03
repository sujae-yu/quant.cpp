/**
 * quant.cpp — Compare all quantization types
 */
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
extern "C" {
#include "turboquant/turboquant.h"
}

int main() {
    tq_context_t* ctx = nullptr;
    tq_init(&ctx, TQ_BACKEND_CPU);

    const int head_dim = 128;
    float key[128], query[128];
    for (int i = 0; i < head_dim; i++) {
        key[i] = sinf(i * 0.1f);
        query[i] = cosf(i * 0.1f);
    }

    float fp32_dot = 0;
    for (int i = 0; i < head_dim; i++) fp32_dot += query[i] * key[i];
    printf("FP32 dot product: %.6f\n\n", fp32_dot);

    tq_type types[] = {TQ_TYPE_POLAR_3B, TQ_TYPE_POLAR_4B, TQ_TYPE_QJL_1B,
                       TQ_TYPE_TURBO_3B, TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B};

    for (auto type : types) {
        size_t sz = tq_quantize_keys_size(1, head_dim, type);
        void* buf = malloc(sz);
        tq_quantize_keys(ctx, key, 1, head_dim, type, buf, sz);

        float deq[128];
        tq_dequantize_keys(ctx, buf, 1, head_dim, type, deq);

        double mse = 0;
        for (int i = 0; i < head_dim; i++) {
            double d = key[i] - deq[i];
            mse += d * d;
        }
        mse /= head_dim;

        float score = 0;
        tq_attention(ctx, query, buf, 1, head_dim, type, &score);

        printf("%-12s  bpe=%.2f  mse=%.6f  attn=%.6f  err=%.4f\n",
               tq_type_name(type), tq_type_bpe(type), mse,
               score, fabsf(score - fp32_dot));
        free(buf);
    }

    tq_free(ctx);
    return 0;
}
