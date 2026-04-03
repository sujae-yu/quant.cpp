/* quant.cpp — Minimal Example (10 lines of logic) */
#include "turboquant/turboquant.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    tq_context_t* ctx;
    tq_init(&ctx, TQ_BACKEND_CPU);

    /* One key, one query */
    float key[128], query[128];
    for (int i = 0; i < 128; i++) {
        key[i] = sinf(i * 0.1f);
        query[i] = cosf(i * 0.1f);
    }

    /* Quantize (7.5x smaller) and compute attention */
    block_tq_uniform_4b block;
    tq_quantize_keys(ctx, key, 1, 128, TQ_TYPE_UNIFORM_4B, &block, sizeof(block));

    float score;
    tq_attention(ctx, query, &block, 1, 128, TQ_TYPE_UNIFORM_4B, &score);
    printf("Attention score: %.6f\n", score);

    tq_free(ctx);
    return 0;
}
