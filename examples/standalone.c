/**
 * quant.cpp — Standalone C example
 * Demonstrates basic quantize → attention workflow
 */
#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    tq_context_t* ctx = NULL;
    tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
    if (status != TQ_OK) {
        fprintf(stderr, "Failed to init: %s\n", tq_status_string(status));
        return 1;
    }

    const int head_dim = 128;
    const int seq_len = 4;

    /* Generate sample keys and query */
    float keys[4][128], query[128];
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < head_dim; j++)
            keys[i][j] = sinf((i + 1) * j * 0.01f);
    for (int j = 0; j < head_dim; j++)
        query[j] = cosf(j * 0.05f);

    /* Quantize keys */
    tq_type type = TQ_TYPE_POLAR_4B;
    size_t buf_size = tq_quantize_keys_size(seq_len, head_dim, type);
    void* quantized = malloc(buf_size);

    status = tq_quantize_keys(ctx, (const float*)keys, seq_len, head_dim,
                              type, quantized, buf_size);
    printf("Quantize: %s\n", tq_status_string(status));
    printf("Type: %s (%.2f bits/element)\n", tq_type_name(type), tq_type_bpe(type));
    printf("Compression: %.1fx\n", 32.0f / tq_type_bpe(type));

    /* Compute attention scores */
    float scores[4];
    status = tq_attention(ctx, query, quantized, seq_len, head_dim, type, scores);
    printf("Attention: %s\n", tq_status_string(status));
    for (int i = 0; i < seq_len; i++)
        printf("  score[%d] = %.4f\n", i, scores[i]);

    free(quantized);
    tq_free(ctx);
    return 0;
}
