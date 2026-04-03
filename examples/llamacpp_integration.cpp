/**
 * quant.cpp — llama.cpp integration example
 *
 * Shows how to use quant.cpp for KV cache compression
 * within a llama.cpp-based application.
 */
#include <cstdio>
extern "C" {
#include "turboquant/turboquant.h"
}

int main() {
    /* Initialize quant.cpp context */
    tq_context_t* ctx = nullptr;
    tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
    if (status != TQ_OK) {
        fprintf(stderr, "Init failed: %s\n", tq_status_string(status));
        return 1;
    }

    printf("quant.cpp v%s initialized\n", TQ_VERSION_STRING);
    printf("Backend: CPU\n\n");

    /* List available quantization types */
    printf("Available KV cache quantization types:\n");
    for (int t = 0; t < TQ_TYPE_COUNT; t++) {
        printf("  %-12s  %.2f bpe  block=%zu\n",
               tq_type_name((tq_type)t),
               tq_type_bpe((tq_type)t),
               tq_type_block_size((tq_type)t));
    }

    /* Recommend strategy for 3-bit target */
    tq_type recommended = tq_recommend_strategy(128, 3, 0.99f);
    printf("\nRecommended for 3-bit/99%% quality: %s\n", tq_type_name(recommended));

    /*
     * In a real llama.cpp integration:
     * 1. Register TQ types: tq_ggml_register_types()
     * 2. Use --kv-cache-type turbo3 CLI option
     * 3. In attention layer, call tq_quantize_keys/tq_attention
     */

    tq_free(ctx);
    return 0;
}
