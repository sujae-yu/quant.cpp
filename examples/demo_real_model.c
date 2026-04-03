/**
 * quant.cpp — Real Model KV Cache Compression Demo
 *
 * Simulates KV cache compression for actual LLM architectures:
 *   - Qwen3.5-0.5B  (24 layers, 14 KV heads, head_dim=64)
 *   - Llama-3.2-1B  (16 layers, 8 KV heads,  head_dim=64)
 *   - Llama-3.2-3B  (28 layers, 8 KV heads,  head_dim=128)
 *   - Phi-3-mini     (32 layers, 32 KV heads, head_dim=96)
 *
 * Shows: memory savings, quantization quality, and throughput
 * on YOUR machine, with data that mimics real attention patterns.
 */

#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================
 * Model specifications
 * ============================================================ */

typedef struct {
    const char* name;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int vocab_size;     /* for display only */
    float param_b;      /* billions of params */
} model_spec_t;

static const model_spec_t MODELS[] = {
    {"Qwen3.5-0.5B",   24, 14,  64, 151936, 0.5f},
    {"Llama-3.2-1B",    16,  8,  64, 128256, 1.2f},
    {"Llama-3.2-3B",    28,  8, 128, 128256, 3.2f},
    {"Phi-3-mini-4k",   32, 32,  96,  32064, 3.8f},
};
#define NUM_MODELS (sizeof(MODELS) / sizeof(MODELS[0]))

/* ============================================================
 * Realistic KV cache data generator
 * ============================================================ */

static uint32_t rng_state = 12345;

static float randf(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

/* Generate data mimicking real attention key distributions:
 * - Most values near 0 (Gaussian-like)
 * - A few outliers (heavy tails, common in LLM keys)
 */
static void generate_realistic_keys(float* keys, int n, int head_dim) {
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < head_dim; d++) {
            /* Box-Muller for approximate Gaussian */
            float u1 = (randf() + 1.0f) * 0.5f + 1e-6f;
            float u2 = (randf() + 1.0f) * 0.5f;
            float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
            /* Scale to typical key magnitude */
            keys[i * head_dim + d] = z * 0.15f;
        }
        /* Add a few outliers (typical in LLM attention) */
        if (i % 32 == 0) {
            int outlier_dim = (int)(randf() * 0.5f + 0.5f) * head_dim;
            if (outlier_dim >= head_dim) outlier_dim = head_dim - 1;
            if (outlier_dim < 0) outlier_dim = 0;
            keys[i * head_dim + outlier_dim] = randf() * 3.0f;
        }
    }
}

/* ============================================================
 * Memory calculation
 * ============================================================ */

static double kv_cache_memory_fp16(const model_spec_t* m, int context_len) {
    /* FP16: 2 bytes per element, K+V = 2x */
    return (double)m->num_layers * m->num_kv_heads * m->head_dim
         * context_len * 2 /* K+V */ * 2 /* fp16 bytes */ / (1024.0 * 1024.0 * 1024.0);
}

static double kv_cache_memory_quantized(const model_spec_t* m, int context_len, tq_type type) {
    float bpe = tq_type_bpe(type);
    /* bpe is for keys only; assume values at 4-bit */
    double key_bytes = (double)m->num_layers * m->num_kv_heads * m->head_dim
                     * context_len * bpe / 8.0;
    double val_bytes = (double)m->num_layers * m->num_kv_heads * m->head_dim
                     * context_len * 4.0 / 8.0; /* 4-bit values */
    return (key_bytes + val_bytes) / (1024.0 * 1024.0 * 1024.0);
}

/* ============================================================
 * Quality measurement
 * ============================================================ */

static double measure_attention_quality(tq_context_t* ctx, tq_type type,
                                        int head_dim, int seq_len) {
    float* keys = (float*)malloc(seq_len * head_dim * sizeof(float));
    float* query = (float*)malloc(head_dim * sizeof(float));
    float* fp32_scores = (float*)malloc(seq_len * sizeof(float));
    float* quant_scores = (float*)malloc(seq_len * sizeof(float));

    generate_realistic_keys(keys, seq_len, head_dim);
    for (int d = 0; d < head_dim; d++) query[d] = randf() * 0.2f;

    /* FP32 reference scores */
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < head_dim; d++)
            dot += query[d] * keys[s * head_dim + d];
        fp32_scores[s] = dot;
    }

    /* Quantized scores */
    size_t buf_size = tq_quantize_keys_size(seq_len, head_dim, type);
    void* quantized = malloc(buf_size);
    tq_quantize_keys(ctx, keys, seq_len, head_dim, type, quantized, buf_size);
    tq_attention(ctx, query, quantized, seq_len, head_dim, type, quant_scores);

    /* Cosine similarity */
    double dot_ab = 0, sq_a = 0, sq_b = 0;
    for (int s = 0; s < seq_len; s++) {
        dot_ab += (double)fp32_scores[s] * (double)quant_scores[s];
        sq_a += (double)fp32_scores[s] * (double)fp32_scores[s];
        sq_b += (double)quant_scores[s] * (double)quant_scores[s];
    }
    double cosine = (sq_a > 0 && sq_b > 0) ? dot_ab / (sqrt(sq_a) * sqrt(sq_b)) : 0;

    free(keys); free(query); free(fp32_scores); free(quant_scores); free(quantized);
    return cosine;
}

/* ============================================================
 * Throughput measurement
 * ============================================================ */

static double measure_throughput(tq_context_t* ctx, tq_type type,
                                 int head_dim, int n_keys) {
    float* keys = (float*)malloc(n_keys * head_dim * sizeof(float));
    generate_realistic_keys(keys, n_keys, head_dim);

    size_t buf_size = tq_quantize_keys_size(n_keys, head_dim, type);
    void* quantized = malloc(buf_size);

    clock_t start = clock();
    tq_quantize_keys(ctx, keys, n_keys, head_dim, type, quantized, buf_size);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double throughput = (elapsed > 0) ? (double)n_keys * head_dim / elapsed / 1e6 : 0;

    free(keys); free(quantized);
    return throughput; /* M elements/sec */
}

/* ============================================================
 * Main demo
 * ============================================================ */

int main(int argc, char** argv) {
    tq_context_t* ctx;
    tq_init(&ctx, TQ_BACKEND_CPU);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         quant.cpp — Real Model Demo v%s           ║\n", TQ_VERSION_STRING);
    printf("║         KV Cache Compression for LLM Inference              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    /* ---- Part 1: Memory Savings ---- */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  PART 1: Memory Savings (KV Cache)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    int context_lengths[] = {4096, 16384, 65536, 131072};
    int n_ctx = sizeof(context_lengths) / sizeof(context_lengths[0]);

    printf("  %-18s | %-8s", "Model", "Context");
    printf(" | %-10s | %-10s | %-10s | %-6s\n", "FP16", "Polar-4B", "Turbo-3B", "Saved");
    printf("  %-18s-+-%-8s", "------------------", "--------");
    printf("-+-%-10s-+-%-10s-+-%-10s-+-%-6s\n", "----------", "----------", "----------", "------");

    for (int mi = 0; mi < (int)NUM_MODELS; mi++) {
        const model_spec_t* m = &MODELS[mi];
        for (int ci = 0; ci < n_ctx; ci++) {
            int ctx_len = context_lengths[ci];
            double fp16_gb = kv_cache_memory_fp16(m, ctx_len);
            double polar_gb = kv_cache_memory_quantized(m, ctx_len, TQ_TYPE_POLAR_4B);
            double turbo_gb = kv_cache_memory_quantized(m, ctx_len, TQ_TYPE_TURBO_3B);
            double saved_pct = (1.0 - turbo_gb / fp16_gb) * 100.0;

            char ctx_str[16];
            if (ctx_len >= 1024) snprintf(ctx_str, sizeof(ctx_str), "%dK", ctx_len / 1024);
            else snprintf(ctx_str, sizeof(ctx_str), "%d", ctx_len);

            printf("  %-18s | %-8s | %7.2f GB | %7.2f GB | %7.2f GB | %4.0f%%\n",
                   (ci == 0) ? m->name : "",
                   ctx_str, fp16_gb, polar_gb, turbo_gb, saved_pct);
        }
        printf("  %-18s-+-%-8s-+-%-10s-+-%-10s-+-%-10s-+-%-6s\n",
               "", "", "", "", "", "");
    }

    /* ---- Part 2: What fits in your GPU? ---- */
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  PART 2: Maximum Context Length (after model weights)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    double gpu_sizes[] = {8.0, 16.0, 24.0}; /* GB */
    const char* gpu_names[] = {"8GB (M2 Air)", "16GB (RTX 4060)", "24GB (RTX 4090)"};

    printf("  %-18s | %-16s | %-12s | %-12s | %-6s\n",
           "Model", "GPU", "FP16 ctx", "Turbo-3B ctx", "Gain");
    printf("  %-18s-+-%-16s-+-%-12s-+-%-12s-+-%-6s\n",
           "------------------", "----------------", "------------", "------------", "------");

    for (int mi = 0; mi < (int)NUM_MODELS; mi++) {
        const model_spec_t* m = &MODELS[mi];
        double weight_gb = m->param_b * 2.0; /* FP16 weights approx */

        for (int gi = 0; gi < 3; gi++) {
            double avail = gpu_sizes[gi] - weight_gb;
            if (avail <= 0) {
                printf("  %-18s | %-16s | %10s | %10s | %-6s\n",
                       (gi == 0) ? m->name : "", gpu_names[gi],
                       "N/A", "N/A", "-");
                continue;
            }

            /* FP16 per-token KV size */
            double fp16_per_token = (double)m->num_layers * m->num_kv_heads * m->head_dim
                                  * 2 * 2 / (1024.0 * 1024.0 * 1024.0);
            int fp16_ctx = (int)(avail / fp16_per_token);

            /* Turbo-3B per-token KV size */
            double turbo_bpe = tq_type_bpe(TQ_TYPE_TURBO_3B);
            double turbo_per_token = (double)m->num_layers * m->num_kv_heads * m->head_dim
                                   * (turbo_bpe / 8.0 + 4.0 / 8.0) / (1024.0 * 1024.0 * 1024.0);
            int turbo_ctx = (int)(avail / turbo_per_token);

            char fp16_str[16], turbo_str[16];
            if (fp16_ctx >= 1000) snprintf(fp16_str, sizeof(fp16_str), "%dK", fp16_ctx / 1000);
            else snprintf(fp16_str, sizeof(fp16_str), "%d", fp16_ctx);
            if (turbo_ctx >= 1000) snprintf(turbo_str, sizeof(turbo_str), "%dK", turbo_ctx / 1000);
            else snprintf(turbo_str, sizeof(turbo_str), "%d", turbo_ctx);

            printf("  %-18s | %-16s | %10s | %10s | %.1fx\n",
                   (gi == 0) ? m->name : "", gpu_names[gi],
                   fp16_str, turbo_str,
                   (fp16_ctx > 0) ? (double)turbo_ctx / fp16_ctx : 0);
        }
        printf("  %-18s-+-%-16s-+-%-12s-+-%-12s-+-%-6s\n",
               "", "", "", "", "");
    }

    /* ---- Part 3: Quality on realistic data ---- */
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  PART 3: Attention Quality (cosine similarity vs FP32)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    printf("  Testing with realistic Gaussian+outlier key distributions...\n\n");

    tq_type test_types[] = {TQ_TYPE_POLAR_4B, TQ_TYPE_POLAR_3B,
                            TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
                            TQ_TYPE_QJL_1B, TQ_TYPE_TURBO_3B};
    int n_types = sizeof(test_types) / sizeof(test_types[0]);

    int test_dims[] = {64, 96, 128};
    int n_dims = 3;

    printf("  %-14s | %-5s", "Type", "BPE");
    for (int di = 0; di < n_dims; di++) printf(" | dim=%-3d", test_dims[di]);
    printf(" | Verdict\n");
    printf("  %-14s-+-%-5s", "--------------", "-----");
    for (int di = 0; di < n_dims; di++) printf("-+--------");
    printf("-+--------\n");

    for (int ti = 0; ti < n_types; ti++) {
        tq_type type = test_types[ti];
        printf("  %-14s | %4.1f", tq_type_name(type), tq_type_bpe(type));

        double min_cos = 1.0;
        for (int di = 0; di < n_dims; di++) {
            int dim = test_dims[di];
            double cos_sim = measure_attention_quality(ctx, type, dim, 256);
            printf(" | %6.4f", cos_sim);
            if (cos_sim < min_cos) min_cos = cos_sim;
        }

        const char* verdict;
        if (min_cos > 0.99) verdict = "Excellent";
        else if (min_cos > 0.95) verdict = "Good";
        else if (min_cos > 0.85) verdict = "OK";
        else verdict = "Low";
        printf(" | %s\n", verdict);
    }

    /* ---- Part 4: Throughput on this machine ---- */
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  PART 4: Quantization Throughput (this machine)\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    printf("  %-14s | %-5s | %-8s | %-15s | %-8s\n",
           "Type", "BPE", "Compress", "Throughput", "Time/1M");
    printf("  %-14s-+-%-5s-+-%-8s-+-%-15s-+-%-8s\n",
           "--------------", "-----", "--------", "---------------", "--------");

    for (int ti = 0; ti < n_types; ti++) {
        tq_type type = test_types[ti];
        double tput = measure_throughput(ctx, type, 128, 10000);
        double compress = 32.0 / tq_type_bpe(type);
        double time_1m = (tput > 0) ? 1.0 / tput : 999;

        printf("  %-14s | %4.1f | %5.1fx  | %8.1f Melem/s | %5.1f ms\n",
               tq_type_name(type), tq_type_bpe(type), compress, tput, time_1m * 1000);
    }

    /* ---- Summary ---- */
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  SUMMARY\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    printf("  Best quality:     polar_4b (4.5 bpe, ~7x compression)\n");
    printf("  Best compression: uniform_2b (2.25 bpe, ~14x compression)\n");
    printf("  Best balance:     turbo_3b (PolarQuant + QJL residual)\n");
    printf("\n");
    printf("  Example: Llama-3.2-3B on RTX 4090 (24GB)\n");
    double fp16_mem = kv_cache_memory_fp16(&MODELS[2], 65536);
    double turbo_mem = kv_cache_memory_quantized(&MODELS[2], 65536, TQ_TYPE_TURBO_3B);
    printf("    FP16 KV cache (64K ctx): %.2f GB\n", fp16_mem);
    printf("    Turbo-3B cache (64K ctx): %.2f GB\n", turbo_mem);
    printf("    Memory saved: %.2f GB (%.0f%%)\n",
           fp16_mem - turbo_mem, (1.0 - turbo_mem / fp16_mem) * 100);
    printf("\n");

    tq_free(ctx);
    return 0;
}
