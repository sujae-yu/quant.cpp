/**
 * quant.cpp -- Memory Usage Benchmark
 *
 * Measures KV cache memory consumption for FP32, FP16, and each quant.cpp
 * quantization type across various sequence lengths (1K, 4K, 16K, 64K).
 *
 * Output format (machine-readable):
 *   memory_<type>_<seqlen>=XXXX   (bytes)
 *   ratio_<type>_<seqlen>=X.XX    (compression ratio vs FP16)
 *
 * Build:
 *   cmake -B build -DTQ_BUILD_BENCH=ON
 *   cmake --build build --target bench_memory
 *
 * Run:
 *   ./build/bench_memory
 */

extern "C" {
#include "turboquant/turboquant.h"
}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

/* Configuration */
static const int HEAD_DIM    = 128;
static const int NUM_HEADS   = 32;
static const int SEQ_LENS[]  = { 1024, 4096, 16384, 65536 };
static const int N_SEQ_LENS  = sizeof(SEQ_LENS) / sizeof(SEQ_LENS[0]);

/* All quantization types to benchmark */
static const struct {
    tq_type     type;
    const char* label;
    int         value_bits;  /* 0 = skip value quant, use FP16 for values */
} BENCH_TYPES[] = {
    { TQ_TYPE_POLAR_3B,   "polar_3b",   4 },
    { TQ_TYPE_POLAR_4B,   "polar_4b",   4 },
    { TQ_TYPE_QJL_1B,     "qjl_1b",     2 },
    { TQ_TYPE_TURBO_3B,   "turbo_3b",   4 },
    { TQ_TYPE_TURBO_4B,   "turbo_4b",   4 },
    { TQ_TYPE_UNIFORM_4B, "uniform_4b", 4 },
    { TQ_TYPE_UNIFORM_2B, "uniform_2b", 2 },
};
static const int N_TYPES = sizeof(BENCH_TYPES) / sizeof(BENCH_TYPES[0]);

/**
 * Compute KV cache memory for FP32 baseline.
 * key + value = 2 * seq_len * num_heads * head_dim * sizeof(float)
 */
static size_t fp32_memory(int seq_len) {
    return 2ULL * (size_t)seq_len * NUM_HEADS * HEAD_DIM * sizeof(float);
}

/**
 * Compute KV cache memory for FP16 baseline.
 * key + value = 2 * seq_len * num_heads * head_dim * 2 bytes
 */
static size_t fp16_memory(int seq_len) {
    return 2ULL * (size_t)seq_len * NUM_HEADS * HEAD_DIM * 2;
}

/**
 * Compute KV cache memory for a quant.cpp type.
 * Keys: tq_quantize_keys_size(seq_len * num_heads, head_dim, type)
 * Values: tq_quantize_values_size(seq_len * num_heads, head_dim, bits)
 *         or FP16 if value_bits == 0
 */
static size_t tq_memory(int seq_len, tq_type key_type, int value_bits) {
    int total_vectors = seq_len * NUM_HEADS;
    size_t key_bytes = tq_quantize_keys_size(total_vectors, HEAD_DIM, key_type);

    size_t val_bytes;
    if (value_bits > 0) {
        val_bytes = tq_quantize_values_size(total_vectors, HEAD_DIM, value_bits);
    } else {
        val_bytes = (size_t)total_vectors * HEAD_DIM * 2;  /* FP16 */
    }

    return key_bytes + val_bytes;
}

static const char* format_bytes(size_t bytes) {
    static char buf[64];
    if (bytes >= 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= 1024ULL * 1024) {
        snprintf(buf, sizeof(buf), "%.2f MB", (double)bytes / (1024.0 * 1024.0));
    } else if (bytes >= 1024) {
        snprintf(buf, sizeof(buf), "%.2f KB", (double)bytes / 1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return buf;
}

int main() {
    printf("# quant.cpp Memory Benchmark\n");
    printf("# HEAD_DIM=%d, NUM_HEADS=%d\n", HEAD_DIM, NUM_HEADS);
    printf("# Model config: similar to Llama-3-8B (per layer)\n\n");

    /* --- Machine-readable output --- */
    for (int si = 0; si < N_SEQ_LENS; si++) {
        int seq_len = SEQ_LENS[si];
        size_t fp16_mem = fp16_memory(seq_len);

        printf("memory_fp32_%d=%zu\n", seq_len, fp32_memory(seq_len));
        printf("memory_fp16_%d=%zu\n", seq_len, fp16_mem);

        for (int ti = 0; ti < N_TYPES; ti++) {
            size_t tq_mem = tq_memory(seq_len, BENCH_TYPES[ti].type,
                                       BENCH_TYPES[ti].value_bits);
            double ratio = (double)fp16_mem / (double)tq_mem;

            printf("memory_%s_%d=%zu\n", BENCH_TYPES[ti].label, seq_len, tq_mem);
            printf("ratio_%s_%d=%.2f\n", BENCH_TYPES[ti].label, seq_len, ratio);
        }
    }

    /* --- Human-readable table --- */
    printf("\n# ============================================================\n");
    printf("# KV Cache Memory per Layer (num_heads=%d, head_dim=%d)\n", NUM_HEADS, HEAD_DIM);
    printf("# ============================================================\n\n");

    /* Header */
    printf("%-14s", "Type");
    for (int si = 0; si < N_SEQ_LENS; si++) {
        char label[32];
        if (SEQ_LENS[si] >= 1024) {
            snprintf(label, sizeof(label), "%dK", SEQ_LENS[si] / 1024);
        } else {
            snprintf(label, sizeof(label), "%d", SEQ_LENS[si]);
        }
        printf("  %12s", label);
    }
    printf("  %8s\n", "Ratio*");

    /* Separator */
    printf("%-14s", "--------------");
    for (int si = 0; si < N_SEQ_LENS; si++) printf("  %12s", "------------");
    printf("  %8s\n", "--------");

    /* FP32 row */
    printf("%-14s", "FP32");
    for (int si = 0; si < N_SEQ_LENS; si++) {
        printf("  %12s", format_bytes(fp32_memory(SEQ_LENS[si])));
    }
    printf("  %8s\n", "0.50x");

    /* FP16 row */
    printf("%-14s", "FP16");
    for (int si = 0; si < N_SEQ_LENS; si++) {
        printf("  %12s", format_bytes(fp16_memory(SEQ_LENS[si])));
    }
    printf("  %8s\n", "1.00x");

    /* TQ type rows */
    for (int ti = 0; ti < N_TYPES; ti++) {
        printf("%-14s", BENCH_TYPES[ti].label);
        double avg_ratio = 0.0;
        for (int si = 0; si < N_SEQ_LENS; si++) {
            size_t tq_mem = tq_memory(SEQ_LENS[si], BENCH_TYPES[ti].type,
                                       BENCH_TYPES[ti].value_bits);
            double ratio = (double)fp16_memory(SEQ_LENS[si]) / (double)tq_mem;
            avg_ratio += ratio;
            printf("  %12s", format_bytes(tq_mem));
        }
        avg_ratio /= N_SEQ_LENS;
        printf("  %7.2fx\n", avg_ratio);
    }

    printf("\n* Ratio = FP16 memory / TQ memory (higher = better compression)\n");

    /* --- Per-element bits summary --- */
    printf("\n# Bits per element (key + value combined)\n");
    printf("%-14s  %8s  %8s  %12s\n", "Type", "Key BPE", "Val BPE", "Total BPE");
    printf("%-14s  %8s  %8s  %12s\n", "--------------", "--------", "--------", "------------");

    printf("%-14s  %8.2f  %8.2f  %12.2f\n", "FP32", 32.0, 32.0, 64.0);
    printf("%-14s  %8.2f  %8.2f  %12.2f\n", "FP16", 16.0, 16.0, 32.0);

    for (int ti = 0; ti < N_TYPES; ti++) {
        float key_bpe = tq_type_bpe(BENCH_TYPES[ti].type);
        float val_bpe;
        if (BENCH_TYPES[ti].value_bits == 4) {
            val_bpe = tq_type_bpe(TQ_TYPE_UNIFORM_4B);
        } else if (BENCH_TYPES[ti].value_bits == 2) {
            val_bpe = tq_type_bpe(TQ_TYPE_UNIFORM_2B);
        } else {
            val_bpe = 16.0f;  /* FP16 */
        }
        printf("%-14s  %8.2f  %8.2f  %12.2f\n",
               BENCH_TYPES[ti].label, key_bpe, val_bpe, key_bpe + val_bpe);
    }

    return 0;
}
