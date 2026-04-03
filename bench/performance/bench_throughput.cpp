/**
 * quant.cpp -- Throughput Benchmark
 *
 * Measures tokens/second for different quantization types across batch sizes.
 * Simulates an inference-like workload: quantize keys, then run attention.
 *
 * Output (machine-readable):
 *   throughput_<type>_bs<batch>=XXXXX   (tokens/second)
 *
 * Build:
 *   cmake -B build -DTQ_BUILD_BENCH=ON
 *   cmake --build build --target bench_throughput
 *
 * Run:
 *   ./build/bench_throughput
 */

extern "C" {
#include "turboquant/turboquant.h"

/* Reference implementations */
void tq_polar_quantize_ref(const float* src, void* dst, int n);
void tq_polar_dequantize_ref(const void* src, float* dst, int n);
void tq_polar_attention_ref(const float* query, const void* kv,
                            float* scores, int seq_len, int head_dim);
void tq_qjl_quantize_ref(const float* src, void* dst, int n);
void tq_qjl_dequantize_ref(const void* src, float* dst, int n);
void tq_qjl_attention_ref(const float* query, const void* kv,
                           float* scores, int seq_len, int head_dim);
void tq_turbo_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_dequantize_ref(const void* src, float* dst, int n);
void tq_turbo_attention_ref(const float* query, const void* kv,
                             float* scores, int seq_len, int head_dim);
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);

#ifdef __ARM_NEON
void tq_uniform_4b_quantize_neon(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_neon(const void* src, float* dst, int n);
void tq_polar_quantize_neon(const float* src, void* dst, int n);
void tq_polar_dequantize_neon(const void* src, float* dst, int n);
void tq_qjl_quantize_neon(const float* src, void* dst, int n);
void tq_qjl_attention_neon(const float* query, const void* kv,
                            float* scores, int seq_len, int head_dim);
#endif
}

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

/* ============================================================
 * Timing
 * ============================================================ */

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

/* ============================================================
 * Simple LCG PRNG
 * ============================================================ */

static uint32_t rng_state = 42;
static float rand_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)(1 << 24)) * 4.0f - 2.0f;
}

/* ============================================================
 * Quantization type descriptors
 * ============================================================ */

struct QuantType {
    const char*      name;
    tq_type          type;
    tq_quantize_fn   quantize;
    tq_dequantize_fn dequantize;
    tq_attention_fn  attention;   /* NULL if not available */
    size_t           block_bytes;
    int              block_elems; /* elements per block */
};

static QuantType make_types() { return {}; }  /* placeholder */

static const QuantType ALL_TYPES[] = {
    {
        "polar_3b", TQ_TYPE_POLAR_3B,
        tq_polar_quantize_ref, tq_polar_dequantize_ref, tq_polar_attention_ref,
        sizeof(block_tq_polar), TQ_BK,
    },
    {
        "polar_4b", TQ_TYPE_POLAR_4B,
        tq_polar_quantize_ref, tq_polar_dequantize_ref, tq_polar_attention_ref,
        sizeof(block_tq_polar), TQ_BK,
    },
    {
        "qjl_1b", TQ_TYPE_QJL_1B,
        tq_qjl_quantize_ref, tq_qjl_dequantize_ref, tq_qjl_attention_ref,
        sizeof(block_tq_qjl), TQ_BK_QJL,
    },
    {
        "turbo_3b", TQ_TYPE_TURBO_3B,
        tq_turbo_quantize_ref, tq_turbo_dequantize_ref, tq_turbo_attention_ref,
        sizeof(block_tq_turbo), TQ_BK,
    },
    {
        "turbo_4b", TQ_TYPE_TURBO_4B,
        tq_turbo_quantize_ref, tq_turbo_dequantize_ref, tq_turbo_attention_ref,
        sizeof(block_tq_turbo), TQ_BK,
    },
    {
        "uniform_4b", TQ_TYPE_UNIFORM_4B,
        tq_uniform_4b_quantize_ref, tq_uniform_4b_dequantize_ref, nullptr,
        sizeof(block_tq_uniform_4b), TQ_BK,
    },
    {
        "uniform_2b", TQ_TYPE_UNIFORM_2B,
        tq_uniform_2b_quantize_ref, tq_uniform_2b_dequantize_ref, nullptr,
        sizeof(block_tq_uniform_2b), TQ_BK,
    },
};
static const int N_TYPES = sizeof(ALL_TYPES) / sizeof(ALL_TYPES[0]);

/* ============================================================
 * Throughput measurement
 *
 * Simulates a decode loop: for each "token", quantize one key vector
 * into the KV cache, then run attention over the full sequence.
 * Throughput = total tokens processed / wall-clock time.
 * ============================================================ */

static double measure_throughput(const QuantType& qt, int batch_size,
                                 int seq_len, int head_dim,
                                 const float* input_keys,
                                 const float* query,
                                 int n_warmup, int n_iters) {
    /* Allocate KV cache: seq_len blocks */
    size_t cache_bytes = qt.block_bytes * (size_t)seq_len;
    std::vector<uint8_t> kv_cache(cache_bytes, 0);

    /* Pre-fill the cache with quantized keys */
    for (int s = 0; s < seq_len; s++) {
        const float* src = input_keys + (s % batch_size) * head_dim;
        qt.quantize(src, kv_cache.data() + (size_t)s * qt.block_bytes, head_dim);
    }

    std::vector<float> scores(seq_len, 0.0f);

    /* The workload per "token": quantize 1 key + run attention.
     * With batch_size queries processed simultaneously. */

    auto run_batch = [&]() {
        for (int b = 0; b < batch_size; b++) {
            /* Quantize one new key into a slot */
            int slot = b % seq_len;
            qt.quantize(input_keys + b * head_dim,
                        kv_cache.data() + (size_t)slot * qt.block_bytes,
                        head_dim);

            /* Run attention (if available) or dequantize as proxy */
            if (qt.attention) {
                qt.attention(query + b * head_dim, kv_cache.data(),
                             scores.data(), seq_len, head_dim);
            } else {
                /* No attention kernel: dequantize last block as proxy work */
                float deq_buf[256];
                qt.dequantize(kv_cache.data() + (size_t)slot * qt.block_bytes,
                              deq_buf, head_dim);
                /* Compute dot product manually */
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += query[b * head_dim + d] * deq_buf[d];
                }
                scores[0] = dot; /* prevent optimization */
            }
        }
    };

    /* Warmup */
    for (int w = 0; w < n_warmup; w++) {
        run_batch();
    }

    /* Timed runs */
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        double t0 = now_sec();
        run_batch();
        double t1 = now_sec();
        times[i] = t1 - t0;
    }

    /* Use median time */
    std::sort(times.begin(), times.end());
    double median_sec = times[n_iters / 2];

    /* Tokens processed per call = batch_size */
    double tokens_per_sec = (double)batch_size / median_sec;
    return tokens_per_sec;
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    const int HEAD_DIM  = 128;
    const int SEQ_LEN   = 512;
    const int N_WARMUP  = 2;
    const int N_ITERS   = 5;

    static const int BATCH_SIZES[] = { 1, 4, 16, 64 };
    static const int N_BATCHES = sizeof(BATCH_SIZES) / sizeof(BATCH_SIZES[0]);

    int max_batch = 64;

    /* Generate input data: enough for the largest batch */
    std::vector<float> input_keys(max_batch * HEAD_DIM);
    std::vector<float> queries(max_batch * HEAD_DIM);
    for (size_t i = 0; i < input_keys.size(); i++) input_keys[i] = rand_float();
    for (size_t i = 0; i < queries.size(); i++) queries[i] = rand_float();

    printf("# quant.cpp Throughput Benchmark\n");
    printf("# HEAD_DIM=%d, SEQ_LEN=%d, N_ITERS=%d\n", HEAD_DIM, SEQ_LEN, N_ITERS);
#ifdef __ARM_NEON
    printf("# NEON: enabled\n");
#else
    printf("# NEON: disabled\n");
#endif
    printf("\n");

    /* --- Machine-readable output --- */
    for (int ti = 0; ti < N_TYPES; ti++) {
        QuantType qt = ALL_TYPES[ti];

        /* On NEON platforms, use accelerated kernels where available */
#ifdef __ARM_NEON
        if (qt.type == TQ_TYPE_UNIFORM_4B) {
            qt.quantize   = tq_uniform_4b_quantize_neon;
            qt.dequantize = tq_uniform_4b_dequantize_neon;
        }
        if (qt.type == TQ_TYPE_POLAR_3B || qt.type == TQ_TYPE_POLAR_4B) {
            qt.quantize   = tq_polar_quantize_neon;
            qt.dequantize = tq_polar_dequantize_neon;
        }
        if (qt.type == TQ_TYPE_QJL_1B) {
            qt.quantize = tq_qjl_quantize_neon;
            qt.attention = tq_qjl_attention_neon;
        }
#endif

        for (int bi = 0; bi < N_BATCHES; bi++) {
            int bs = BATCH_SIZES[bi];

            double tps = measure_throughput(
                qt, bs, SEQ_LEN, HEAD_DIM,
                input_keys.data(), queries.data(),
                N_WARMUP, N_ITERS
            );

            printf("throughput_%s_bs%d=%.0f\n", qt.name, bs, tps);
        }
    }

    /* --- Human-readable summary table --- */
    printf("\n# ============================================================\n");
    printf("# Throughput Summary (tokens/second)\n");
    printf("# ============================================================\n\n");

    /* Header */
    printf("%-14s", "Type");
    for (int bi = 0; bi < N_BATCHES; bi++) {
        char hdr[32];
        snprintf(hdr, sizeof(hdr), "BS=%d", BATCH_SIZES[bi]);
        printf("  %12s", hdr);
    }
    printf("\n");

    printf("%-14s", "--------------");
    for (int bi = 0; bi < N_BATCHES; bi++) {
        printf("  %12s", "------------");
    }
    printf("\n");

    /* Data rows */
    for (int ti = 0; ti < N_TYPES; ti++) {
        QuantType qt = ALL_TYPES[ti];

#ifdef __ARM_NEON
        if (qt.type == TQ_TYPE_UNIFORM_4B) {
            qt.quantize   = tq_uniform_4b_quantize_neon;
            qt.dequantize = tq_uniform_4b_dequantize_neon;
        }
        if (qt.type == TQ_TYPE_POLAR_3B || qt.type == TQ_TYPE_POLAR_4B) {
            qt.quantize   = tq_polar_quantize_neon;
            qt.dequantize = tq_polar_dequantize_neon;
        }
        if (qt.type == TQ_TYPE_QJL_1B) {
            qt.quantize = tq_qjl_quantize_neon;
            qt.attention = tq_qjl_attention_neon;
        }
#endif

        printf("%-14s", qt.name);

        for (int bi = 0; bi < N_BATCHES; bi++) {
            int bs = BATCH_SIZES[bi];

            double tps = measure_throughput(
                qt, bs, SEQ_LEN, HEAD_DIM,
                input_keys.data(), queries.data(),
                N_WARMUP, N_ITERS
            );

            printf("  %12.0f", tps);
        }
        printf("\n");
    }

    /* --- Compression ratio summary --- */
    printf("\n# Compression Ratios (vs FP32)\n");
    printf("%-14s  %12s  %12s\n", "Type", "Bytes/Vec", "Ratio");
    printf("%-14s  %12s  %12s\n", "--------------", "------------", "------------");

    double fp32_size = (double)HEAD_DIM * sizeof(float);
    printf("%-14s  %12.0f  %12.2f\n", "fp32", fp32_size, 1.0);

    for (int ti = 0; ti < N_TYPES; ti++) {
        const QuantType& qt = ALL_TYPES[ti];
        double ratio = fp32_size / (double)qt.block_bytes;
        printf("%-14s  %12zu  %12.2f\n", qt.name, qt.block_bytes, ratio);
    }

    return 0;
}
