/**
 * quant.cpp -- Latency Benchmark
 *
 * Measures per-call latency for quantization and attention operations
 * across all quant.cpp quantization types.
 *
 * Metrics output (machine-readable):
 *   quant_latency_<type>=XXX.XX      (microseconds per vector)
 *   dequant_latency_<type>=XXX.XX    (microseconds per vector)
 *   attention_latency_<type>=XXX.XX  (microseconds per query)
 *   quant_throughput_<type>=XXXXX    (vectors/sec)
 *   attention_throughput_<type>=XXXXX (queries/sec)
 *
 * Build:
 *   cmake -B build -DTQ_BUILD_BENCH=ON
 *   cmake --build build --target bench_latency
 *
 * Run:
 *   ./build/bench_latency
 */

extern "C" {
#include "turboquant/turboquant.h"
void tq_polar_quantize_ref(const float* src, void* dst, int n);
void tq_polar_dequantize_ref(const void* src, float* dst, int n);
void tq_polar_attention_ref(const float* query, const void* kv,
                            float* scores, int seq_len, int head_dim);
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
void tq_qjl_quantize_ref(const float* src, void* dst, int n);
void tq_qjl_dequantize_ref(const void* src, float* dst, int n);
void tq_qjl_attention_ref(const float* query, const void* kv,
                           float* scores, int seq_len, int head_dim);
void tq_turbo_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_dequantize_ref(const void* src, float* dst, int n);
void tq_turbo_attention_ref(const float* query, const void* kv,
                             float* scores, int seq_len, int head_dim);
}

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>

/* ============================================================
 * Timing utilities
 * ============================================================ */

static double now_us() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t.time_since_epoch()).count();
}

/**
 * Run a function N_WARMUP + N_ITERS times, return the median iteration
 * time in microseconds.
 */
template <typename Fn>
static double benchmark_us(Fn fn, int n_warmup = 3, int n_iters = 10) {
    /* Warmup */
    for (int i = 0; i < n_warmup; i++) {
        fn();
    }

    /* Timed iterations */
    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        double t0 = now_us();
        fn();
        double t1 = now_us();
        times[i] = t1 - t0;
    }

    /* Return median */
    std::sort(times.begin(), times.end());
    return times[n_iters / 2];
}

/* ============================================================
 * Simple LCG PRNG for reproducible test data
 * ============================================================ */

static uint32_t rng_state = 12345;
static float rand_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

/* ============================================================
 * Benchmark definitions
 * ============================================================ */

struct BenchType {
    const char*      name;
    tq_type          type;
    tq_quantize_fn   quantize;
    tq_dequantize_fn dequantize;
    tq_attention_fn  attention;
    size_t           block_bytes;
};

static const BenchType TYPES[] = {
    {
        "polar_3b", TQ_TYPE_POLAR_3B,
        tq_polar_quantize_ref, tq_polar_dequantize_ref, tq_polar_attention_ref,
        sizeof(block_tq_polar),
    },
    {
        "polar_4b", TQ_TYPE_POLAR_4B,
        tq_polar_quantize_ref, tq_polar_dequantize_ref, tq_polar_attention_ref,
        sizeof(block_tq_polar),
    },
    {
        "qjl_1b", TQ_TYPE_QJL_1B,
        tq_qjl_quantize_ref, tq_qjl_dequantize_ref, tq_qjl_attention_ref,
        sizeof(block_tq_qjl),
    },
    {
        "turbo_3b", TQ_TYPE_TURBO_3B,
        tq_turbo_quantize_ref, tq_turbo_dequantize_ref, tq_turbo_attention_ref,
        sizeof(block_tq_turbo),
    },
    {
        "turbo_4b", TQ_TYPE_TURBO_4B,
        tq_turbo_quantize_ref, tq_turbo_dequantize_ref, tq_turbo_attention_ref,
        sizeof(block_tq_turbo),
    },
    {
        "uniform_4b", TQ_TYPE_UNIFORM_4B,
        tq_uniform_4b_quantize_ref, tq_uniform_4b_dequantize_ref, nullptr,
        sizeof(block_tq_uniform_4b),
    },
    {
        "uniform_2b", TQ_TYPE_UNIFORM_2B,
        tq_uniform_2b_quantize_ref, tq_uniform_2b_dequantize_ref, nullptr,
        sizeof(block_tq_uniform_2b),
    },
};
static const int N_TYPES = sizeof(TYPES) / sizeof(TYPES[0]);

int main() {
    const int HEAD_DIM    = 128;
    const int N_VECTORS   = 1000;
    const int SEQ_LEN     = 1024;

    printf("# quant.cpp Latency Benchmark\n");
    printf("# HEAD_DIM=%d, N_VECTORS=%d, SEQ_LEN=%d\n\n", HEAD_DIM, N_VECTORS, SEQ_LEN);

    /* Generate random input data */
    std::vector<float> input_data(N_VECTORS * HEAD_DIM);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = rand_float();
    }

    std::vector<float> query(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; i++) query[i] = rand_float();

    /* --- Machine-readable metrics --- */
    for (int ti = 0; ti < N_TYPES; ti++) {
        const BenchType& bt = TYPES[ti];

        /* Allocate output buffers */
        size_t quant_buf_size = bt.block_bytes * N_VECTORS;
        std::vector<uint8_t> quant_buf(quant_buf_size, 0);
        std::vector<float> deq_buf(N_VECTORS * HEAD_DIM, 0.0f);

        /* --- Quantize benchmark --- */
        double quant_time = benchmark_us([&]() {
            uint8_t* dst = quant_buf.data();
            for (int v = 0; v < N_VECTORS; v++) {
                bt.quantize(input_data.data() + v * HEAD_DIM, dst, HEAD_DIM);
                dst += bt.block_bytes;
            }
        });
        double quant_per_vec = quant_time / N_VECTORS;
        double quant_throughput = 1e6 / quant_per_vec;  /* vectors/sec */

        /* --- Dequantize benchmark --- */
        double dequant_time = benchmark_us([&]() {
            const uint8_t* src = quant_buf.data();
            for (int v = 0; v < N_VECTORS; v++) {
                bt.dequantize(src, deq_buf.data() + v * HEAD_DIM, HEAD_DIM);
                src += bt.block_bytes;
            }
        });
        double dequant_per_vec = dequant_time / N_VECTORS;

        printf("quant_latency_%s=%.2f\n", bt.name, quant_per_vec);
        printf("dequant_latency_%s=%.2f\n", bt.name, dequant_per_vec);
        printf("quant_throughput_%s=%.0f\n", bt.name, quant_throughput);

        /* --- Attention benchmark (if available) --- */
        if (bt.attention != nullptr) {
            /* Prepare quantized KV cache for SEQ_LEN vectors */
            size_t attn_buf_size = bt.block_bytes * SEQ_LEN;
            std::vector<uint8_t> kv_cache(attn_buf_size, 0);
            uint8_t* dst = kv_cache.data();
            for (int s = 0; s < SEQ_LEN; s++) {
                int idx = s % N_VECTORS;
                bt.quantize(input_data.data() + idx * HEAD_DIM, dst, HEAD_DIM);
                dst += bt.block_bytes;
            }

            std::vector<float> scores(SEQ_LEN, 0.0f);

            double attn_time = benchmark_us([&]() {
                bt.attention(query.data(), kv_cache.data(),
                             scores.data(), SEQ_LEN, HEAD_DIM);
            });

            double attn_throughput = 1e6 / attn_time;  /* queries/sec */

            printf("attention_latency_%s=%.2f\n", bt.name, attn_time);
            printf("attention_throughput_%s=%.0f\n", bt.name, attn_throughput);
        }
    }

    /* --- Human-readable summary table --- */
    printf("\n# ============================================================\n");
    printf("# Latency Summary (microseconds)\n");
    printf("# ============================================================\n\n");

    printf("%-14s  %12s  %12s  %12s  %14s\n",
           "Type", "Quant (us)", "Dequant (us)", "Attn (us)", "Quant (vec/s)");
    printf("%-14s  %12s  %12s  %12s  %14s\n",
           "--------------", "------------", "------------", "------------", "--------------");

    for (int ti = 0; ti < N_TYPES; ti++) {
        const BenchType& bt = TYPES[ti];

        /* Re-run measurements for the table */
        std::vector<uint8_t> quant_buf(bt.block_bytes * N_VECTORS, 0);
        std::vector<float> deq_buf(N_VECTORS * HEAD_DIM, 0.0f);

        double quant_time = benchmark_us([&]() {
            uint8_t* dst = quant_buf.data();
            for (int v = 0; v < N_VECTORS; v++) {
                bt.quantize(input_data.data() + v * HEAD_DIM, dst, HEAD_DIM);
                dst += bt.block_bytes;
            }
        });
        double quant_per_vec = quant_time / N_VECTORS;

        double dequant_time = benchmark_us([&]() {
            const uint8_t* src = quant_buf.data();
            for (int v = 0; v < N_VECTORS; v++) {
                bt.dequantize(src, deq_buf.data() + v * HEAD_DIM, HEAD_DIM);
                src += bt.block_bytes;
            }
        });
        double dequant_per_vec = dequant_time / N_VECTORS;

        double quant_throughput = 1e6 / quant_per_vec;

        if (bt.attention != nullptr) {
            size_t attn_buf_size = bt.block_bytes * SEQ_LEN;
            std::vector<uint8_t> kv_cache(attn_buf_size, 0);
            uint8_t* dst = kv_cache.data();
            for (int s = 0; s < SEQ_LEN; s++) {
                bt.quantize(input_data.data() + (s % N_VECTORS) * HEAD_DIM,
                            dst, HEAD_DIM);
                dst += bt.block_bytes;
            }
            std::vector<float> scores(SEQ_LEN, 0.0f);

            double attn_time = benchmark_us([&]() {
                bt.attention(query.data(), kv_cache.data(),
                             scores.data(), SEQ_LEN, HEAD_DIM);
            });

            printf("%-14s  %12.2f  %12.2f  %12.2f  %14.0f\n",
                   bt.name, quant_per_vec, dequant_per_vec, attn_time,
                   quant_throughput);
        } else {
            printf("%-14s  %12.2f  %12.2f  %12s  %14.0f\n",
                   bt.name, quant_per_vec, dequant_per_vec, "N/A",
                   quant_throughput);
        }
    }

    /* --- Seq-length scaling --- */
    printf("\n# Attention Latency vs Sequence Length (us)\n");
    printf("%-14s", "Type");
    int scale_lens[] = { 64, 256, 1024, 4096 };
    int n_scale = sizeof(scale_lens) / sizeof(scale_lens[0]);
    for (int si = 0; si < n_scale; si++) {
        char label[32];
        snprintf(label, sizeof(label), "seq=%d", scale_lens[si]);
        printf("  %12s", label);
    }
    printf("\n");

    printf("%-14s", "--------------");
    for (int si = 0; si < n_scale; si++) printf("  %12s", "------------");
    printf("\n");

    for (int ti = 0; ti < N_TYPES; ti++) {
        const BenchType& bt = TYPES[ti];
        if (bt.attention == nullptr) continue;

        printf("%-14s", bt.name);
        for (int si = 0; si < n_scale; si++) {
            int slen = scale_lens[si];

            /* Build KV cache */
            std::vector<uint8_t> kv_cache(bt.block_bytes * slen, 0);
            uint8_t* dst = kv_cache.data();
            for (int s = 0; s < slen; s++) {
                int idx = s % N_VECTORS;
                bt.quantize(input_data.data() + idx * HEAD_DIM, dst, HEAD_DIM);
                dst += bt.block_bytes;
            }
            std::vector<float> scores(slen, 0.0f);

            double attn_time = benchmark_us([&]() {
                bt.attention(query.data(), kv_cache.data(),
                             scores.data(), slen, HEAD_DIM);
            });

            printf("  %12.2f", attn_time);
        }
        printf("\n");
    }

    return 0;
}
