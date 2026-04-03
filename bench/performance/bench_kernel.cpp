/**
 * quant.cpp -- Individual Kernel Performance Benchmark
 *
 * Times each operation (quantize, dequantize, attention) separately for
 * all 7 quantization types. Reports both machine-readable metrics and a
 * human-readable comparison table.
 *
 * Output (machine-readable):
 *   kernel_quantize_<type>=XXXXX       (elements/ms)
 *   kernel_dequantize_<type>=XXXXX     (elements/ms)
 *   kernel_attention_<type>=XXXXX      (queries/sec, seq=512)
 *
 * Build:
 *   cmake -B build -DTQ_BUILD_BENCH=ON
 *   cmake --build build --target bench_kernel
 *
 * Run:
 *   ./build/bench_kernel
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
 * Timing utilities
 * ============================================================ */

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

/**
 * Run fn() for n_warmup + n_iters times, return median elapsed seconds
 * for a single iteration.
 */
template <typename Fn>
static double bench_median_sec(Fn fn, int n_warmup = 3, int n_iters = 7) {
    for (int i = 0; i < n_warmup; i++) fn();

    std::vector<double> times(n_iters);
    for (int i = 0; i < n_iters; i++) {
        double t0 = now_sec();
        fn();
        double t1 = now_sec();
        times[i] = t1 - t0;
    }
    std::sort(times.begin(), times.end());
    return times[n_iters / 2];
}

/* ============================================================
 * PRNG
 * ============================================================ */

static uint32_t rng_state = 7777;
static float rand_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)(1 << 24)) * 4.0f - 2.0f;
}

/* ============================================================
 * Kernel descriptor
 * ============================================================ */

struct KernelType {
    const char*      name;
    const char*      backend;  /* "ref" or "neon" or "avx2" */
    tq_type          type;
    tq_quantize_fn   quantize;
    tq_dequantize_fn dequantize;
    tq_attention_fn  attention;   /* NULL if unavailable */
    size_t           block_bytes;
    int              block_elems;
};

/* ============================================================
 * Build the kernel list (ref + NEON where available)
 * ============================================================ */

static std::vector<KernelType> build_kernel_list() {
    std::vector<KernelType> kernels;

    /* Reference implementations for all 7 types */
    kernels.push_back({
        "polar_3b", "ref", TQ_TYPE_POLAR_3B,
        tq_polar_quantize_ref, tq_polar_dequantize_ref, tq_polar_attention_ref,
        sizeof(block_tq_polar), TQ_BK,
    });
    kernels.push_back({
        "polar_4b", "ref", TQ_TYPE_POLAR_4B,
        tq_polar_quantize_ref, tq_polar_dequantize_ref, tq_polar_attention_ref,
        sizeof(block_tq_polar), TQ_BK,
    });
    kernels.push_back({
        "qjl_1b", "ref", TQ_TYPE_QJL_1B,
        tq_qjl_quantize_ref, tq_qjl_dequantize_ref, tq_qjl_attention_ref,
        sizeof(block_tq_qjl), TQ_BK_QJL,
    });
    kernels.push_back({
        "turbo_3b", "ref", TQ_TYPE_TURBO_3B,
        tq_turbo_quantize_ref, tq_turbo_dequantize_ref, tq_turbo_attention_ref,
        sizeof(block_tq_turbo), TQ_BK,
    });
    kernels.push_back({
        "turbo_4b", "ref", TQ_TYPE_TURBO_4B,
        tq_turbo_quantize_ref, tq_turbo_dequantize_ref, tq_turbo_attention_ref,
        sizeof(block_tq_turbo), TQ_BK,
    });
    kernels.push_back({
        "uniform_4b", "ref", TQ_TYPE_UNIFORM_4B,
        tq_uniform_4b_quantize_ref, tq_uniform_4b_dequantize_ref, nullptr,
        sizeof(block_tq_uniform_4b), TQ_BK,
    });
    kernels.push_back({
        "uniform_2b", "ref", TQ_TYPE_UNIFORM_2B,
        tq_uniform_2b_quantize_ref, tq_uniform_2b_dequantize_ref, nullptr,
        sizeof(block_tq_uniform_2b), TQ_BK,
    });

#ifdef __ARM_NEON
    /* NEON-accelerated variants */
    kernels.push_back({
        "polar_3b", "neon", TQ_TYPE_POLAR_3B,
        tq_polar_quantize_neon, tq_polar_dequantize_neon, tq_polar_attention_ref,
        sizeof(block_tq_polar), TQ_BK,
    });
    kernels.push_back({
        "polar_4b", "neon", TQ_TYPE_POLAR_4B,
        tq_polar_quantize_neon, tq_polar_dequantize_neon, tq_polar_attention_ref,
        sizeof(block_tq_polar), TQ_BK,
    });
    kernels.push_back({
        "qjl_1b", "neon", TQ_TYPE_QJL_1B,
        tq_qjl_quantize_neon, tq_qjl_dequantize_ref, tq_qjl_attention_neon,
        sizeof(block_tq_qjl), TQ_BK_QJL,
    });
    kernels.push_back({
        "uniform_4b", "neon", TQ_TYPE_UNIFORM_4B,
        tq_uniform_4b_quantize_neon, tq_uniform_4b_dequantize_neon, nullptr,
        sizeof(block_tq_uniform_4b), TQ_BK,
    });
#endif

    return kernels;
}

/* ============================================================
 * Main benchmark
 * ============================================================ */

int main() {
    const int HEAD_DIM     = 128;
    const int N_VECTORS    = 10000;
    const int SEQ_LEN      = 512;

    printf("# quant.cpp Kernel Performance Benchmark\n");
    printf("# HEAD_DIM=%d, N_VECTORS=%d (quant/dequant), SEQ_LEN=%d (attention)\n",
           HEAD_DIM, N_VECTORS, SEQ_LEN);
#ifdef __ARM_NEON
    printf("# NEON: enabled\n");
#else
    printf("# NEON: disabled\n");
#endif
    printf("\n");

    /* Generate input data */
    std::vector<float> input_data(N_VECTORS * HEAD_DIM);
    for (size_t i = 0; i < input_data.size(); i++) input_data[i] = rand_float();

    std::vector<float> query(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; i++) query[i] = rand_float();

    auto kernels = build_kernel_list();

    /* Storage for results */
    struct KernelResult {
        const char* name;
        const char* backend;
        double quant_elem_per_ms;
        double dequant_elem_per_ms;
        double attn_queries_per_sec;
        bool   has_attention;
    };
    std::vector<KernelResult> results;

    for (const auto& kt : kernels) {
        KernelResult res;
        res.name    = kt.name;
        res.backend = kt.backend;
        res.has_attention = (kt.attention != nullptr);

        /* --- Quantize benchmark --- */
        {
            size_t buf_size = kt.block_bytes * N_VECTORS;
            std::vector<uint8_t> quant_buf(buf_size, 0);

            double dt = bench_median_sec([&]() {
                for (int v = 0; v < N_VECTORS; v++) {
                    kt.quantize(input_data.data() + v * HEAD_DIM,
                                quant_buf.data() + (size_t)v * kt.block_bytes,
                                HEAD_DIM);
                }
            });

            double total_elems = (double)N_VECTORS * HEAD_DIM;
            res.quant_elem_per_ms = total_elems / (dt * 1000.0);
        }

        /* --- Dequantize benchmark --- */
        {
            /* First, quantize data so we have valid blocks */
            size_t buf_size = kt.block_bytes * N_VECTORS;
            std::vector<uint8_t> quant_buf(buf_size, 0);
            for (int v = 0; v < N_VECTORS; v++) {
                kt.quantize(input_data.data() + v * HEAD_DIM,
                            quant_buf.data() + (size_t)v * kt.block_bytes,
                            HEAD_DIM);
            }

            std::vector<float> deq_buf(N_VECTORS * HEAD_DIM, 0.0f);

            double dt = bench_median_sec([&]() {
                for (int v = 0; v < N_VECTORS; v++) {
                    kt.dequantize(quant_buf.data() + (size_t)v * kt.block_bytes,
                                  deq_buf.data() + v * HEAD_DIM,
                                  HEAD_DIM);
                }
            });

            double total_elems = (double)N_VECTORS * HEAD_DIM;
            res.dequant_elem_per_ms = total_elems / (dt * 1000.0);
        }

        /* --- Attention benchmark --- */
        if (kt.attention != nullptr) {
            /* Build KV cache of SEQ_LEN blocks */
            size_t cache_size = kt.block_bytes * SEQ_LEN;
            std::vector<uint8_t> kv_cache(cache_size, 0);
            for (int s = 0; s < SEQ_LEN; s++) {
                int idx = s % N_VECTORS;
                kt.quantize(input_data.data() + idx * HEAD_DIM,
                            kv_cache.data() + (size_t)s * kt.block_bytes,
                            HEAD_DIM);
            }

            std::vector<float> scores(SEQ_LEN, 0.0f);

            double dt = bench_median_sec([&]() {
                kt.attention(query.data(), kv_cache.data(),
                             scores.data(), SEQ_LEN, HEAD_DIM);
            });

            res.attn_queries_per_sec = 1.0 / dt;
        } else {
            res.attn_queries_per_sec = 0.0;
        }

        results.push_back(res);
    }

    /* --- Machine-readable output --- */
    for (const auto& r : results) {
        const char* suffix = (strcmp(r.backend, "ref") == 0) ? "" : r.backend;
        if (strlen(suffix) > 0) {
            printf("kernel_quantize_%s_%s=%.0f\n", r.name, suffix, r.quant_elem_per_ms);
            printf("kernel_dequantize_%s_%s=%.0f\n", r.name, suffix, r.dequant_elem_per_ms);
            if (r.has_attention) {
                printf("kernel_attention_%s_%s=%.0f\n", r.name, suffix, r.attn_queries_per_sec);
            }
        } else {
            printf("kernel_quantize_%s=%.0f\n", r.name, r.quant_elem_per_ms);
            printf("kernel_dequantize_%s=%.0f\n", r.name, r.dequant_elem_per_ms);
            if (r.has_attention) {
                printf("kernel_attention_%s=%.0f\n", r.name, r.attn_queries_per_sec);
            }
        }
    }

    /* --- Human-readable comparison table --- */
    printf("\n# ============================================================\n");
    printf("# Kernel Performance Comparison\n");
    printf("# ============================================================\n\n");

    printf("%-14s  %-6s  %14s  %14s  %14s\n",
           "Type", "Backend", "Quant (elem/ms)", "Deq (elem/ms)", "Attn (q/sec)");
    printf("%-14s  %-6s  %14s  %14s  %14s\n",
           "--------------", "------", "--------------", "--------------", "--------------");

    for (const auto& r : results) {
        if (r.has_attention) {
            printf("%-14s  %-6s  %14.0f  %14.0f  %14.0f\n",
                   r.name, r.backend,
                   r.quant_elem_per_ms, r.dequant_elem_per_ms,
                   r.attn_queries_per_sec);
        } else {
            printf("%-14s  %-6s  %14.0f  %14.0f  %14s\n",
                   r.name, r.backend,
                   r.quant_elem_per_ms, r.dequant_elem_per_ms,
                   "N/A");
        }
    }

    /* --- SIMD speedup summary (if NEON entries exist) --- */
#ifdef __ARM_NEON
    printf("\n# SIMD Speedup (NEON / Reference)\n");
    printf("%-14s  %14s  %14s  %14s\n",
           "Type", "Quant Speedup", "Deq Speedup", "Attn Speedup");
    printf("%-14s  %14s  %14s  %14s\n",
           "--------------", "--------------", "--------------", "--------------");

    for (const auto& neon_r : results) {
        if (strcmp(neon_r.backend, "neon") != 0) continue;

        /* Find matching ref entry */
        const KernelResult* ref_r = nullptr;
        for (const auto& rr : results) {
            if (strcmp(rr.backend, "ref") == 0 &&
                strcmp(rr.name, neon_r.name) == 0) {
                ref_r = &rr;
                break;
            }
        }

        if (!ref_r) continue;

        double q_speedup = (ref_r->quant_elem_per_ms > 0)
            ? neon_r.quant_elem_per_ms / ref_r->quant_elem_per_ms : 0.0;
        double d_speedup = (ref_r->dequant_elem_per_ms > 0)
            ? neon_r.dequant_elem_per_ms / ref_r->dequant_elem_per_ms : 0.0;

        if (neon_r.has_attention && ref_r->has_attention &&
            ref_r->attn_queries_per_sec > 0) {
            double a_speedup = neon_r.attn_queries_per_sec / ref_r->attn_queries_per_sec;
            printf("%-14s  %13.2fx  %13.2fx  %13.2fx\n",
                   neon_r.name, q_speedup, d_speedup, a_speedup);
        } else {
            printf("%-14s  %13.2fx  %13.2fx  %14s\n",
                   neon_r.name, q_speedup, d_speedup, "N/A");
        }
    }
#endif

    return 0;
}
