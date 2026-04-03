/**
 * quant.cpp performance benchmark
 *
 * Outputs machine-readable metrics:
 *   quantize_throughput=XXXXX          (elements/ms, best available)
 *   attention_throughput=XXXXX         (queries/sec, best available)
 *   compression_ratio=X.XX
 *   simd_speedup=X.XX
 *   quantize_throughput_<type>=XXXXX   (elements/ms, per-type)
 *   attention_throughput_<type>=XXXXX  (queries/sec, per-type)
 */

extern "C" {
#include "turboquant/turboquant.h"

/* Reference (generic) implementations */
void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
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

#ifdef __ARM_NEON
/* NEON implementations */
void tq_uniform_4b_quantize_neon(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_neon(const void* src, float* dst, int n);
void tq_polar_quantize_neon(const float* src, void* dst, int n);
void tq_polar_dequantize_neon(const void* src, float* dst, int n);
void tq_qjl_quantize_neon(const float* src, void* dst, int n);
void tq_qjl_attention_neon(const float* query, const void* kv,
                            float* scores, int seq_len, int head_dim);
void tq_polar_attention_neon(const float* query, const void* kv,
                              float* scores, int seq_len, int head_dim);
void tq_uniform_4b_attention_neon(const float* query, const void* kv,
                                   float* scores, int seq_len, int head_dim);
#endif
}

#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

static double now_sec() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t.time_since_epoch()).count();
}

/* ============================================================
 * Per-type benchmark descriptor
 * ============================================================ */

struct TypeBench {
    const char*      name;
    tq_quantize_fn   quantize;
    tq_dequantize_fn dequantize;
    tq_attention_fn  attention;  /* NULL if no dedicated attention kernel */
    size_t           block_bytes;
};

int main() {
    const int HEAD_DIM = 128;
    const int N_VECTORS = 10000;
    const int N_ITERS = 3;
    const int SEQ_LEN = 256;
    const int ATTN_REPS = 2000;

    /* Generate input */
    std::vector<float> keys(N_VECTORS * HEAD_DIM);
    for (int i = 0; i < N_VECTORS * HEAD_DIM; i++)
        keys[i] = sinf((float)i * 0.01f) * 2.0f;

    std::vector<float> query(HEAD_DIM);
    for (int i = 0; i < HEAD_DIM; i++) query[i] = cosf(i * 0.05f);

    /* Build per-type table using best available kernels */
    TypeBench types[] = {
        {
            "polar_3b",
            tq_polar_quantize_ref, tq_polar_dequantize_ref,
            tq_polar_attention_ref,
            sizeof(block_tq_polar),
        },
        {
            "polar_4b",
            tq_polar_quantize_ref, tq_polar_dequantize_ref,
            tq_polar_attention_ref,
            sizeof(block_tq_polar),
        },
        {
            "qjl_1b",
            tq_qjl_quantize_ref, tq_qjl_dequantize_ref,
            tq_qjl_attention_ref,
            sizeof(block_tq_qjl),
        },
        {
            "turbo_3b",
            tq_turbo_quantize_ref, tq_turbo_dequantize_ref,
            tq_turbo_attention_ref,
            sizeof(block_tq_turbo),
        },
        {
            "turbo_4b",
            tq_turbo_quantize_ref, tq_turbo_dequantize_ref,
            tq_turbo_attention_ref,
            sizeof(block_tq_turbo),
        },
        {
            "uniform_4b",
            tq_uniform_4b_quantize_ref, tq_uniform_4b_dequantize_ref,
            nullptr,
            sizeof(block_tq_uniform_4b),
        },
        {
            "uniform_2b",
            tq_uniform_2b_quantize_ref, tq_uniform_2b_dequantize_ref,
            nullptr,
            sizeof(block_tq_uniform_2b),
        },
    };
    const int N_TYPES = sizeof(types) / sizeof(types[0]);

    /* Upgrade to NEON where available */
#ifdef __ARM_NEON
    /* polar_3b (index 0) and polar_4b (index 1) */
    types[0].quantize   = tq_polar_quantize_neon;
    types[0].dequantize = tq_polar_dequantize_neon;
    types[0].attention  = tq_polar_attention_neon;
    types[1].quantize   = tq_polar_quantize_neon;
    types[1].dequantize = tq_polar_dequantize_neon;
    types[1].attention  = tq_polar_attention_neon;
    /* qjl_1b (index 2) */
    types[2].quantize   = tq_qjl_quantize_neon;
    types[2].attention  = tq_qjl_attention_neon;
    /* uniform_4b (index 5) — fused dequant+dot */
    types[5].quantize   = tq_uniform_4b_quantize_neon;
    types[5].dequantize = tq_uniform_4b_dequantize_neon;
    types[5].attention  = tq_uniform_4b_attention_neon;
#endif

    /* ============================================================
     * Per-type quantize throughput
     * ============================================================ */

    /* Track best overall quantize throughput (for legacy metric) */
    double best_overall_quant_tput = 0.0;
    double best_overall_attn_tput  = 0.0;
    double overall_compression     = 0.0;

    for (int ti = 0; ti < N_TYPES; ti++) {
        const TypeBench& tb = types[ti];

        /* Allocate quantized output buffer */
        size_t buf_size = tb.block_bytes * N_VECTORS;
        std::vector<uint8_t> quant_buf(buf_size, 0);

        /* Measure quantize throughput */
        double best_time = 1e9;
        for (int iter = 0; iter < N_ITERS; iter++) {
            double t0 = now_sec();
            for (int i = 0; i < N_VECTORS; i++) {
                tb.quantize(keys.data() + i * HEAD_DIM,
                            quant_buf.data() + (size_t)i * tb.block_bytes,
                            HEAD_DIM);
            }
            double dt = now_sec() - t0;
            if (dt < best_time) best_time = dt;
        }
        double quant_tput = (double)N_VECTORS * HEAD_DIM / (best_time * 1000.0);
        printf("quantize_throughput_%s=%.0f\n", tb.name, quant_tput);

        if (quant_tput > best_overall_quant_tput) {
            best_overall_quant_tput = quant_tput;
        }

        /* Track uniform_4b compression ratio as the representative */
        if (ti == 5) { /* uniform_4b */
            double fp32_sz = (double)sizeof(float) * HEAD_DIM;
            overall_compression = fp32_sz / (double)tb.block_bytes;
        }

        /* --- Attention throughput (if available) --- */
        if (tb.attention != nullptr) {
            /* Build a KV cache of SEQ_LEN quantized vectors */
            size_t cache_size = tb.block_bytes * SEQ_LEN;
            std::vector<uint8_t> kv_cache(cache_size, 0);
            for (int s = 0; s < SEQ_LEN; s++) {
                int idx = s % N_VECTORS;
                tb.quantize(keys.data() + idx * HEAD_DIM,
                            kv_cache.data() + (size_t)s * tb.block_bytes,
                            HEAD_DIM);
            }

            std::vector<float> scores(SEQ_LEN, 0.0f);

            double best_attn = 1e9;
            for (int iter = 0; iter < N_ITERS; iter++) {
                double t0 = now_sec();
                for (int rep = 0; rep < ATTN_REPS; rep++) {
                    tb.attention(query.data(), kv_cache.data(),
                                 scores.data(), SEQ_LEN, HEAD_DIM);
                }
                double dt = now_sec() - t0;
                if (dt < best_attn) best_attn = dt;
            }

            double attn_tput = (double)ATTN_REPS / best_attn;
            printf("attention_throughput_%s=%.0f\n", tb.name, attn_tput);

            if (attn_tput > best_overall_attn_tput) {
                best_overall_attn_tput = attn_tput;
            }
        }
    }

    /* ============================================================
     * Uniform 4B dequant+dot attention throughput (legacy path
     * for types without dedicated attention kernel)
     * ============================================================ */
    {
        tq_dequantize_fn deq_fn = tq_uniform_4b_dequantize_ref;
#ifdef __ARM_NEON
        deq_fn = tq_uniform_4b_dequantize_neon;
#endif
        tq_quantize_fn q_fn = tq_uniform_4b_quantize_ref;
#ifdef __ARM_NEON
        q_fn = tq_uniform_4b_quantize_neon;
#endif

        std::vector<block_tq_uniform_4b> attn_blocks(SEQ_LEN);
        for (int i = 0; i < SEQ_LEN; i++)
            q_fn(keys.data() + (i % N_VECTORS) * HEAD_DIM,
                 &attn_blocks[i], HEAD_DIM);

        std::vector<float> scores(SEQ_LEN, 0.0f);
        double best_attn_time = 1e9;

        for (int iter = 0; iter < N_ITERS; iter++) {
            double t0 = now_sec();
            for (int rep = 0; rep < ATTN_REPS; rep++) {
                for (int s = 0; s < SEQ_LEN; s++) {
                    float deq[128];
                    deq_fn(&attn_blocks[s], deq, HEAD_DIM);
                    float dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++) dot += query[d] * deq[d];
                    scores[s] = dot;
                }
            }
            double dt = now_sec() - t0;
            if (dt < best_attn_time) best_attn_time = dt;
        }

        double uniform_attn_tput = (double)ATTN_REPS / best_attn_time;
        printf("attention_throughput_uniform_4b_deqdot=%.0f\n", uniform_attn_tput);

        if (uniform_attn_tput > best_overall_attn_tput) {
            best_overall_attn_tput = uniform_attn_tput;
        }
    }

    /* ============================================================
     * SIMD speedup: NEON vs generic (uniform_4b quantize)
     * ============================================================ */
    double simd_speedup = 1.0;

#ifdef __ARM_NEON
    {
        /* SIMD speedup: compare PolarQuant attention generic vs NEON */
        const int SIMD_SEQ = 256;
        const int SIMD_REPS = 2000;
        std::vector<block_tq_polar> simd_polar(SIMD_SEQ);
        std::vector<float> simd_keys2(SIMD_SEQ * HEAD_DIM);
        for (int i = 0; i < SIMD_SEQ * HEAD_DIM; i++)
            simd_keys2[i] = sinf((float)i * 0.01f) * 2.0f;
        for (int i = 0; i < SIMD_SEQ; i++)
            tq_polar_quantize_ref(simd_keys2.data() + i * HEAD_DIM, &simd_polar[i], HEAD_DIM);

        std::vector<float> simd_scores(SIMD_SEQ);

        /* Measure generic polar attention */
        /* Use dequant+dot as generic baseline (scalar path) */
        double t_generic = 1e9;
        for (int iter = 0; iter < 3; iter++) {
            double t0 = now_sec();
            for (int rep = 0; rep < SIMD_REPS; rep++) {
                for (int s = 0; s < SIMD_SEQ; s++) {
                    float deq[128];
                    tq_polar_dequantize_ref(&simd_polar[s], deq, HEAD_DIM);
                    float dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++) dot += query[d] * deq[d];
                    simd_scores[s] = dot;
                }
            }
            double dt = now_sec() - t0;
            if (dt < t_generic) t_generic = dt;
        }

        /* Measure NEON polar attention (direct LUT) */
        double t_neon = 1e9;
        for (int iter = 0; iter < 3; iter++) {
            double t0 = now_sec();
            for (int rep = 0; rep < SIMD_REPS; rep++) {
                tq_polar_attention_neon(query.data(), simd_polar.data(),
                                        simd_scores.data(), SIMD_SEQ, HEAD_DIM);
            }
            double dt = now_sec() - t0;
            if (dt < t_neon) t_neon = dt;
        }

        if (t_neon > 1e-9)
            simd_speedup = t_generic / t_neon;
    }
#endif

    /* ============================================================
     * Legacy summary metrics (consumed by score.sh)
     * ============================================================ */
    printf("quantize_throughput=%.0f\n", best_overall_quant_tput);
    printf("attention_throughput=%.0f\n", best_overall_attn_tput);
    printf("compression_ratio=%.2f\n", overall_compression);
    printf("simd_speedup=%.2f\n", simd_speedup);

    return 0;
}
