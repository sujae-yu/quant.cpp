/**
 * quant.cpp v0.7 — Honest A/B Speed Benchmark
 *
 * Fixes from v1:
 * 1. FP32 baseline uses NEON-optimized dot product (fair comparison)
 * 2. Nanosecond precision for small seq_lens
 * 3. Query Q8 quantization cost explicitly measured and included
 * 4. Volatile sink to prevent dead code elimination
 * 5. Multiple runs, report median (not single run)
 */

#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static volatile float g_sink = 0;

/* ============================================================
 * FP32 NEON-optimized dot product (FAIR baseline)
 * ============================================================ */
static void fp32_attention_neon(const float* query, const float* keys,
                                 float* scores, int seq_len, int head_dim) {
    for (int s = 0; s < seq_len; s++) {
        const float* k = keys + s * head_dim;
#ifdef __ARM_NEON
        float32x4_t acc = vdupq_n_f32(0);
        int d;
        for (d = 0; d + 3 < head_dim; d += 4) {
            float32x4_t vq = vld1q_f32(query + d);
            float32x4_t vk = vld1q_f32(k + d);
            acc = vfmaq_f32(acc, vq, vk);
        }
        float dot = vaddvq_f32(acc);
        for (; d < head_dim; d++) dot += query[d] * k[d];
#else
        float dot = 0;
        for (int d = 0; d < head_dim; d++) dot += query[d] * k[d];
#endif
        scores[s] = dot;
    }
}

/* ============================================================
 * Measurement helpers
 * ============================================================ */
static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

#define MEDIAN_RUNS 7

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static double median_of(double* arr, int n) {
    qsort(arr, n, sizeof(double), cmp_double);
    return arr[n / 2];
}

/* ============================================================
 * External functions
 * ============================================================ */
extern void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_4b_attention_int_ref(const float* query, const void* kv,
                                             float* scores, int seq_len, int head_dim);
#ifdef __ARM_NEON
extern void tq_uniform_4b_attention_int_neon(const float* query, const void* kv,
                                              float* scores, int seq_len, int head_dim);
extern void tq_uniform_4b_attention_neon(const float* query, const void* kv,
                                          float* scores, int seq_len, int head_dim);
#endif

int main(void) {
    printf("\n");
    printf("================================================================\n");
    printf("  quant.cpp v0.7 — Honest A/B Speed Benchmark\n");
    printf("  FP32 NEON vs Integer Q4xQ8 NEON (fair comparison)\n");
    printf("================================================================\n");
    printf("\n");
    printf("  Methodology:\n");
    printf("  - FP32 baseline: NEON vfmaq_f32 (not scalar loop)\n");
    printf("  - Integer path: includes query Q8 quantization cost\n");
    printf("  - Median of %d runs (eliminates outliers)\n", MEDIAN_RUNS);
    printf("  - volatile sink prevents dead code elimination\n");
    printf("\n");

    int configs[][2] = {{128, 64}, {128, 512}, {128, 2048}, {128, 8192}, {256, 512}, {256, 2048}};
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    printf("  %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s\n",
           "dim", "seq", "FP32 NEON", "Deq NEON", "Int NEON",
           "Int/FP32", "Int/Deq");
    printf("  %-6s %-6s | %-10s %-10s %-10s | %-8s %-8s\n",
           "------", "------", "----------", "----------", "----------",
           "--------", "--------");

    for (int ci = 0; ci < n_configs; ci++) {
        int head_dim = configs[ci][0];
        int seq_len = configs[ci][1];
        int reps = (seq_len <= 128) ? 5000 : (seq_len <= 1024) ? 1000 : (seq_len <= 4096) ? 200 : 50;

        /* Setup */
        float* query = (float*)malloc(head_dim * sizeof(float));
        float* fp32_keys = (float*)malloc(seq_len * head_dim * sizeof(float));
        float* scores = (float*)malloc(seq_len * sizeof(float));

        for (int d = 0; d < head_dim; d++) query[d] = cosf(d * 0.05f) * 0.3f;
        for (int i = 0; i < seq_len * head_dim; i++) fp32_keys[i] = sinf(i * 0.001f) * 0.5f;

        /* Quantize keys */
        block_tq_uniform_4b* q4_blocks = (block_tq_uniform_4b*)malloc(seq_len * sizeof(block_tq_uniform_4b));
        for (int s = 0; s < seq_len; s++)
            tq_uniform_4b_quantize_ref(fp32_keys + s * head_dim, &q4_blocks[s], head_dim);

        double times_fp32[MEDIAN_RUNS], times_deq[MEDIAN_RUNS], times_int[MEDIAN_RUNS];

        /* Warmup */
        for (int w = 0; w < 3; w++) {
            fp32_attention_neon(query, fp32_keys, scores, seq_len, head_dim);
            g_sink += scores[0];
        }

        /* === A: FP32 NEON (fair baseline) === */
        for (int r = 0; r < MEDIAN_RUNS; r++) {
            double t0 = now_us();
            for (int i = 0; i < reps; i++) {
                fp32_attention_neon(query, fp32_keys, scores, seq_len, head_dim);
                g_sink += scores[seq_len / 2];
            }
            times_fp32[r] = (now_us() - t0) / reps;
        }

#ifdef __ARM_NEON
        /* === B: NEON Dequant+Dot (old path) === */
        for (int r = 0; r < MEDIAN_RUNS; r++) {
            double t0 = now_us();
            for (int i = 0; i < reps; i++) {
                tq_uniform_4b_attention_neon(query, q4_blocks, scores, seq_len, head_dim);
                g_sink += scores[seq_len / 2];
            }
            times_deq[r] = (now_us() - t0) / reps;
        }

        /* === C: NEON Integer Q4×Q8 (v0.7, includes query quantization) === */
        for (int r = 0; r < MEDIAN_RUNS; r++) {
            double t0 = now_us();
            for (int i = 0; i < reps; i++) {
                /* NOTE: tq_uniform_4b_attention_int_neon internally quantizes
                 * the query each call. This is the HONEST measurement including
                 * query quantization overhead. */
                tq_uniform_4b_attention_int_neon(query, q4_blocks, scores, seq_len, head_dim);
                g_sink += scores[seq_len / 2];
            }
            times_int[r] = (now_us() - t0) / reps;
        }
#else
        for (int r = 0; r < MEDIAN_RUNS; r++) times_deq[r] = times_int[r] = 0;
#endif

        double fp32_us = median_of(times_fp32, MEDIAN_RUNS);
        double deq_us = median_of(times_deq, MEDIAN_RUNS);
        double int_us = median_of(times_int, MEDIAN_RUNS);

        double speedup_vs_fp32 = fp32_us / (int_us > 0.001 ? int_us : 0.001);
        double speedup_vs_deq = deq_us / (int_us > 0.001 ? int_us : 0.001);

        /* Format with appropriate units */
        const char* unit = "us";
        printf("  %-6d %-6d | %7.1f %-2s %7.1f %-2s %7.1f %-2s | %6.1fx   %6.1fx\n",
               head_dim, seq_len,
               fp32_us, unit, deq_us, unit, int_us, unit,
               speedup_vs_fp32, speedup_vs_deq);

        free(query); free(fp32_keys); free(scores); free(q4_blocks);
    }

    /* === Quality verification === */
    printf("\n  --- Quality Check ---\n");
    {
        int hd = 128, sl = 256;
        float* q = (float*)malloc(hd * sizeof(float));
        float* keys = (float*)malloc(sl * hd * sizeof(float));
        for (int d = 0; d < hd; d++) q[d] = cosf(d * 0.1f) * 0.2f;
        for (int i = 0; i < sl * hd; i++) keys[i] = sinf(i * 0.01f) * 0.3f;

        block_tq_uniform_4b* blk = (block_tq_uniform_4b*)malloc(sl * sizeof(block_tq_uniform_4b));
        for (int s = 0; s < sl; s++)
            tq_uniform_4b_quantize_ref(keys + s * hd, &blk[s], hd);

        float* fp32_scores = (float*)malloc(sl * sizeof(float));
        float* int_scores = (float*)malloc(sl * sizeof(float));

        fp32_attention_neon(q, keys, fp32_scores, sl, hd);
        tq_uniform_4b_attention_int_ref(q, blk, int_scores, sl, hd);

        double dot_ab = 0, sq_a = 0, sq_b = 0;
        for (int s = 0; s < sl; s++) {
            dot_ab += (double)fp32_scores[s] * (double)int_scores[s];
            sq_a += (double)fp32_scores[s] * (double)fp32_scores[s];
            sq_b += (double)int_scores[s] * (double)int_scores[s];
        }
        double cosine = dot_ab / (sqrt(sq_a) * sqrt(sq_b) + 1e-10);
        printf("  FP32 vs Integer attention cosine: %.6f\n", cosine);
        printf("  Grade: %s\n", cosine > 0.99 ? "A+ (>0.99)" : cosine > 0.95 ? "A (>0.95)" : "B");

        free(q); free(keys); free(blk); free(fp32_scores); free(int_scores);
    }

    printf("\n  --- Interpretation ---\n");
    printf("  Int/FP32 > 1.0x = integer attention is FASTER than FP32\n");
    printf("  Int/FP32 < 1.0x = integer attention is SLOWER than FP32\n");
    printf("  Int/Deq  > 1.0x = integer is faster than old dequant path\n");
    printf("\n");

    return 0;
}
