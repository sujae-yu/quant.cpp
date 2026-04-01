/**
 * bench_cblas_expert.c — Benchmark: fused IQ2_XXS dot vs dequant+cblas_sgemv
 *
 * Research question: Is pre-dequantizing expert weights to FP32 and using
 * Apple's Accelerate cblas_sgemv (which leverages the AMX coprocessor)
 * faster than our hand-rolled fused IQ2_XXS NEON dot product?
 *
 * Test matrix dimensions mirror a real MoE expert:
 *   gate/up: [expert_inter=512, hidden_dim=2048]  (512 output rows)
 *   down:    [hidden_dim=2048, expert_inter=512]   (2048 output rows)
 *
 * We benchmark the full SwiGLU FFN dispatch for ONE expert:
 *   Path A (fused):  tq_matmul_gguf x3 (IQ2_XXS fused dot, NEON)
 *   Path B (cblas):  dequant to FP32 once + cblas_sgemv x3
 *   Path C (cblas, cached): cblas_sgemv x3 only (dequant already done)
 *
 * Build (standalone, macOS):
 *   cc -O2 -o bench_cblas_expert bench/bench_cblas_expert.c \
 *      src/engine/tq_gguf_quants.c src/engine/tq_moe.c \
 *      src/engine/tq_transformer.c src/engine/tq_model.c \
 *      -Iinclude -framework Accelerate -lm
 *
 * Or via CMake (bench target links turboquant which has everything).
 */

#include "turboquant/tq_gguf.h"
#include "turboquant/tq_engine.h"

#include <Accelerate/Accelerate.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ============================================================
 * Timing helpers (nanosecond precision via clock_gettime)
 * ============================================================ */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Prevent dead-code elimination */
static volatile float g_sink = 0.0f;

/* ============================================================
 * Synthetic IQ2_XXS block generation
 *
 * IQ2_XXS block: 66 bytes per 256 elements (super-block).
 *   - 2 bytes: fp16 scale (d)
 *   - 64 bytes: 8 groups of 8 bytes each
 *     Per group: 3x uint16 grid indices + 2 bytes sign/scale info
 *
 * We fill with plausible random data: valid grid indices, random
 * signs, small positive scale.
 * ============================================================ */

/* Generate random IQ2_XXS quantized data for (rows x cols) matrix.
 * Returns heap-allocated buffer. cols must be multiple of 256.
 * This is synthetic data; the exact values don't matter for benchmarking,
 * only the data layout and access patterns do. */
static void* make_random_iq2_xxs(int rows, int cols) {
    const int n_per_row = cols;
    const int nb_per_row = n_per_row / 256;      /* super-blocks per row */
    const size_t row_bytes = (size_t)nb_per_row * 66;
    const size_t total = (size_t)rows * row_bytes;

    uint8_t* data = (uint8_t*)calloc(1, total);
    if (!data) return NULL;

    /* Fill with semi-valid data: small scale, random grid indices and signs.
     * The grid indices point into iq2xxs_grid[] (256 entries). We use
     * indices 0-255 to stay in bounds. The sign bits and sub-scale nibble
     * are random. */
    srand(42);
    for (size_t i = 0; i < total; i++) {
        data[i] = (uint8_t)(rand() & 0xFF);
    }

    /* Set plausible fp16 scales at the start of each 66-byte block.
     * fp16 for 0.01 ~ 0x2066 */
    for (int r = 0; r < rows; r++) {
        uint8_t* row_base = data + r * row_bytes;
        for (int b = 0; b < nb_per_row; b++) {
            /* fp16 encoding of ~0.01 */
            uint16_t d_fp16 = 0x2066;
            memcpy(row_base + b * 66, &d_fp16, 2);
        }
    }

    return data;
}

/* ============================================================
 * SwiGLU activation (matching tq_moe.c)
 * ============================================================ */
static inline float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

static void swiglu_apply(float* restrict hb, const float* restrict hb2, int n) {
    for (int i = 0; i < n; i++) {
        hb[i] = fast_silu(hb[i]) * hb2[i];
    }
}

/* ============================================================
 * BENCHMARK
 * ============================================================ */

int main(int argc, char** argv) {
    /* Expert dimensions (Qwen2-MoE style) */
    const int hidden_dim = 2048;  /* model hidden dim */
    const int expert_dim = 512;   /* expert intermediate dim */
    const int n_warmup = 3;
    const int n_iters  = 20;

    (void)argc; (void)argv;

    printf("=== bench_cblas_expert: fused IQ2_XXS vs dequant+cblas_sgemv ===\n");
    printf("Matrix dims: gate/up=[%d,%d], down=[%d,%d]\n",
           expert_dim, hidden_dim, hidden_dim, expert_dim);
    printf("Iterations: %d (warmup: %d)\n\n", n_iters, n_warmup);

    /* Allocate synthetic IQ2_XXS expert weights */
    void* gate_iq2 = make_random_iq2_xxs(expert_dim, hidden_dim);
    void* up_iq2   = make_random_iq2_xxs(expert_dim, hidden_dim);
    void* down_iq2 = make_random_iq2_xxs(hidden_dim, expert_dim);

    if (!gate_iq2 || !up_iq2 || !down_iq2) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Random input vector */
    float* input = (float*)malloc((size_t)hidden_dim * sizeof(float));
    srand(123);
    for (int i = 0; i < hidden_dim; i++)
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    /* Output and workspace buffers */
    float* hb_fused  = (float*)malloc((size_t)expert_dim * sizeof(float));
    float* hb2_fused = (float*)malloc((size_t)expert_dim * sizeof(float));
    float* out_fused = (float*)malloc((size_t)hidden_dim * sizeof(float));

    float* hb_cblas  = (float*)malloc((size_t)expert_dim * sizeof(float));
    float* hb2_cblas = (float*)malloc((size_t)expert_dim * sizeof(float));
    float* out_cblas = (float*)malloc((size_t)hidden_dim * sizeof(float));

    /* Pre-dequantized FP32 weight matrices for cblas path */
    float* gate_fp32 = (float*)malloc((size_t)expert_dim * hidden_dim * sizeof(float));
    float* up_fp32   = (float*)malloc((size_t)expert_dim * hidden_dim * sizeof(float));
    float* down_fp32 = (float*)malloc((size_t)hidden_dim * expert_dim * sizeof(float));

    /* ============================================================
     * Dequantize expert weights to FP32 (one-time cost)
     * ============================================================ */
    printf("--- Dequantization cost (one-time per expert) ---\n");
    {
        uint64_t t0 = now_ns();
        tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, gate_iq2, gate_fp32,
                            expert_dim * hidden_dim);
        tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, up_iq2, up_fp32,
                            expert_dim * hidden_dim);
        tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, down_iq2, down_fp32,
                            hidden_dim * expert_dim);
        uint64_t t1 = now_ns();
        double dequant_ms = (double)(t1 - t0) / 1e6;
        printf("  Dequant 3 matrices (gate+up+down): %.3f ms\n", dequant_ms);

        size_t iq2_bytes = (size_t)(expert_dim * hidden_dim / 256) * 66 * 2
                         + (size_t)(hidden_dim * expert_dim / 256) * 66;
        size_t fp32_bytes = ((size_t)expert_dim * hidden_dim * 2
                           + (size_t)hidden_dim * expert_dim) * sizeof(float);
        printf("  IQ2_XXS memory: %.2f KB -> FP32 memory: %.2f KB (%.1fx expansion)\n",
               (double)iq2_bytes / 1024.0, (double)fp32_bytes / 1024.0,
               (double)fp32_bytes / (double)iq2_bytes);
    }
    printf("\n");

    /* Set thread count for tq_matmul_gguf */
    tq_set_threads(1);  /* Single-threaded for fair comparison */

    /* ============================================================
     * Path A: Fused IQ2_XXS (tq_matmul_gguf with fused_dot_iq2_xxs_neon)
     * ============================================================ */
    printf("--- Path A: Fused IQ2_XXS dot (tq_matmul_gguf) ---\n");
    {
        /* Warmup */
        for (int w = 0; w < n_warmup; w++) {
            tq_matmul_gguf(hb_fused, input, gate_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           expert_dim, hidden_dim);
            tq_matmul_gguf(hb2_fused, input, up_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           expert_dim, hidden_dim);
            swiglu_apply(hb_fused, hb2_fused, expert_dim);
            tq_matmul_gguf(out_fused, hb_fused, down_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           hidden_dim, expert_dim);
            g_sink += out_fused[0];
        }

        /* Timed runs */
        double times_ms[64];
        for (int i = 0; i < n_iters; i++) {
            uint64_t t0 = now_ns();
            tq_matmul_gguf(hb_fused, input, gate_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           expert_dim, hidden_dim);
            tq_matmul_gguf(hb2_fused, input, up_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           expert_dim, hidden_dim);
            swiglu_apply(hb_fused, hb2_fused, expert_dim);
            tq_matmul_gguf(out_fused, hb_fused, down_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           hidden_dim, expert_dim);
            uint64_t t1 = now_ns();
            times_ms[i] = (double)(t1 - t0) / 1e6;
            g_sink += out_fused[0];
        }

        /* Sort for median */
        for (int i = 0; i < n_iters - 1; i++)
            for (int j = i + 1; j < n_iters; j++)
                if (times_ms[j] < times_ms[i]) {
                    double tmp = times_ms[i]; times_ms[i] = times_ms[j]; times_ms[j] = tmp;
                }

        double median = times_ms[n_iters / 2];
        double min_t = times_ms[0];
        double max_t = times_ms[n_iters - 1];
        printf("  Per-expert SwiGLU FFN (single thread):\n");
        printf("    Median: %.3f ms | Min: %.3f ms | Max: %.3f ms\n",
               median, min_t, max_t);
    }
    printf("\n");

    /* ============================================================
     * Path B: Dequant + cblas_sgemv (includes dequant in timing)
     * ============================================================ */
    printf("--- Path B: Dequant + cblas_sgemv (full cost, cold) ---\n");
    {
        float* temp_gate = (float*)malloc((size_t)expert_dim * hidden_dim * sizeof(float));
        float* temp_up   = (float*)malloc((size_t)expert_dim * hidden_dim * sizeof(float));
        float* temp_down = (float*)malloc((size_t)hidden_dim * expert_dim * sizeof(float));

        /* Warmup */
        for (int w = 0; w < n_warmup; w++) {
            tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, gate_iq2, temp_gate,
                                expert_dim * hidden_dim);
            tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, up_iq2, temp_up,
                                expert_dim * hidden_dim);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        expert_dim, hidden_dim,
                        1.0f, temp_gate, hidden_dim,
                        input, 1,
                        0.0f, hb_cblas, 1);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        expert_dim, hidden_dim,
                        1.0f, temp_up, hidden_dim,
                        input, 1,
                        0.0f, hb2_cblas, 1);
            swiglu_apply(hb_cblas, hb2_cblas, expert_dim);
            tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, down_iq2, temp_down,
                                hidden_dim * expert_dim);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        hidden_dim, expert_dim,
                        1.0f, temp_down, expert_dim,
                        hb_cblas, 1,
                        0.0f, out_cblas, 1);
            g_sink += out_cblas[0];
        }

        /* Timed runs */
        double times_ms[64];
        for (int i = 0; i < n_iters; i++) {
            uint64_t t0 = now_ns();
            /* Dequant all 3 matrices */
            tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, gate_iq2, temp_gate,
                                expert_dim * hidden_dim);
            tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, up_iq2, temp_up,
                                expert_dim * hidden_dim);
            /* gate matmul */
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        expert_dim, hidden_dim,
                        1.0f, temp_gate, hidden_dim,
                        input, 1,
                        0.0f, hb_cblas, 1);
            /* up matmul */
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        expert_dim, hidden_dim,
                        1.0f, temp_up, hidden_dim,
                        input, 1,
                        0.0f, hb2_cblas, 1);
            /* SwiGLU */
            swiglu_apply(hb_cblas, hb2_cblas, expert_dim);
            /* down dequant + matmul */
            tq_dequant_row_gguf(TQ_GGML_TYPE_IQ2_XXS, down_iq2, temp_down,
                                hidden_dim * expert_dim);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        hidden_dim, expert_dim,
                        1.0f, temp_down, expert_dim,
                        hb_cblas, 1,
                        0.0f, out_cblas, 1);
            uint64_t t1 = now_ns();
            times_ms[i] = (double)(t1 - t0) / 1e6;
            g_sink += out_cblas[0];
        }

        /* Sort for median */
        for (int i = 0; i < n_iters - 1; i++)
            for (int j = i + 1; j < n_iters; j++)
                if (times_ms[j] < times_ms[i]) {
                    double tmp = times_ms[i]; times_ms[i] = times_ms[j]; times_ms[j] = tmp;
                }

        double median = times_ms[n_iters / 2];
        double min_t = times_ms[0];
        double max_t = times_ms[n_iters - 1];
        printf("  Per-expert SwiGLU FFN (dequant + cblas, single thread):\n");
        printf("    Median: %.3f ms | Min: %.3f ms | Max: %.3f ms\n",
               median, min_t, max_t);

        free(temp_gate);
        free(temp_up);
        free(temp_down);
    }
    printf("\n");

    /* ============================================================
     * Path C: cblas_sgemv only (pre-dequantized / cached FP32)
     *
     * This represents the amortized cost when expert FP32 weights
     * are cached in memory (e.g., LRU FP32 cache for hot experts).
     * ============================================================ */
    printf("--- Path C: cblas_sgemv only (pre-dequantized FP32, cached) ---\n");
    {
        /* Warmup */
        for (int w = 0; w < n_warmup; w++) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        expert_dim, hidden_dim,
                        1.0f, gate_fp32, hidden_dim,
                        input, 1,
                        0.0f, hb_cblas, 1);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        expert_dim, hidden_dim,
                        1.0f, up_fp32, hidden_dim,
                        input, 1,
                        0.0f, hb2_cblas, 1);
            swiglu_apply(hb_cblas, hb2_cblas, expert_dim);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        hidden_dim, expert_dim,
                        1.0f, down_fp32, expert_dim,
                        hb_cblas, 1,
                        0.0f, out_cblas, 1);
            g_sink += out_cblas[0];
        }

        /* Timed runs */
        double times_ms[64];
        for (int i = 0; i < n_iters; i++) {
            uint64_t t0 = now_ns();
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        expert_dim, hidden_dim,
                        1.0f, gate_fp32, hidden_dim,
                        input, 1,
                        0.0f, hb_cblas, 1);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        expert_dim, hidden_dim,
                        1.0f, up_fp32, hidden_dim,
                        input, 1,
                        0.0f, hb2_cblas, 1);
            swiglu_apply(hb_cblas, hb2_cblas, expert_dim);
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        hidden_dim, expert_dim,
                        1.0f, down_fp32, expert_dim,
                        hb_cblas, 1,
                        0.0f, out_cblas, 1);
            uint64_t t1 = now_ns();
            times_ms[i] = (double)(t1 - t0) / 1e6;
            g_sink += out_cblas[0];
        }

        /* Sort for median */
        for (int i = 0; i < n_iters - 1; i++)
            for (int j = i + 1; j < n_iters; j++)
                if (times_ms[j] < times_ms[i]) {
                    double tmp = times_ms[i]; times_ms[i] = times_ms[j]; times_ms[j] = tmp;
                }

        double median = times_ms[n_iters / 2];
        double min_t = times_ms[0];
        double max_t = times_ms[n_iters - 1];
        printf("  Per-expert SwiGLU FFN (cblas only, cached FP32):\n");
        printf("    Median: %.3f ms | Min: %.3f ms | Max: %.3f ms\n",
               median, min_t, max_t);
    }
    printf("\n");

    /* ============================================================
     * Path D: Multi-threaded fused IQ2_XXS (default thread count)
     *
     * The real production path uses multiple threads for matmul.
     * ============================================================ */
    printf("--- Path D: Fused IQ2_XXS (multi-threaded, default) ---\n");
    {
        tq_set_threads(4);  /* multi-threaded */

        /* Warmup */
        for (int w = 0; w < n_warmup; w++) {
            tq_matmul_gguf(hb_fused, input, gate_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           expert_dim, hidden_dim);
            tq_matmul_gguf(hb2_fused, input, up_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           expert_dim, hidden_dim);
            swiglu_apply(hb_fused, hb2_fused, expert_dim);
            tq_matmul_gguf(out_fused, hb_fused, down_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           hidden_dim, expert_dim);
            g_sink += out_fused[0];
        }

        /* Timed runs */
        double times_ms[64];
        for (int i = 0; i < n_iters; i++) {
            uint64_t t0 = now_ns();
            tq_matmul_gguf(hb_fused, input, gate_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           expert_dim, hidden_dim);
            tq_matmul_gguf(hb2_fused, input, up_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           expert_dim, hidden_dim);
            swiglu_apply(hb_fused, hb2_fused, expert_dim);
            tq_matmul_gguf(out_fused, hb_fused, down_iq2, TQ_GGML_TYPE_IQ2_XXS,
                           hidden_dim, expert_dim);
            uint64_t t1 = now_ns();
            times_ms[i] = (double)(t1 - t0) / 1e6;
            g_sink += out_fused[0];
        }

        /* Sort for median */
        for (int i = 0; i < n_iters - 1; i++)
            for (int j = i + 1; j < n_iters; j++)
                if (times_ms[j] < times_ms[i]) {
                    double tmp = times_ms[i]; times_ms[i] = times_ms[j]; times_ms[j] = tmp;
                }

        double median = times_ms[n_iters / 2];
        double min_t = times_ms[0];
        double max_t = times_ms[n_iters - 1];
        printf("  Per-expert SwiGLU FFN (multi-threaded fused IQ2):\n");
        printf("    Median: %.3f ms | Min: %.3f ms | Max: %.3f ms\n",
               median, min_t, max_t);
        printf("  Threads: %d\n", tq_get_threads());
    }
    printf("\n");

    /* ============================================================
     * Path E: cblas_sgemm batched (all 3 matmuls as sgemm, not sgemv)
     *
     * For vector-matrix multiply, sgemm with M=1 should dispatch
     * to the same AMX kernel as sgemv, but let's verify.
     * ============================================================ */
    printf("--- Path E: cblas_sgemm M=1 (pre-dequantized FP32, cached) ---\n");
    {
        /* Warmup */
        for (int w = 0; w < n_warmup; w++) {
            /* gate: out[1,expert_dim] = input[1,hidden_dim] * gate_fp32^T[hidden_dim,expert_dim] */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        1, expert_dim, hidden_dim,
                        1.0f, input, hidden_dim,
                        gate_fp32, hidden_dim,
                        0.0f, hb_cblas, expert_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        1, expert_dim, hidden_dim,
                        1.0f, input, hidden_dim,
                        up_fp32, hidden_dim,
                        0.0f, hb2_cblas, expert_dim);
            swiglu_apply(hb_cblas, hb2_cblas, expert_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        1, hidden_dim, expert_dim,
                        1.0f, hb_cblas, expert_dim,
                        down_fp32, expert_dim,
                        0.0f, out_cblas, hidden_dim);
            g_sink += out_cblas[0];
        }

        /* Timed runs */
        double times_ms[64];
        for (int i = 0; i < n_iters; i++) {
            uint64_t t0 = now_ns();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        1, expert_dim, hidden_dim,
                        1.0f, input, hidden_dim,
                        gate_fp32, hidden_dim,
                        0.0f, hb_cblas, expert_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        1, expert_dim, hidden_dim,
                        1.0f, input, hidden_dim,
                        up_fp32, hidden_dim,
                        0.0f, hb2_cblas, expert_dim);
            swiglu_apply(hb_cblas, hb2_cblas, expert_dim);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        1, hidden_dim, expert_dim,
                        1.0f, hb_cblas, expert_dim,
                        down_fp32, expert_dim,
                        0.0f, out_cblas, hidden_dim);
            uint64_t t1 = now_ns();
            times_ms[i] = (double)(t1 - t0) / 1e6;
            g_sink += out_cblas[0];
        }

        /* Sort for median */
        for (int i = 0; i < n_iters - 1; i++)
            for (int j = i + 1; j < n_iters; j++)
                if (times_ms[j] < times_ms[i]) {
                    double tmp = times_ms[i]; times_ms[i] = times_ms[j]; times_ms[j] = tmp;
                }

        double median = times_ms[n_iters / 2];
        double min_t = times_ms[0];
        double max_t = times_ms[n_iters - 1];
        printf("  Per-expert SwiGLU FFN (cblas_sgemm M=1, cached FP32):\n");
        printf("    Median: %.3f ms | Min: %.3f ms | Max: %.3f ms\n",
               median, min_t, max_t);
    }
    printf("\n");

    /* ============================================================
     * Summary
     * ============================================================ */
    printf("=== Memory analysis ===\n");
    {
        /* Per expert, IQ2_XXS storage */
        size_t iq2_per_expert = (size_t)(expert_dim * hidden_dim / 256) * 66 * 2   /* gate + up */
                              + (size_t)(hidden_dim * expert_dim / 256) * 66;       /* down */
        /* Per expert, FP32 cached */
        size_t fp32_per_expert = ((size_t)expert_dim * hidden_dim * 2
                                + (size_t)hidden_dim * expert_dim) * sizeof(float);

        printf("  IQ2_XXS per expert: %.1f KB\n", (double)iq2_per_expert / 1024.0);
        printf("  FP32 cache per expert: %.1f KB (%.1fx expansion)\n",
               (double)fp32_per_expert / 1024.0,
               (double)fp32_per_expert / (double)iq2_per_expert);

        /* For 8 active experts per layer (Qwen2-MoE) */
        printf("  FP32 cache for 8 active experts: %.1f MB\n",
               8.0 * (double)fp32_per_expert / (1024.0 * 1024.0));
        printf("  FP32 cache for 8 experts x 28 layers: %.1f MB\n",
               8.0 * 28.0 * (double)fp32_per_expert / (1024.0 * 1024.0));
    }
    printf("\n");

    printf("=== Conclusion ===\n");
    printf("If Path C (cached cblas) << Path A (fused IQ2), then:\n");
    printf("  -> Use LRU FP32 cache + cblas_sgemv for hot experts\n");
    printf("  -> Amortized dequant cost becomes negligible with reuse\n");
    printf("  -> No custom Metal shaders needed\n");
    printf("\n");
    printf("If Path A ~ Path C, then:\n");
    printf("  -> Fused IQ2 NEON is already optimal\n");
    printf("  -> Memory overhead of FP32 cache is not justified\n");

    /* Cleanup */
    free(gate_iq2); free(up_iq2); free(down_iq2);
    free(input);
    free(hb_fused); free(hb2_fused); free(out_fused);
    free(hb_cblas); free(hb2_cblas); free(out_cblas);
    free(gate_fp32); free(up_fp32); free(down_fp32);

    return 0;
}
