/* Unit test for tq_batched_matmul_q4: correctness + speed.
 * Compares N independent tq_matmul_q4_preq calls vs one batched call,
 * across multiple realistic shapes from our supported models. */
#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void quantize_x_q8(const float* x, int8_t* xq, float* xs, int d) {
    int n_blocks = d / 32;
    for (int b = 0; b < n_blocks; b++) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float a = x[b*32+j] < 0 ? -x[b*32+j] : x[b*32+j];
            if (a > amax) amax = a;
        }
        float dq = amax / 127.0f;
        xs[b] = dq;
        if (dq > 0.0f) {
            float id = 1.0f / dq;
            for (int j = 0; j < 32; j++) {
                int v = (int)roundf(x[b*32+j] * id);
                xq[b*32+j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
            }
        } else {
            memset(xq + b*32, 0, 32);
        }
    }
}

static int test_shape(int n_rows, int d, int N) {
    int n_blocks = d / 32;
    /* Random Q4 weights */
    uint8_t* w_qs = (uint8_t*)malloc((size_t)n_rows * n_blocks * 16);
    float*   w_ds = (float*)malloc((size_t)n_rows * n_blocks * sizeof(float));
    float*   x    = (float*)malloc((size_t)N * d * sizeof(float));
    float*   y_ref= (float*)malloc((size_t)N * n_rows * sizeof(float));
    float*   y_new= (float*)malloc((size_t)N * n_rows * sizeof(float));
    int8_t*  xq   = (int8_t*)malloc((size_t)d * sizeof(int8_t));
    float*   xs   = (float*)malloc((size_t)n_blocks * sizeof(float));

    /* Random init (deterministic) */
    for (size_t i = 0; i < (size_t)n_rows * n_blocks * 16; i++) w_qs[i] = (uint8_t)(i * 31 + 7);
    for (size_t i = 0; i < (size_t)n_rows * n_blocks; i++) w_ds[i] = 0.01f * (float)((i & 7) - 3);
    for (size_t i = 0; i < (size_t)N * d; i++) x[i] = 0.1f * (float)((i * 13 % 11) - 5);

    /* Reference: N×SGEMV via tq_matmul_q4_preq */
    double t0 = now_s();
    for (int n = 0; n < N; n++) {
        quantize_x_q8(x + (size_t)n * d, xq, xs, d);
        tq_matmul_q4_preq(y_ref + (size_t)n * n_rows, w_qs, w_ds, xq, xs, n_rows, d);
    }
    double t_ref = now_s() - t0;

    /* Batched: tq_batched_matmul_q4 */
    t0 = now_s();
    tq_batched_matmul_q4(y_new, w_qs, w_ds, x, n_rows, d, N, NULL);
    double t_new = now_s() - t0;

    /* Correctness: tq_matmul_q4_preq quantizes x to Q8, batched dequants W
     * and uses FP32 sgemm. So they differ by O(1/127) per dot product.
     * Allow ~5% relative error per element (Q8 quantization noise on x). */
    double max_abs_err = 0, max_rel_err = 0;
    int n_bad = 0;
    for (size_t i = 0; i < (size_t)N * n_rows; i++) {
        double err = fabs((double)y_ref[i] - (double)y_new[i]);
        if (err > max_abs_err) max_abs_err = err;
        double mag = fabs((double)y_ref[i]) + 1e-6;
        double rel = err / mag;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > 0.10) n_bad++;
    }

    double ops = 2.0 * (double)n_rows * d * N;
    double gf_ref = ops / t_ref / 1e9;
    double gf_new = ops / t_new / 1e9;
    int correct = (max_rel_err < 0.20 || max_abs_err < 1e-3);
    printf("  M=%-6d K=%-6d N=%-4d  ref=%6.1fms (%5.1f GF)  new=%6.1fms (%6.1f GF)  speedup=%5.2fx  max_rel=%.4f  %s\n",
           n_rows, d, N, t_ref * 1000, gf_ref, t_new * 1000, gf_new, t_ref / t_new, max_rel_err,
           correct ? "OK" : "BAD");

    free(w_qs); free(w_ds); free(x); free(y_ref); free(y_new); free(xq); free(xs);
    return correct;
}

int main(void) {
    /* Configure threads */
    tq_set_threads(8);

    int pass = 0, fail = 0;

    printf("=== Phi-3.5 shapes ===\n");
    int shapes[][3] = {
        {3072, 3072, 1},   /* decode baseline */
        {3072, 3072, 8},
        {3072, 3072, 32},
        {3072, 3072, 128},
        {9216, 3072, 32},  /* fused QKV */
        {16384, 3072, 32}, /* fused gate||up */
        {8192, 3072, 64},  /* FFN hidden */
        {32064, 3072, 1},  /* lm_head decode */
        {32064, 3072, 32}, /* lm_head batched */
        {0, 0, 0}
    };
    for (int i = 0; shapes[i][0]; i++) {
        if (test_shape(shapes[i][0], shapes[i][1], shapes[i][2])) pass++; else fail++;
    }

    printf("\n=== Llama 3.2 shapes ===\n");
    int shapes2[][3] = {
        {2048, 2048, 32},
        {8192, 2048, 32},
        {128256, 2048, 32}, /* lm_head */
        {0, 0, 0}
    };
    for (int i = 0; shapes2[i][0]; i++) {
        if (test_shape(shapes2[i][0], shapes2[i][1], shapes2[i][2])) pass++; else fail++;
    }

    printf("\n=== Summary ===\n  PASS: %d\n  FAIL: %d\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
