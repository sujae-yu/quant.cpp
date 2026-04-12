/**
 * ARM SVE backend stub — scaffolding for Scalable Vector Extension kernels
 *
 * SVE is needed for AWS Graviton3/4 and other modern ARM servers.
 * Currently all functions delegate to the generic reference implementations.
 * Replace with SVE intrinsics for real optimization.
 *
 * Only compiled when __ARM_FEATURE_SVE is defined.
 */

#include "turboquant/turboquant.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>

/* ================================================================
 * Uniform 4-bit — SVE stubs (delegate to reference)
 * ================================================================ */

extern void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);

void tq_uniform_4b_quantize_sve(const float* src, void* dst, int n) {
    /* TODO: SVE implementation — use svptrue/svld1/svmin/svmax for vectorized min-max */
    tq_uniform_4b_quantize_ref(src, dst, n);
}

void tq_uniform_4b_dequantize_sve(const void* src, float* dst, int n) {
    /* TODO: SVE implementation */
    tq_uniform_4b_dequantize_ref(src, dst, n);
}

/* ================================================================
 * Polar 3/4-bit — SVE stubs (delegate to reference)
 * ================================================================ */

extern void tq_polar_quantize_ref(const float* src, void* dst, int n);
extern void tq_polar_dequantize_ref(const void* src, float* dst, int n);

void tq_polar_quantize_sve(const float* src, void* dst, int n) {
    /* TODO: SVE implementation — vectorize L2 norm + angular quantization */
    tq_polar_quantize_ref(src, dst, n);
}

void tq_polar_dequantize_sve(const void* src, float* dst, int n) {
    /* TODO: SVE implementation */
    tq_polar_dequantize_ref(src, dst, n);
}

/* ================================================================
 * QJL 1-bit — SVE stubs (delegate to reference)
 * ================================================================ */

extern void tq_qjl_quantize_ref(const float* src, void* dst, int n);
extern void tq_qjl_attention_ref(const float* q, const void* kv,
                                  float* s, int seq, int hd);

void tq_qjl_quantize_sve(const float* src, void* dst, int n) {
    /* TODO: SVE implementation — vectorize sign hashing with svcompact */
    tq_qjl_quantize_ref(src, dst, n);
}

void tq_qjl_attention_sve(const float* q, const void* kv,
                           float* s, int seq, int hd) {
    /* TODO: SVE implementation — vectorize popcount-based dot product */
    tq_qjl_attention_ref(q, kv, s, seq, hd);
}

#endif /* __ARM_FEATURE_SVE */
