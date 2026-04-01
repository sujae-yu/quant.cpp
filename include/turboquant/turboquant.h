#ifndef TURBOQUANT_H
#define TURBOQUANT_H

/**
 * TurboQuant.cpp — Cross-platform KV cache compression library
 *
 * Public C API — single header include for all functionality.
 * Zero external dependencies (libc/libm only).
 */

#include "tq_types.h"
#include "tq_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Version
 * ============================================================ */

#define TQ_VERSION_STRING "0.1.0"

/* ============================================================
 * Error codes
 * ============================================================ */

typedef enum {
    TQ_OK              =  0,
    TQ_ERR_NULL_PTR    = -1,
    TQ_ERR_INVALID_TYPE= -2,
    TQ_ERR_INVALID_DIM = -3,
    TQ_ERR_OUT_OF_MEM  = -4,
    TQ_ERR_NOT_IMPL    = -5,
    TQ_ERR_BACKEND     = -6,
    TQ_ERR_BUFFER_TOO_SMALL = -7,
} tq_status;

const char* tq_status_string(tq_status status);

/* ============================================================
 * Backend selection
 * ============================================================ */

typedef enum {
    TQ_BACKEND_CPU    = 0,
    TQ_BACKEND_CUDA   = 1,
    TQ_BACKEND_METAL  = 2,
    TQ_BACKEND_VULKAN = 3,   /* AMD + cross-platform (SPIR-V compute shaders) */
    TQ_BACKEND_ROCM   = 4,   /* AMD ROCm/HIP (CUDA-compatible API) */
    TQ_BACKEND_AUTO   = 99,
} tq_backend;

/* ============================================================
 * Context (opaque handle)
 * ============================================================ */

typedef struct tq_context tq_context_t;

tq_status   tq_init(tq_context_t** ctx, tq_backend backend);
void        tq_free(tq_context_t* ctx);
tq_backend  tq_get_backend(const tq_context_t* ctx);

/* ============================================================
 * Type info
 * ============================================================ */

const char* tq_type_name(tq_type type);
float       tq_type_bpe(tq_type type);
size_t      tq_type_block_size(tq_type type);
size_t      tq_type_type_size(tq_type type);

/* ============================================================
 * Quantization
 * ============================================================ */

/**
 * Quantize key vectors.
 * @param ctx       TurboQuant context
 * @param keys      Input FP32 keys [n × head_dim]
 * @param n         Number of key vectors
 * @param head_dim  Dimension of each key vector
 * @param type      Quantization type (TQ_TYPE_*)
 * @param out       Output buffer (caller allocated, size from tq_quantize_keys_size)
 * @param out_size  Size of output buffer in bytes
 */
tq_status tq_quantize_keys(tq_context_t* ctx,
                           const float* keys, int n, int head_dim,
                           tq_type type,
                           void* out, size_t out_size);

/** Compute required output buffer size for tq_quantize_keys */
size_t tq_quantize_keys_size(int n, int head_dim, tq_type type);

/**
 * Quantize value vectors.
 * @param ctx       TurboQuant context
 * @param values    Input FP32 values [n × head_dim]
 * @param n         Number of value vectors
 * @param head_dim  Dimension per value
 * @param bits      Quantization bits (2 or 4)
 * @param out       Output buffer
 * @param out_size  Size of output buffer
 */
tq_status tq_quantize_values(tq_context_t* ctx,
                             const float* values, int n, int head_dim,
                             int bits,
                             void* out, size_t out_size);

size_t tq_quantize_values_size(int n, int head_dim, int bits);

/**
 * Dequantize keys back to FP32 (for debugging/testing).
 */
tq_status tq_dequantize_keys(tq_context_t* ctx,
                             const void* quantized, int n, int head_dim,
                             tq_type type,
                             float* out);

/* ============================================================
 * K/V Asymmetric Quantization
 * ============================================================ */

/**
 * Quantize keys and values with independent types (K/V asymmetric).
 * @param ctx          TurboQuant context
 * @param keys         Input FP32 keys [n x head_dim]
 * @param values       Input FP32 values [n x head_dim]
 * @param n            Number of key/value vectors
 * @param head_dim     Dimension of each vector
 * @param key_type     Quantization type for keys (TQ_TYPE_*)
 * @param value_type   Quantization type for values (TQ_TYPE_*)
 * @param key_out      Output buffer for quantized keys
 * @param key_out_size Size of key output buffer in bytes
 * @param val_out      Output buffer for quantized values
 * @param val_out_size Size of value output buffer in bytes
 */
tq_status tq_quantize_kv(tq_context_t* ctx,
                          const float* keys, const float* values,
                          int n, int head_dim,
                          tq_type key_type, tq_type value_type,
                          void* key_out, size_t key_out_size,
                          void* val_out, size_t val_out_size);

/** Compute required output buffer size for keys in K/V asymmetric quantization */
size_t tq_quantize_kv_key_size(int n, int head_dim, tq_type key_type);

/** Compute required output buffer size for values in K/V asymmetric quantization */
size_t tq_quantize_kv_value_size(int n, int head_dim, tq_type value_type);

/* ============================================================
 * Attention
 * ============================================================ */

/**
 * Compute attention scores from quantized KV cache.
 * @param ctx       TurboQuant context
 * @param query     Query vector [head_dim]
 * @param kv_cache  Quantized key cache
 * @param seq_len   Number of cached keys
 * @param head_dim  Dimension per head
 * @param type      Quantization type used for keys
 * @param scores    Output attention scores [seq_len]
 */
tq_status tq_attention(tq_context_t* ctx,
                       const float* query,
                       const void* kv_cache,
                       int seq_len, int head_dim,
                       tq_type type,
                       float* scores);

/* ============================================================
 * Paged cache management
 * ============================================================ */

typedef struct tq_cache tq_cache_t;

tq_status tq_cache_create(tq_cache_t** cache,
                          int block_size, int max_blocks,
                          int num_heads, int head_dim,
                          tq_type default_type);

tq_status tq_cache_append(tq_cache_t* cache,
                          int head_idx,
                          const float* key, const float* value,
                          int head_dim);

tq_status tq_cache_get_block(const tq_cache_t* cache,
                             int head_idx, int block_idx,
                             const void** data, tq_type* type);

int  tq_cache_seq_len(const tq_cache_t* cache, int head_idx);
void tq_cache_free(tq_cache_t* cache);

/** Copy-on-Write: increment ref_count on a block (share it) */
tq_status tq_cache_share_block(tq_cache_t* cache, int head_idx, int block_idx);

/** Copy-on-Write: decrement ref_count, free block data when it reaches 0 */
tq_status tq_cache_free_block(tq_cache_t* cache, int head_idx, int block_idx);

/** Get the quantized value block for a given head and block index */
tq_status tq_cache_get_value(const tq_cache_t* cache, int head_idx, int block_idx,
                             const void** data);

/** Get ref_count of a block (for testing/debugging) */
int tq_cache_block_ref_count(const tq_cache_t* cache, int head_idx, int block_idx);

/* ============================================================
 * Strategy recommendation
 * ============================================================ */

tq_type tq_recommend_strategy(int head_dim, int target_bits,
                              float quality_threshold);

/* ============================================================
 * Random Hadamard Transform (RHT) for quantization pre-processing
 * ============================================================ */

/** In-place Random Hadamard Transform.
 * Decorrelates channels for improved scalar quantization quality.
 * n is rounded down to nearest power of 2 internally.
 * @param data   FP32 vector to transform in-place
 * @param n      Length of the vector
 * @param seed   Random seed for sign flipping (must match for inverse)
 */
void tq_rht_transform(float* data, int n, uint32_t seed);

/** Inverse RHT — recovers original vector from transformed data.
 * Must use the same seed as the forward transform.
 */
void tq_rht_inverse(float* data, int n, uint32_t seed);

/** Quantize key vectors with RHT pre-processing (higher quality).
 * Pipeline: copy -> RHT each vector -> quantize.
 * @param ctx       TurboQuant context
 * @param keys      Input FP32 keys [n x head_dim]
 * @param n         Number of key vectors
 * @param head_dim  Dimension of each key vector (should be power of 2)
 * @param type      Quantization type (TQ_TYPE_*)
 * @param rht_seed  Random seed for RHT sign flipping
 * @param out       Output buffer (same size as tq_quantize_keys_size)
 * @param out_size  Size of output buffer in bytes
 */
tq_status tq_quantize_keys_rht(tq_context_t* ctx,
                                const float* keys, int n, int head_dim,
                                tq_type type, uint32_t rht_seed,
                                void* out, size_t out_size);

/** Dequantize key vectors with RHT post-processing.
 * Pipeline: dequantize -> inverse RHT each vector.
 * Must use the same rht_seed as quantize.
 */
tq_status tq_dequantize_keys_rht(tq_context_t* ctx,
                                  const void* quantized, int n, int head_dim,
                                  tq_type type, uint32_t rht_seed,
                                  float* out);

/* ============================================================
 * Utility
 * ============================================================ */

/** Get format spec for a quantization type */
tq_format_spec_t tq_get_format_spec(tq_type type);

/* ============================================================
 * Convenience functions
 * ============================================================ */

int     tq_type_count(void);
tq_type tq_type_from_name(const char* name);

/* ============================================================
 * Progressive compression
 * ============================================================ */

typedef struct tq_progressive tq_progressive_t;

tq_status tq_progressive_create(tq_progressive_t** out,
                                const tq_progressive_config_t* config,
                                int head_dim, int max_tokens);
tq_status tq_progressive_append(tq_progressive_t* p,
                                const float* key, int head_dim);
tq_status tq_progressive_attention(const tq_progressive_t* p,
                                   const float* query,
                                   float* scores, int head_dim);
int       tq_progressive_count(const tq_progressive_t* p);
void      tq_progressive_free(tq_progressive_t* p);

tq_progressive_config_t tq_progressive_default_config(void);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_H */
