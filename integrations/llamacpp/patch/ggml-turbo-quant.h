/**
 * ggml-turbo-quant.h -- TurboQuant 1-bit KV cache quantization for llama.cpp
 *
 * Apache 2.0 License, QuantumAI Inc.
 *
 * Self-contained implementation of TurboQuant 1-bit KV cache compression.
 * Algorithm: L2-normalize -> Random Hadamard Transform -> sign extraction.
 * Attention: XOR + popcount Hamming distance -> inner product estimator.
 *
 * Reference: TurboQuant (arXiv 2504.19874)
 *   - 1-bit per dimension with RHT decorrelation
 *   - Theoretical attention cosine similarity: 2/pi ~ 0.637
 *   - Compression: 20 bytes per 128 elements (1.25 bpw including metadata)
 *
 * Usage in llama.cpp:
 *   --cache-type-k tq_kv_1b   (for key cache)
 *   --cache-type-v tq_kv_1b   (for value cache, though key-only is recommended)
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Block definition: TurboQuant 1-bit KV cache
 *
 * 20 bytes per 128 elements = 1.25 bits per element (with metadata)
 * Pure sign bits = 1.0 bpw, metadata overhead = 0.25 bpw
 *
 * Layout:
 *   norm     (2B) - L2 norm of original vector, stored as FP16
 *   _pad     (2B) - alignment padding (reserved for future use)
 *   rht_seed (4B) - RHT random seed for inverse transform
 *   signs   (16B) - 128 sign bits, LSB-first packing
 *
 * Total: 24 bytes per 128 elements
 * Compression vs FP16: 256 bytes / 24 bytes = 10.7x
 * Compression vs FP32: 512 bytes / 24 bytes = 21.3x
 * ============================================================ */

#define TQ_KV_1B_BLOCK_SIZE 128

typedef struct {
    uint16_t norm;                              /* L2 norm in FP16               */
    uint16_t _pad;                              /* alignment padding             */
    uint32_t rht_seed;                          /* RHT seed for inverse          */
    uint8_t  signs[TQ_KV_1B_BLOCK_SIZE / 8];   /* 128 sign bits = 16 bytes      */
} block_tq_kv_1b;

/* Compile-time size check: 2 + 2 + 4 + 16 = 24 bytes */
typedef char tq_check_block_size[(sizeof(block_tq_kv_1b) == 24) ? 1 : -1];

/* ============================================================
 * Public API (matches llama.cpp quantize/dequantize convention)
 *
 * k: number of elements (must be multiple of TQ_KV_1B_BLOCK_SIZE)
 * ============================================================ */

/**
 * Quantize a row of float values to 1-bit TurboQuant KV blocks.
 *
 * Pipeline: L2-normalize -> RHT (Walsh-Hadamard + random signs) -> sign extraction.
 *
 * @param x   Input float array (k elements)
 * @param y   Output block array (k / TQ_KV_1B_BLOCK_SIZE blocks)
 * @param k   Number of elements (must be multiple of 128)
 */
void quantize_row_tq_kv_1b_ref(const float * x, block_tq_kv_1b * y, int64_t k);

/**
 * Dequantize 1-bit TurboQuant KV blocks back to float.
 *
 * Pipeline: sign -> scale by sqrt(2/pi)/sqrt(dim) -> inverse RHT -> scale by norm.
 * Note: This is a rough reconstruction. The real value of 1-bit is in Hamming attention.
 *
 * @param x   Input block array
 * @param y   Output float array (k elements)
 * @param k   Number of elements (must be multiple of 128)
 */
void dequantize_row_tq_kv_1b(const block_tq_kv_1b * x, float * y, int64_t k);

/**
 * Compute attention scores between a query and quantized KV cache.
 *
 * Uses XOR + popcount Hamming distance for ultra-fast attention:
 *   score = q_norm * k_norm * sqrt(pi/2) / dim * (2*agree - dim)
 *
 * @param query     Float query vector (head_dim elements)
 * @param kv_cache  Array of quantized key blocks (seq_len blocks)
 * @param scores    Output attention scores (seq_len elements)
 * @param seq_len   Number of keys in the cache
 * @param head_dim  Dimension of each head (must be <= 128)
 */
void tq_kv_1b_attention(const float * query, const block_tq_kv_1b * kv_cache,
                         float * scores, int seq_len, int head_dim);

#ifdef __cplusplus
}
#endif
