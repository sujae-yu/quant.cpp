/**
 * TurboQuant -- QJL (Quantized Johnson-Lindenstrauss) ROCm/HIP kernels
 *
 * Based on refs/QJL/qjl_kernel/csrc/qjl_quant_kernel.cu patterns.
 * Performs 1-bit sign hash quantization with outlier detection and
 * Hamming distance-based attention scoring on AMD GPU.
 *
 * Mechanically converted from CUDA (tq_qjl.cu).
 */
#ifdef TQ_BUILD_ROCM

#include "tq_rocm_common.h"
#include <math.h>

/* ============================================================
 * QJL quantize kernel
 *
 * Grid:  (num_keys, sketch_dim / (WARPS_PER_BLOCK * 8), 1)
 *   - blockIdx.x: key index within the batch
 *   - blockIdx.y: sketch chunk index (each chunk = WARPS_PER_BLOCK * 8 bits)
 * Block: (WARP_SIZE, WARPS_PER_BLOCK, 1)
 *   - threadIdx.x: lane within warp (for reduction over embedding dim)
 *   - threadIdx.y: warp index (each warp handles one sketch projection)
 *
 * Each warp computes the dot product of one key with one projection
 * vector, extracts the sign bit, and packs 8 warp results into a byte.
 * ============================================================ */

#define QJL_WARPS_PER_BLOCK  8
#define QJL_EMB_DIM_MAX     256

__global__ void tq_qjl_quantize_kernel(
    const float* __restrict__ keys,     /* [num_keys, emb_dim] */
    tq_qjl_block_d* __restrict__ out,   /* [num_keys] */
    int num_keys,
    int emb_dim)
{
    const int key_idx     = blockIdx.x;
    const int sketch_base = blockIdx.y * (QJL_WARPS_PER_BLOCK * 8);
    const int lane        = threadIdx.x;
    const int warp_id     = threadIdx.y;

    if (key_idx >= num_keys) return;

    /* Load key into shared memory for fast repeated access */
    __shared__ float s_key[QJL_EMB_DIM_MAX];
    {
        int tid_flat = warp_id * TQ_WARP_SIZE + lane;
        int stride   = QJL_WARPS_PER_BLOCK * TQ_WARP_SIZE;
        for (int i = tid_flat; i < emb_dim; i += stride) {
            s_key[i] = keys[key_idx * emb_dim + i];
        }
    }
    __syncthreads();

    /* Each warp processes 8 consecutive sketch dimensions.
       warp_id selects which group of 8, lane reduces over emb_dim. */
    int sketch_group_base = sketch_base + warp_id * 8;
    uint8_t packed_byte = 0;

    for (int bit = 0; bit < 8; bit++) {
        int sketch_idx = sketch_group_base + bit;
        if (sketch_idx >= TQ_SKETCH_DIM_ROCM) break;

        /* Compute partial dot product: key[d] * random(d, sketch_idx) */
        float partial = 0.0f;
        for (int d = lane; d < emb_dim; d += TQ_WARP_SIZE) {
            partial += s_key[d] * tq_random_entry_d(d, sketch_idx);
        }

        /* Warp-level reduction to get full dot product */
        float dot = warpReduceSum(partial);

        /* Lane 0 extracts the sign bit */
        if (lane == 0) {
            if (dot > 0.0f) {
                packed_byte |= (1u << bit);
            }
        }
    }

    /* Lane 0 of each warp writes its packed byte */
    if (lane == 0 && sketch_group_base < TQ_SKETCH_DIM_ROCM) {
        int byte_idx = sketch_group_base / 8;
        out[key_idx].hash[byte_idx] = packed_byte;
    }

    /* First warp of the first sketch chunk computes norm and outliers */
    if (blockIdx.y == 0 && warp_id == 0) {
        /* L2 norm computation */
        float norm_partial = 0.0f;
        for (int d = lane; d < emb_dim; d += TQ_WARP_SIZE) {
            float v = s_key[d];
            norm_partial += v * v;
        }
        float norm_sq = warpReduceSum(norm_partial);

        if (lane == 0) {
            out[key_idx].norm = tq_float_to_half(sqrtf(norm_sq));
        }

        /* Outlier detection: find top-k largest absolute values */
        /* Use a simple serial scan on lane 0 (4 outliers only) */
        if (lane == 0) {
            float abs_max[TQ_OUTLIERS_ROCM];
            uint8_t max_idx[TQ_OUTLIERS_ROCM];
            for (int o = 0; o < TQ_OUTLIERS_ROCM; o++) {
                abs_max[o] = -1.0f;
                max_idx[o] = 0;
            }

            for (int d = 0; d < emb_dim && d < 256; d++) {
                float av = fabsf(s_key[d]);
                /* Insert into sorted outlier list */
                for (int o = 0; o < TQ_OUTLIERS_ROCM; o++) {
                    if (av > abs_max[o]) {
                        /* Shift down */
                        for (int j = TQ_OUTLIERS_ROCM - 1; j > o; j--) {
                            abs_max[j] = abs_max[j - 1];
                            max_idx[j] = max_idx[j - 1];
                        }
                        abs_max[o] = av;
                        max_idx[o] = (uint8_t)d;
                        break;
                    }
                }
            }

            float outlier_norm_sq = 0.0f;
            for (int o = 0; o < TQ_OUTLIERS_ROCM; o++) {
                out[key_idx].outlier_idx[o] = max_idx[o];
                float v = s_key[max_idx[o]];
                outlier_norm_sq += v * v;
            }
            out[key_idx].outlier_norm = tq_float_to_half(sqrtf(outlier_norm_sq));
        }
    }
}

/* ============================================================
 * QJL attention (Hamming distance) kernel
 *
 * Grid:  (seq_len, 1, 1)
 * Block: (WARP_SIZE, num_warps, 1)
 *
 * Each block computes one attention score by:
 *  1. Projecting the query through the random matrix (sign bits)
 *  2. XOR + popcount with stored key hash
 *  3. Converting Hamming distance to estimated dot product
 * ============================================================ */
__global__ void tq_qjl_attention_kernel(
    const float* __restrict__ query,         /* [head_dim] */
    const tq_qjl_block_d* __restrict__ keys, /* [seq_len] */
    float* __restrict__ scores,               /* [seq_len] */
    int seq_len,
    int head_dim)
{
    const int s    = blockIdx.x;   /* sequence position */
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;

    if (s >= seq_len) return;

    /* Load query into shared memory */
    __shared__ float s_query[QJL_EMB_DIM_MAX];
    {
        int tid_flat = warp_id * TQ_WARP_SIZE + lane;
        int stride = blockDim.x * blockDim.y;
        for (int i = tid_flat; i < head_dim; i += stride) {
            s_query[i] = query[i];
        }
    }
    __syncthreads();

    /* Compute query projection sign bits and compare with key hash.
       Distribute sketch bytes across threads within warp 0. */
    int hash_bytes = TQ_SKETCH_DIM_ROCM / 8; /* 32 bytes */
    int agree_count = 0;

    if (warp_id == 0) {
        /* Each lane handles one or more hash bytes */
        for (int b = lane; b < hash_bytes; b += TQ_WARP_SIZE) {
            /* Compute 8 query projections for this byte */
            uint8_t q_hash = 0;
            for (int bit = 0; bit < 8; bit++) {
                int sketch_idx = b * 8 + bit;
                if (sketch_idx >= TQ_SKETCH_DIM_ROCM) break;

                float proj = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    proj += s_query[d] * tq_random_entry_d(d, sketch_idx);
                }
                if (proj > 0.0f) {
                    q_hash |= (1u << bit);
                }
            }

            /* XOR with key hash, popcount gives disagreements */
            uint8_t key_hash = keys[s].hash[b];
            uint8_t xored = q_hash ^ key_hash;
            int disagree = __popc((unsigned int)xored);
            agree_count += (8 - disagree);
        }

        /* Warp reduction of agreement count */
        agree_count = warpReduceSumInt(agree_count);
    }

    /* Lane 0 of warp 0 computes final score */
    if (warp_id == 0 && lane == 0) {
        float key_norm = tq_half_to_float(keys[s].norm);

        /* Query norm */
        float q_norm_sq = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            q_norm_sq += s_query[d] * s_query[d];
        }
        float q_norm = sqrtf(q_norm_sq);

        /* fraction of agreements -> cosine estimate */
        float frac = (float)agree_count / TQ_SKETCH_DIM_ROCM;
        float cos_est = cosf(3.14159265f * (1.0f - frac));

        scores[s] = cos_est * q_norm * key_norm;
    }
}

/* ============================================================
 * Host-callable wrappers
 * ============================================================ */

extern "C" void tq_qjl_quantize_rocm(
    const float* d_keys,
    void* d_out,
    int num_keys,
    int emb_dim,
    hipStream_t stream)
{
    int sketch_chunks = (TQ_SKETCH_DIM_ROCM + QJL_WARPS_PER_BLOCK * 8 - 1)
                        / (QJL_WARPS_PER_BLOCK * 8);
    dim3 grid(num_keys, sketch_chunks, 1);
    dim3 block(TQ_WARP_SIZE, QJL_WARPS_PER_BLOCK, 1);

    hipLaunchKernelGGL(tq_qjl_quantize_kernel,
        grid, block, 0, stream,
        d_keys,
        reinterpret_cast<tq_qjl_block_d*>(d_out),
        num_keys,
        emb_dim);
}

extern "C" void tq_qjl_attention_rocm(
    const float* d_query,
    const void* d_keys,
    float* d_scores,
    int seq_len,
    int head_dim,
    hipStream_t stream)
{
    dim3 grid(seq_len, 1, 1);
    dim3 block(TQ_WARP_SIZE, 1, 1); /* single warp per block */

    hipLaunchKernelGGL(tq_qjl_attention_kernel,
        grid, block, 0, stream,
        d_query,
        reinterpret_cast<const tq_qjl_block_d*>(d_keys),
        d_scores,
        seq_len,
        head_dim);
}

#endif /* TQ_BUILD_ROCM */
