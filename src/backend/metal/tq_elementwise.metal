/**
 * TurboQuant -- Element-wise Metal compute shaders
 *
 * Provides GPU kernels for operations between matmuls that would
 * otherwise force GPU->CPU->GPU round-trips:
 *   - RMSNorm (with threadgroup reduction)
 *   - SiLU activation
 *   - Element-wise multiply
 *   - Vector add
 */
#include <metal_stdlib>
using namespace metal;

/* ============================================================
 * SIMD-group sum reduction (matches tq_polar.metal helpers)
 * ============================================================ */

inline float simd_reduce_sum_ew(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

/* ============================================================
 * RMSNorm kernel
 *
 * out[i] = (x[i] / rms(x)) * weight[i]
 * rms(x) = sqrt(mean(x^2) + eps)
 *
 * Two-phase design:
 *   Phase 1: Parallel reduction to compute sum of squares.
 *   Phase 2: Each thread normalizes and scales its element(s).
 *
 * Dispatch: one threadgroup per row (n elements).
 * Threadgroup size: 256 threads (8 SIMD groups of 32).
 * Each thread handles ceil(n / tgsize) elements.
 * ============================================================ */
kernel void rmsnorm(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgsize     [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_gid   [[simdgroup_index_in_threadgroup]])
{
    /* Scratch for cross-SIMD-group reduction (max 8 SIMD groups for TG=256) */
    threadgroup float scratch[8];

    /* Phase 1: accumulate sum of squares */
    float ss = 0.0f;
    for (uint i = tid; i < n; i += tgsize) {
        float v = x[i];
        ss += v * v;
    }

    /* SIMD-group reduction */
    ss = simd_reduce_sum_ew(ss);
    uint num_simd_groups = (tgsize + 31) / 32;

    if (simd_lane == 0) {
        scratch[simd_gid] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Final reduction in first SIMD group */
    if (simd_gid == 0) {
        float val = (tid < num_simd_groups) ? scratch[tid] : 0.0f;
        val = simd_reduce_sum_ew(val);
        if (tid == 0) {
            scratch[0] = rsqrt(val / float(n) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Phase 2: normalize and scale */
    float inv_rms = scratch[0];
    for (uint i = tid; i < n; i += tgsize) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/* ============================================================
 * SiLU (Sigmoid Linear Unit) activation
 *
 * out[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
 *
 * Dispatch: grid covers all n elements, one thread per element.
 * ============================================================ */
kernel void silu(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        float v = x[tid];
        out[tid] = v / (1.0f + exp(-v));
    }
}

/* ============================================================
 * Element-wise multiply
 *
 * out[i] = a[i] * b[i]
 *
 * Dispatch: grid covers all n elements, one thread per element.
 * ============================================================ */
kernel void mul_elementwise(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        out[tid] = a[tid] * b[tid];
    }
}

/* ============================================================
 * Vector add
 *
 * out[i] = a[i] + b[i]
 *
 * Dispatch: grid covers all n elements, one thread per element.
 * ============================================================ */
kernel void add_vectors(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      n   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
}

/* ============================================================
 * RoPE (Rotary Position Embedding)
 *
 * Applies rotation to pairs (x[2i], x[2i+1]) using:
 *   theta = pos * base^(-2i/head_dim)
 *   x'[2i]   = x[2i]*cos(theta) - x[2i+1]*sin(theta)
 *   x'[2i+1] = x[2i]*sin(theta) + x[2i+1]*cos(theta)
 *
 * Applies to both Q (n_heads heads) and K (n_kv_heads heads)
 * packed contiguously: Q[0..n_heads*head_dim-1], K follows.
 *
 * Dispatch: one thread per pair in Q and K combined.
 *   Total threads = (n_heads + n_kv_heads) * head_dim / 2
 * ============================================================ */
kernel void rope(
    device float*    q          [[buffer(0)]],
    device float*    k          [[buffer(1)]],
    constant uint&   pos        [[buffer(2)]],
    constant uint&   head_dim   [[buffer(3)]],
    constant uint&   n_heads    [[buffer(4)]],
    constant uint&   n_kv_heads [[buffer(5)]],
    constant float&  rope_base  [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint half_hd = head_dim / 2;
    uint total_q_pairs = n_heads * half_hd;

    device float* vec;
    uint pair_in_head;

    if (id < total_q_pairs) {
        /* Q region */
        uint head = id / half_hd;
        pair_in_head = id % half_hd;
        vec = q + head * head_dim;
    } else {
        /* K region */
        uint kid = id - total_q_pairs;
        uint total_k_pairs = n_kv_heads * half_hd;
        if (kid >= total_k_pairs) return;
        uint head = kid / half_hd;
        pair_in_head = kid % half_hd;
        vec = k + head * head_dim;
    }

    float freq = 1.0f / pow(rope_base, 2.0f * float(pair_in_head) / float(head_dim));
    float theta = float(pos) * freq;
    float cos_t = cos(theta);
    float sin_t = sin(theta);

    uint idx = pair_in_head * 2;
    float v0 = vec[idx];
    float v1 = vec[idx + 1];
    vec[idx]     = v0 * cos_t - v1 * sin_t;
    vec[idx + 1] = v0 * sin_t + v1 * cos_t;
}

/* ============================================================
 * GELU with tanh approximation
 *
 * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * In-place: x[i] = gelu(x[i])
 * Dispatch: grid covers all n elements, one thread per element.
 * ============================================================ */
kernel void gelu_tanh(
    device float*  x   [[buffer(0)]],
    constant uint& n   [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        float v = x[tid];
        /* sqrt(2/pi) ≈ 0.7978845608 */
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[tid] = 0.5f * v * (1.0f + tanh(inner));
    }
}

/* ============================================================
 * Softmax (in-place, per-head)
 *
 * Each threadgroup processes one head's scores[0..len-1].
 * Two-pass: find max, then compute exp and sum, then normalize.
 *
 * Dispatch: threadgroups = n_heads, threads_per_threadgroup = 256
 * ============================================================ */
kernel void softmax_inplace(
    device float*       x    [[buffer(0)]],
    constant uint&      len  [[buffer(1)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[8];

    device float* row = x + gid * len;

    /* Phase 1: find max */
    float local_max = -INFINITY;
    for (uint i = tid; i < len; i += tgsize) {
        float v = row[i];
        if (v > local_max) local_max = v;
    }

    /* SIMD reduction for max */
    local_max = simd_max(local_max);
    uint num_simd = (tgsize + 31) / 32;
    if (simd_lane == 0) scratch[simd_gid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        float val = (tid < num_simd) ? scratch[tid] : -INFINITY;
        val = simd_max(val);
        if (tid == 0) scratch[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = scratch[0];

    /* Phase 2: exp and sum */
    float local_sum = 0.0f;
    for (uint i = tid; i < len; i += tgsize) {
        float e = exp(row[i] - max_val);
        row[i] = e;
        local_sum += e;
    }

    /* SIMD reduction for sum */
    local_sum = simd_reduce_sum_ew(local_sum);
    if (simd_lane == 0) scratch[simd_gid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        float val = (tid < num_simd) ? scratch[tid] : 0.0f;
        val = simd_reduce_sum_ew(val);
        if (tid == 0) scratch[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / scratch[0];

    /* Phase 3: normalize */
    for (uint i = tid; i < len; i += tgsize) {
        row[i] *= inv_sum;
    }
}

/* ============================================================
 * Attention Q·K scoring
 *
 * For each head h, compute: scores[h * seq_len + t] = dot(Q_h, K_cache[t, h])
 * where K_cache layout is [seq_len, n_kv_heads, head_dim].
 *
 * With GQA: multiple Q heads share one KV head (kv_mul = n_heads / n_kv_heads).
 *
 * Dispatch: one threadgroup per (head, position) pair.
 *   Grid = (n_heads * seq_len, 1, 1), threadgroup = (256, 1, 1)
 * ============================================================ */
kernel void attention_qk(
    device const float* q         [[buffer(0)]],
    device const float* k_cache   [[buffer(1)]],
    device float*       scores    [[buffer(2)]],
    constant uint&      head_dim  [[buffer(3)]],
    constant uint&      seq_len   [[buffer(4)]],
    constant uint&      n_heads   [[buffer(5)]],
    constant uint&      n_kv_heads[[buffer(6)]],
    constant uint&      kv_dim    [[buffer(7)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[8];

    uint h = gid / seq_len;       /* query head index */
    uint t = gid % seq_len;       /* position in sequence */
    if (h >= n_heads) return;

    /* GQA: map query head to KV head */
    uint kv_mul = n_heads / n_kv_heads;
    uint kv_h = h / kv_mul;

    device const float* q_head = q + h * head_dim;
    /* K cache layout: [seq_len * kv_dim], position t at offset t * kv_dim + kv_h * head_dim */
    device const float* k_vec = k_cache + t * kv_dim + kv_h * head_dim;

    /* Parallel dot product */
    float dot = 0.0f;
    for (uint i = tid; i < head_dim; i += tgsize) {
        dot += q_head[i] * k_vec[i];
    }

    /* SIMD reduction */
    dot = simd_reduce_sum_ew(dot);
    uint num_simd = (tgsize + 31) / 32;
    if (simd_lane == 0) scratch[simd_gid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        float val = (tid < num_simd) ? scratch[tid] : 0.0f;
        val = simd_reduce_sum_ew(val);
        if (tid == 0) scratch[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        /* Scale by 1/sqrt(head_dim) */
        scores[h * seq_len + t] = scratch[0] * rsqrt(float(head_dim));
    }
}

/* ============================================================
 * Attention value weighted sum
 *
 * For each head h: output[h*head_dim + d] = sum_t(attn[h*seq_len+t] * V[t, kv_h, d])
 * V cache layout: [seq_len, n_kv_heads, head_dim] (same as K cache).
 *
 * Dispatch: one threadgroup per (head, head_dim_element) pair.
 *   Grid = (n_heads * head_dim, 1, 1), threadgroup = (256, 1, 1)
 *   Each threadgroup reduces across seq_len for one output element.
 * ============================================================ */
kernel void attention_v(
    device const float* attn_weights [[buffer(0)]],
    device const float* v_cache      [[buffer(1)]],
    device float*       output       [[buffer(2)]],
    constant uint&      head_dim     [[buffer(3)]],
    constant uint&      seq_len      [[buffer(4)]],
    constant uint&      n_heads      [[buffer(5)]],
    constant uint&      n_kv_heads   [[buffer(6)]],
    constant uint&      kv_dim       [[buffer(7)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_gid  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[8];

    uint h = gid / head_dim;       /* query head index */
    uint d = gid % head_dim;       /* element within head */
    if (h >= n_heads) return;

    /* GQA: map query head to KV head */
    uint kv_mul = n_heads / n_kv_heads;
    uint kv_h = h / kv_mul;

    device const float* attn_h = attn_weights + h * seq_len;

    /* Parallel weighted sum across seq positions */
    float sum = 0.0f;
    for (uint t = tid; t < seq_len; t += tgsize) {
        sum += attn_h[t] * v_cache[t * kv_dim + kv_h * head_dim + d];
    }

    /* SIMD reduction */
    sum = simd_reduce_sum_ew(sum);
    uint num_simd = (tgsize + 31) / 32;
    if (simd_lane == 0) scratch[simd_gid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        float val = (tid < num_simd) ? scratch[tid] : 0.0f;
        val = simd_reduce_sum_ew(val);
        if (tid == 0) scratch[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        output[h * head_dim + d] = scratch[0];
    }
}

/* ============================================================
 * In-place vector add (aliased output)
 *
 * a[i] += b[i]
 *
 * Unlike add_vectors which writes to separate output, this
 * adds b into a in-place. Used in residual connections where
 * we want x += xb2 without a separate output buffer.
 *
 * Dispatch: grid covers all n elements, one thread per element.
 * ============================================================ */
kernel void add_inplace(
    device float*       a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n) {
        a[tid] += b[tid];
    }
}
