/**
 * tq_moe.c — Mixture of Experts routing and expert dispatch
 *
 * Implements top-K expert selection with softmax renormalization,
 * SwiGLU FFN dispatch per expert, shared expert support,
 * runtime LRU Q8_0 cache for routed experts, and memory advise hints.
 */

#include "turboquant/tq_gguf.h"
#include "turboquant/tq_engine.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define TQ_MOE_HAS_NEON 1
#else
#define TQ_MOE_HAS_NEON 0
#endif

#ifdef _MSC_VER
#define __builtin_prefetch(addr, ...) ((void)0)
#endif

#ifdef TQ_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

/* ============================================================
 * Fast SiLU (Swish) approximation
 *
 * silu(x) = x / (1 + exp(-x))
 *
 * Uses Schraudolph's fast exp approximation (~1% accuracy for |x|<10).
 * Called 270 times/token with expert_dim=512, so ~138K calls saved
 * vs. standard expf per token.
 * ============================================================ */
static inline float fast_expf_moe(float x) {
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) return expf(x);
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return v.f;
}

#if TQ_MOE_HAS_NEON
/* Vectorized Schraudolph exp — 4 float lanes in one shot.
 * For |x|>~16 result saturates (exponent field overflow); the SiLU /
 * GELU callers pass negated values in a bounded range (x in [-16, 16]
 * after the swiglu/geglu math) so we don't need the scalar branch here. */
static inline float32x4_t fast_exp_neon(float32x4_t vx) {
    /* Clamp to [-80, 80] so we never produce NaN from int32 overflow.
     * sigmoid saturates at |x|~16 anyway. */
    float32x4_t clamped = vminq_f32(vmaxq_f32(vx, vdupq_n_f32(-80.0f)),
                                    vdupq_n_f32(80.0f));
    float32x4_t scaled = vfmaq_f32(vdupq_n_f32(1065353216.0f),
                                   clamped, vdupq_n_f32(12102203.0f));
    int32x4_t ivals = vcvtq_s32_f32(scaled);
    return vreinterpretq_f32_s32(ivals);
}
#endif

/* Vectorized SwiGLU: hb[i] = silu(hb[i]) * hb2[i].
 * Pillar 1.5 R8: TQ_MOE_EXACT_EXP=1 routes SwiGLU through exact expf
 * instead of Schraudolph approximation. Test if the ~2% precision
 * error of fast_expf compounds over 30-layer × 500-token prefill to
 * produce the Qwen3.6-35B long-context degradation. */
static void swiglu_fused(float* restrict hb, const float* restrict hb2, int n) {
    /* Pillar 1.5 R8: SwiGLU uses exact expf by default. Schraudolph
     * approximation (~2% per-call error) compounds over 30 MoE layers ×
     * 500+ tokens and degraded Qwen3.6 long-context output. Speed cost:
     * unmeasurable on warm decode (SwiGLU is not the bottleneck).
     * Opt-out: TQ_MOE_FAST_EXP=1 reverts to Schraudolph NEON path. */
    static int fast_checked = 0;
    static int use_fast = 0;
    if (!fast_checked) { use_fast = getenv("TQ_MOE_FAST_EXP") != NULL; fast_checked = 1; }
    if (!use_fast) {
        for (int i = 0; i < n; i++) {
            float g = hb[i];
            hb[i] = (g / (1.0f + expf(-g))) * hb2[i];
        }
        return;
    }
#if TQ_MOE_HAS_NEON
    int i = 0;
    float32x4_t vone = vdupq_n_f32(1.0f);
    for (; i + 7 < n; i += 8) {
        /* Process 8 elements: 2x float32x4_t.
         * Fully vectorized — prior code round-tripped through scalar
         * fast_expf_moe(), wasting the NEON SwiGLU pipeline. */
        float32x4_t vg0 = vld1q_f32(hb + i);
        float32x4_t vg1 = vld1q_f32(hb + i + 4);
        float32x4_t vu0 = vld1q_f32(hb2 + i);
        float32x4_t vu1 = vld1q_f32(hb2 + i + 4);

        float32x4_t vexp0 = fast_exp_neon(vnegq_f32(vg0));
        float32x4_t vexp1 = fast_exp_neon(vnegq_f32(vg1));

        /* sigmoid = 1 / (1 + exp(-x))  — Newton-refined reciprocal */
        float32x4_t denom0 = vaddq_f32(vone, vexp0);
        float32x4_t denom1 = vaddq_f32(vone, vexp1);
        float32x4_t vsig0 = vrecpeq_f32(denom0);
        vsig0 = vmulq_f32(vsig0, vrecpsq_f32(denom0, vsig0));
        float32x4_t vsig1 = vrecpeq_f32(denom1);
        vsig1 = vmulq_f32(vsig1, vrecpsq_f32(denom1, vsig1));
        /* silu(x) * up = x * sigmoid(x) * up */
        vst1q_f32(hb + i,     vmulq_f32(vmulq_f32(vg0, vsig0), vu0));
        vst1q_f32(hb + i + 4, vmulq_f32(vmulq_f32(vg1, vsig1), vu1));
    }
    for (; i < n; i++) {
        float g = hb[i];
        hb[i] = (g / (1.0f + fast_expf_moe(-g))) * hb2[i];
    }
#else
    for (int i = 0; i < n; i++) {
        float g = hb[i];
        hb[i] = (g / (1.0f + fast_expf_moe(-g))) * hb2[i];
    }
#endif
}

/* GeGLU: hb[i] = gelu_tanh(hb[i]) * hb2[i]
 * Used by Gemma 4 MoE experts instead of SwiGLU.
 * NEON version uses fast tanh approximation via Schraudolph exp. */
static void geglu_fused(float* restrict hb, const float* restrict hb2, int n) {
    const float c1 = 0.7978845608028654f; /* sqrt(2/pi) */
    const float c2 = 0.044715f;
#if TQ_MOE_HAS_NEON
    int i = 0;
    float32x4_t vc1 = vdupq_n_f32(c1);
    float32x4_t vc2 = vdupq_n_f32(c2);
    float32x4_t vhalf = vdupq_n_f32(0.5f);
    float32x4_t vone = vdupq_n_f32(1.0f);
    float32x4_t vtwo = vdupq_n_f32(2.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(hb + i);
        float32x4_t vu = vld1q_f32(hb2 + i);
        /* arg = c1 * (x + c2 * x^3) = c1 * x * (1 + c2 * x^2) */
        float32x4_t vx2 = vmulq_f32(vx, vx);
        float32x4_t varg = vmulq_f32(vc1, vmulq_f32(vx, vfmaq_f32(vone, vc2, vx2)));
        /* tanh(arg) via sigmoid: tanh(x) = 2*sigmoid(2x) - 1 */
        float32x4_t vexp_arg = vmulq_f32(vtwo, vnegq_f32(varg));
        /* Fast exp via Schraudolph for each lane */
        float neg[4];
        vst1q_f32(neg, vexp_arg);
        float32x4_t vexp = {fast_expf_moe(neg[0]), fast_expf_moe(neg[1]),
                            fast_expf_moe(neg[2]), fast_expf_moe(neg[3])};
        /* sigmoid(2*arg) = 1/(1+exp(-2*arg)) */
        float32x4_t vsig = vrecpeq_f32(vaddq_f32(vone, vexp));
        vsig = vmulq_f32(vsig, vrecpsq_f32(vaddq_f32(vone, vexp), vsig));
        float32x4_t vtanh = vsubq_f32(vmulq_f32(vtwo, vsig), vone);
        /* gelu = 0.5 * x * (1 + tanh) * up */
        float32x4_t vgelu = vmulq_f32(vmulq_f32(vhalf, vx), vaddq_f32(vone, vtanh));
        vst1q_f32(hb + i, vmulq_f32(vgelu, vu));
    }
    for (; i < n; i++) {
        float x = hb[i];
        float t = tanhf(c1 * (x + c2 * x * x * x));
        hb[i] = (0.5f * x * (1.0f + t)) * hb2[i];
    }
#else
    for (int i = 0; i < n; i++) {
        float x = hb[i];
        float t = tanhf(c1 * (x + c2 * x * x * x));
        hb[i] = (0.5f * x * (1.0f + t)) * hb2[i];
    }
#endif
}

#ifndef _WIN32
#include <sys/mman.h>
#endif

/* ============================================================
 * Runtime Expert Q8_0 LRU Cache — REMOVED (Round 13, 2026-04-19)
 *
 * A per-layer Q8_0 LRU (dequant IQ2_XXS → FP32 → Q8_0 blocks, hot
 * slots dispatched through fused_dot_q8_0) was prototyped earlier.
 * Empirically, the dequant cost on cache miss exceeded the direct
 * fused_dot_iq2_xxs_neon cost when expert reuse rate is low (typical
 * for Qwen3.6 K=8/N=256 routing). The LRU was left behind an always-
 * false `if (0 &&` guard for months; removing both the dead dispatch
 * site and its supporting infrastructure eliminates ~200 LOC of dead
 * code and silences a drift vs quant.h (which already had no-op
 * stubs). See git history of this file for the prior implementation.
 * ============================================================ */

void tq_moe_cache_init(int n_layers, const tq_moe_config_t* config, int hidden_dim)
{
    (void)n_layers; (void)config; (void)hidden_dim;
}

void tq_moe_cache_free(void)
{
}

/* ============================================================
 * Accelerate/cblas FP32 Expert LRU Cache
 *
 * On Apple Silicon, cblas_sgemv leverages the AMX coprocessor which is
 * ~36x faster than manual FP32 dot and ~10x faster than NEON for sgemv.
 * The strategy: dequant IQ2_XXS -> FP32 once (on cache miss), then
 * use cblas_sgemv for all subsequent matmuls (cache hit = near-free).
 *
 * Memory per cached expert: 3 matrices x (inter*dim) x 4 bytes
 *   For 512x2048: 3 x 1M x 4 = 12 MB per expert
 *   32 cache slots: 384 MB total (fits comfortably in 16GB+)
 *
 * Cache hit: cblas only = ~0.019 ms per matmul (36x faster)
 * Cache miss: dequant + store + cblas ~ 1-2 ms (amortized quickly)
 * ============================================================ */

#ifdef TQ_HAS_ACCELERATE

#define CBLAS_CACHE_SIZE 2   /* per layer — 2 × 12 MB × 40 layers = 0.96 GB max */

typedef struct {
    int      expert_id;       /* -1 = empty slot */
    float*   gate_fp32;       /* [expert_dim x hidden_dim] row-major */
    float*   up_fp32;         /* [expert_dim x hidden_dim] row-major */
    float*   down_fp32;       /* [hidden_dim x expert_dim] row-major */
    int      last_used;       /* token counter for LRU eviction */
} cblas_cache_entry_t;

typedef struct {
    cblas_cache_entry_t entries[CBLAS_CACHE_SIZE];
    int count;
} cblas_layer_cache_t;

static cblas_layer_cache_t* g_cblas_cache     = NULL; /* [n_layers] */
static int                  g_cblas_n_layers   = 0;
static int                  g_cblas_hidden_dim = 0;
static int                  g_cblas_exp_inter  = 0;
static int                  g_cblas_token      = 0;
static float*               g_cblas_fp32_temp  = NULL; /* reusable dequant buffer */

void tq_moe_cblas_cache_init(int n_layers, const tq_moe_config_t* config, int hidden_dim)
{
    if (g_cblas_cache) return; /* already initialized */
    if (!config) return;

    g_cblas_n_layers   = n_layers;
    g_cblas_hidden_dim = hidden_dim;
    g_cblas_exp_inter  = config->expert_intermediate_dim;
    g_cblas_token      = 0;

    g_cblas_cache = (cblas_layer_cache_t*)calloc(
        (size_t)n_layers, sizeof(cblas_layer_cache_t));
    if (!g_cblas_cache) {
        fprintf(stderr, "tq_moe_cblas_cache_init: allocation failed\n");
        return;
    }

    for (int l = 0; l < n_layers; l++) {
        for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
            g_cblas_cache[l].entries[s].expert_id = -1;
        }
    }

    /* Reusable FP32 temp buffer for dequantization */
    size_t max_elems = (size_t)g_cblas_exp_inter * hidden_dim;
    size_t down_elems = (size_t)hidden_dim * g_cblas_exp_inter;
    if (down_elems > max_elems) max_elems = down_elems;
    g_cblas_fp32_temp = (float*)malloc(max_elems * sizeof(float));

    float cache_mb = (float)(n_layers * CBLAS_CACHE_SIZE) *
                     (3.0f * (float)((size_t)g_cblas_exp_inter * hidden_dim) * 4.0f) /
                     (1024.0f * 1024.0f);
    fprintf(stderr, "tq_moe_cblas_cache_init: FP32/AMX LRU cache for %d layers x %d slots "
            "(max %.0f MB)\n", n_layers, CBLAS_CACHE_SIZE, (double)cache_mb);
}

static void cblas_free_entry(cblas_cache_entry_t* e)
{
    free(e->gate_fp32);  e->gate_fp32 = NULL;
    free(e->up_fp32);    e->up_fp32 = NULL;
    free(e->down_fp32);  e->down_fp32 = NULL;
    e->expert_id = -1;
}

void tq_moe_cblas_cache_free(void)
{
    if (!g_cblas_cache) return;
    for (int l = 0; l < g_cblas_n_layers; l++) {
        for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
            cblas_free_entry(&g_cblas_cache[l].entries[s]);
        }
    }
    free(g_cblas_cache);
    g_cblas_cache = NULL;
    free(g_cblas_fp32_temp);
    g_cblas_fp32_temp = NULL;
    g_cblas_n_layers = 0;
}

/* Find cached FP32 entry or evict LRU and dequant from IQ2_XXS */
static cblas_cache_entry_t* cblas_cache_get_or_create(
    int layer_idx, int expert_id, const tq_expert_weights_t* exp)
{
    cblas_layer_cache_t* lc = &g_cblas_cache[layer_idx];

    /* Cache hit */
    for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
        if (lc->entries[s].expert_id == expert_id) {
            lc->entries[s].last_used = g_cblas_token;
            return &lc->entries[s];
        }
    }

    /* Cache miss: find empty slot or evict LRU */
    int target = -1;
    if (lc->count < CBLAS_CACHE_SIZE) {
        for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
            if (lc->entries[s].expert_id < 0) {
                target = s;
                break;
            }
        }
        lc->count++;
    } else {
        int oldest_time = g_cblas_token + 1;
        for (int s = 0; s < CBLAS_CACHE_SIZE; s++) {
            if (lc->entries[s].last_used < oldest_time) {
                oldest_time = lc->entries[s].last_used;
                target = s;
            }
        }
        cblas_free_entry(&lc->entries[target]);
    }

    cblas_cache_entry_t* ce = &lc->entries[target];
    ce->expert_id = expert_id;
    ce->last_used = g_cblas_token;

    int dim   = g_cblas_hidden_dim;
    int inter = g_cblas_exp_inter;

    /* Dequant gate: [inter, dim] -> FP32 */
    {
        int n = inter * dim;
        ce->gate_fp32 = (float*)malloc((size_t)n * sizeof(float));
        if (ce->gate_fp32) {
            tq_dequant_row_gguf(exp->gate_type, exp->w_gate, ce->gate_fp32, n);
        }
    }

    /* Dequant up: [inter, dim] -> FP32 */
    {
        int n = inter * dim;
        ce->up_fp32 = (float*)malloc((size_t)n * sizeof(float));
        if (ce->up_fp32) {
            tq_dequant_row_gguf(exp->up_type, exp->w_up, ce->up_fp32, n);
        }
    }

    /* Dequant down: [dim, inter] -> FP32 */
    {
        int n = dim * inter;
        ce->down_fp32 = (float*)malloc((size_t)n * sizeof(float));
        if (ce->down_fp32) {
            tq_dequant_row_gguf(exp->down_type, exp->w_down, ce->down_fp32, n);
        }
    }

    return ce;
}

#endif /* TQ_HAS_ACCELERATE */

/* ============================================================
 * State management
 * ============================================================ */

tq_moe_state_t* tq_moe_create_state(const tq_moe_config_t* config, int hidden_dim)
{
    tq_moe_state_t* s = (tq_moe_state_t*)calloc(1, sizeof(tq_moe_state_t));
    if (!s) return NULL;

    s->router_logits  = (float*)malloc((size_t)config->num_experts * sizeof(float));
    s->top_experts    = (int*)calloc((size_t)config->num_active, sizeof(int));
    s->expert_weights = (float*)malloc((size_t)config->num_active * sizeof(float));
    s->expert_out     = (float*)malloc((size_t)hidden_dim * sizeof(float));

    /* Workspace buffers sized to the larger of expert / shared-expert intermediate dim */
    int inter = config->expert_intermediate_dim;
    if (config->has_shared_expert && config->shared_expert_intermediate_dim > inter)
        inter = config->shared_expert_intermediate_dim;

    s->expert_hb  = (float*)malloc((size_t)inter * sizeof(float));
    s->expert_hb2 = (float*)malloc((size_t)inter * sizeof(float));

    /* Cross-expert parallel scratch pool: per-active-expert slot of
     * (2*expert_dim + hidden_dim) floats. Up to num_active concurrent. */
    size_t per_slot = (size_t)(2 * config->expert_intermediate_dim + hidden_dim);
    s->expert_scratch_pool = malloc((size_t)config->num_active * per_slot * sizeof(float));

    return s;
}

void tq_moe_free_state(tq_moe_state_t* state)
{
    if (!state) return;
    free(state->router_logits);
    free(state->top_experts);
    free(state->expert_weights);
    free(state->expert_out);
    free(state->expert_hb);
    free(state->expert_hb2);
    free(state->expert_scratch_pool);
    free(state);
}

/* ============================================================
 * Top-K expert routing
 * ============================================================ */

void tq_moe_route(const float* hidden, const float* router_weight,
                  int num_experts, int num_active, int hidden_dim,
                  int* out_expert_ids, float* out_expert_weights)
{
    /* Thread-local scratch — avoids malloc on hot path (called per layer per
     * token). 256 experts × 4 bytes + 256 used flags = 1.25 KB per thread. */
    static __thread float   tls_logits[512];   /* upper bound on num_experts */
    static __thread uint8_t tls_used[512];
    float*   logits = (num_experts <= 512) ? tls_logits
                       : (float*)malloc((size_t)num_experts * sizeof(float));
    uint8_t* used   = (num_experts <= 512) ? tls_used
                       : (uint8_t*)calloc((size_t)num_experts, sizeof(uint8_t));
    if (!logits || !used) {
        if (logits && logits != tls_logits) free(logits);
        if (used && used != tls_used) free(used);
        return;
    }
    if (num_experts <= 512) memset(used, 0, (size_t)num_experts);

    /* Step 1: Compute router logits — logits[e] = dot(hidden, router_weight[e])
     * NEON-vectorized: 4 FMA pipes × 4 lanes = 16 floats/cycle peak.
     * Per-token cost on Qwen3.6 (256×2048): ~1ms scalar → ~0.1ms NEON. */
    for (int e = 0; e < num_experts; e++) {
        const float* row = router_weight + (size_t)e * hidden_dim;
#if TQ_MOE_HAS_NEON
        float32x4_t a0 = vdupq_n_f32(0.f);
        float32x4_t a1 = vdupq_n_f32(0.f);
        float32x4_t a2 = vdupq_n_f32(0.f);
        float32x4_t a3 = vdupq_n_f32(0.f);
        int j = 0;
        for (; j + 15 < hidden_dim; j += 16) {
            a0 = vfmaq_f32(a0, vld1q_f32(hidden + j),      vld1q_f32(row + j));
            a1 = vfmaq_f32(a1, vld1q_f32(hidden + j + 4),  vld1q_f32(row + j + 4));
            a2 = vfmaq_f32(a2, vld1q_f32(hidden + j + 8),  vld1q_f32(row + j + 8));
            a3 = vfmaq_f32(a3, vld1q_f32(hidden + j + 12), vld1q_f32(row + j + 12));
        }
        float32x4_t s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
        float sum = vaddvq_f32(s);
        for (; j < hidden_dim; j++) sum += hidden[j] * row[j];
#else
        float sum = 0.0f;
        for (int j = 0; j < hidden_dim; j++)
            sum += hidden[j] * row[j];
#endif
        logits[e] = sum;
    }

    /* Step 2: Top-K selection via partial sort (K passes, K << num_experts)
     *
     * Use >= for tie-breaking so that when multiple experts have equal logits,
     * the first unused one always wins. Also guard against NaN logits (NaN
     * comparisons return false, so without >= the loop could leave best == -1).
     */

    int n_valid = 0;
    for (int k = 0; k < num_active; k++) {
        int best = -1;
        float best_val = -HUGE_VALF;
        for (int e = 0; e < num_experts; e++) {
            if (!used[e] && logits[e] >= best_val) {
                best_val = logits[e];
                best = e;
            }
        }
        out_expert_ids[k] = best;
        if (best >= 0) {
            used[best] = 1;
            n_valid++;
        } else {
            out_expert_weights[k] = 0.0f;
        }
    }

    /* Step 3: Softmax over selected experts (renormalize top-K) */
    if (n_valid == 0) {
        /* All experts invalid (NaN logits or num_experts=0) — uniform fallback */
        for (int k = 0; k < num_active; k++) {
            out_expert_weights[k] = 0.0f;
        }
        if (used != tls_used) free(used);
        if (logits != tls_logits) free(logits);
        return;
    }

    float max_val = -HUGE_VALF;
    for (int k = 0; k < num_active; k++) {
        if (out_expert_ids[k] < 0) continue;
        float v = logits[out_expert_ids[k]];
        if (v > max_val) max_val = v;
    }

    /* Optional softmax temperature (TQ_MOE_ROUTE_TEMP). T>1 spreads the
     * top-K distribution (less peaky); T<1 sharpens. Read once at first
     * call to avoid env parsing on hot path. */
    static float route_temp = 0.0f;
    if (route_temp == 0.0f) {
        const char* s = getenv("TQ_MOE_ROUTE_TEMP");
        route_temp = (s && atof(s) > 0.0f) ? (float)atof(s) : 1.0f;
    }
    float inv_temp = 1.0f / route_temp;

    float sum_exp = 0.0f;
    for (int k = 0; k < num_active; k++) {
        if (out_expert_ids[k] < 0) { out_expert_weights[k] = 0.0f; continue; }
        float e = expf((logits[out_expert_ids[k]] - max_val) * inv_temp);
        out_expert_weights[k] = e;
        sum_exp += e;
    }

    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < num_active; k++)
            out_expert_weights[k] *= inv_sum;
    }

    if (used != tls_used) free(used);
    if (logits != tls_logits) free(logits);
}

/* ============================================================
 * Cross-expert parallel worker: one thread handles one expert entirely
 * (gate_up matmul → activation → down matmul). Inner matmuls run
 * single-threaded via tq_tls_force_serial_matmul to avoid nested pool.
 * ============================================================ */
typedef struct {
    tq_moe_state_t* state;
    const tq_moe_layer_t* layer;
    const tq_moe_config_t* config;
    const float* input;
    int k;
    int expert_dim;
    int hidden_dim;
    void (*activation_fn)(float* restrict, const float* restrict, int);
    float* scratch_hb;
    float* scratch_hb2;
    float* scratch_out;
} expert_parallel_task_t;

extern __thread int tq_tls_force_serial_matmul;

void* expert_parallel_worker(void* arg) {
    expert_parallel_task_t* t = (expert_parallel_task_t*)arg;
    int eid = t->state->top_experts[t->k];
    if (eid < 0 || eid >= t->config->num_experts) {
        memset(t->scratch_out, 0, (size_t)t->hidden_dim * sizeof(float));
        return NULL;
    }
    const tq_expert_weights_t* exp = &t->layer->experts[eid];

    /* Force inner matmuls to run single-threaded */
    int prev_flag = tq_tls_force_serial_matmul;
    tq_tls_force_serial_matmul = 1;

    /* Gate+up fused if contiguous, else separate */
    if (exp->gate_type == exp->up_type &&
        (const uint8_t*)exp->w_up == (const uint8_t*)exp->w_gate +
            tq_ggml_type_size(exp->gate_type) *
            ((size_t)t->expert_dim * t->hidden_dim / tq_ggml_type_blck(exp->gate_type))) {
        float fused_out[16384];
        if (2 * t->expert_dim <= 16384) {
            tq_matmul_gguf(fused_out, t->input, exp->w_gate, exp->gate_type,
                           2 * t->expert_dim, t->hidden_dim);
            memcpy(t->scratch_hb,  fused_out,                 (size_t)t->expert_dim * sizeof(float));
            memcpy(t->scratch_hb2, fused_out + t->expert_dim, (size_t)t->expert_dim * sizeof(float));
        } else {
            tq_matmul_gguf(t->scratch_hb,  t->input, exp->w_gate, exp->gate_type,
                           t->expert_dim, t->hidden_dim);
            tq_matmul_gguf(t->scratch_hb2, t->input, exp->w_up, exp->up_type,
                           t->expert_dim, t->hidden_dim);
        }
    } else {
        tq_matmul_gguf(t->scratch_hb,  t->input, exp->w_gate, exp->gate_type,
                       t->expert_dim, t->hidden_dim);
        tq_matmul_gguf(t->scratch_hb2, t->input, exp->w_up, exp->up_type,
                       t->expert_dim, t->hidden_dim);
    }

    /* Activation: GeGLU or SwiGLU */
    t->activation_fn(t->scratch_hb, t->scratch_hb2, t->expert_dim);

    /* Down projection */
    tq_matmul_gguf(t->scratch_out, t->scratch_hb, exp->w_down, exp->down_type,
                   t->hidden_dim, t->expert_dim);

    tq_tls_force_serial_matmul = prev_flag;
    return NULL;
}

/* ============================================================
 * Full MoE FFN forward pass
 * ============================================================ */

/* Forward decl for self-test wrapper */
void tq_moe_forward_batch(const tq_moe_layer_t* layer,
                          const tq_moe_config_t* config,
                          tq_moe_state_t* state,
                          const float* hidden, float* output,
                          int N, int hidden_dim, int layer_idx);

void tq_moe_forward(const tq_moe_layer_t* layer,
                    const tq_moe_config_t* config,
                    tq_moe_state_t* state,
                    const float* input, float* output,
                    int hidden_dim, int layer_idx)
{
    /* TQ_MOE_BATCH_SELFTEST=1: route this single-token call through the
     * batched path (N=1). Activates the new code on every MoE invocation
     * during decode/prefill for correctness validation. Adds modest
     * overhead but ensures regression picks up any divergence. */
    static int selftest_checked = 0;
    static int selftest_enabled = 0;
    if (!selftest_checked) {
        selftest_enabled = getenv("TQ_MOE_BATCH_SELFTEST") != NULL;
        selftest_checked = 1;
    }
    if (selftest_enabled && !state->routing_precomputed) {
        /* Must NOT recurse — clear the flag while calling the batched path. */
        static __thread int in_selftest = 0;
        if (!in_selftest) {
            in_selftest = 1;
            memset(output, 0, (size_t)hidden_dim * sizeof(float));
            tq_moe_forward_batch(layer, config, state, input, output,
                                 1, hidden_dim, layer_idx);
            in_selftest = 0;
            return;
        }
    }

    int num_active = config->num_active;
    int expert_dim = config->expert_intermediate_dim;

    /* Select activation: GeGLU for Gemma 4, SwiGLU for others */
    void (*activation_fn)(float* restrict, const float* restrict, int) =
        config->use_gelu ? geglu_fused : swiglu_fused;

    /* Step 1: Route — select top-K experts.
     * If routing was precomputed externally (Gemma 4 dual-FFN path),
     * skip the routing step and use the pre-filled top_experts/expert_weights. */
    if (!state->routing_precomputed) {
        const float* route_input = input;
        float scaled_input_buf[4096]; /* stack buffer for scaled input */
        if (layer->router_input_scale && hidden_dim <= 4096) {
            float inv_sqrt_dim = 1.0f / sqrtf((float)hidden_dim);
            for (int i = 0; i < hidden_dim; i++)
                scaled_input_buf[i] = input[i] * inv_sqrt_dim * layer->router_input_scale[i];
            route_input = scaled_input_buf;
        }
        tq_moe_route(route_input, layer->router_weight,
                     config->num_experts, num_active, hidden_dim,
                     state->top_experts, state->expert_weights);
    }
    state->routing_precomputed = 0; /* reset for next call */

    /* Probe: TQ_MOE_PROBE=pos1,pos2,... prints per-layer top-K expert IDs
     * and routing weights. Use the layer-0 call counter to gate since this
     * is called once per MoE layer per token. */
    {
        static __thread int _moe_call_count = 0;
        if (layer_idx == 0) _moe_call_count++;
        const char* _probe = getenv("TQ_MOE_PROBE");
        if (_probe) {
            int match = 0;
            const char* p = _probe;
            while (*p) {
                int v = atoi(p);
                if (v == _moe_call_count) { match = 1; break; }
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
            if (match) {
                fprintf(stderr, "[moe-probe] call=%d L%d experts=[",
                        _moe_call_count, layer_idx);
                for (int k = 0; k < num_active; k++)
                    fprintf(stderr, "%d%s", state->top_experts[k],
                            k+1<num_active?",":"");
                fprintf(stderr, "] weights=[");
                for (int k = 0; k < num_active; k++)
                    fprintf(stderr, "%.3f%s", state->expert_weights[k],
                            k+1<num_active?",":"");
                fprintf(stderr, "]\n");
            }
        }
    }

    /* Step 2: Zero the output accumulator */
    memset(output, 0, (size_t)hidden_dim * sizeof(float));

    /* Kahan compensation buffer for MoE aggregation. Opt-in via
     * TQ_MOE_KAHAN=1. R48 round 3 empirically tested on Qwen3.6-35B:
     * 126 tok vs 147 tok baseline (NOT an improvement; catastrophic
     * cancellation hypothesis RULED OUT for this model). Kept as opt-in
     * for future investigation or different MoE architectures. */
    float moe_kahan_comp[4096] = {0};
    int use_kahan = (hidden_dim <= 4096) && (getenv("TQ_MOE_KAHAN") != NULL);

#ifdef TQ_HAS_ACCELERATE
    g_cblas_token++;
#endif

    /* Step 2.5: Try fused MoE Metal dispatch (3 dispatches for all experts).
     * This replaces the entire per-expert loop below when available.
     * Requires: all expert weights are same IQ2_XXS type and contiguous in mmap. */
#ifdef TQ_HAS_METAL
    extern int tq_metal_moe_available(void);
    extern int tq_metal_moe_forward(
        const float* input, float* output, float* hb_output,
        const void* weight_base, size_t weight_size,
        const uint64_t* gate_offsets, const uint64_t* up_offsets, const uint64_t* down_offsets,
        const int* active_expert_ids, const float* expert_routing_weights,
        int num_active, int expert_dim, int hidden_dim, int num_experts_total, int weight_type,
        const int* gate_types, const int* up_types, const int* down_types);

    /* Metal MoE: single command buffer with memoryBarrier between phases.
     * Eliminates per-phase waitUntilCompleted overhead. */
    if (tq_metal_moe_available() && num_active > 0) {
        /* Check that all active experts use IQ2_XXS and have valid weights */
        int can_fuse = 1;
        const void* base_ptr = NULL;
        size_t base_size = 0;

        /* Find the lowest and highest addressed weight to determine the
         * weight region span for the zero-copy Metal buffer. */
        uintptr_t min_addr = ~(uintptr_t)0;
        uintptr_t max_addr = 0;

        for (int k = 0; k < num_active; k++) {
            int eid = state->top_experts[k];
            if (eid < 0 || eid >= config->num_experts) { can_fuse = 0; break; }
            const tq_expert_weights_t* exp = &layer->experts[eid];
            if (exp->q4_converted) {
                can_fuse = 0; break;
            }
            /* Accept IQ2_XXS (16) and IQ2_S (22) — UD models use mixed types */
            int gt = exp->gate_type, ut = exp->up_type, dt = exp->down_type;
            int is_iq2 = (gt == TQ_GGML_TYPE_IQ2_XXS || gt == TQ_GGML_TYPE_IQ2_S) &&
                         (ut == TQ_GGML_TYPE_IQ2_XXS || ut == TQ_GGML_TYPE_IQ2_S) &&
                         (dt == TQ_GGML_TYPE_IQ2_XXS || dt == TQ_GGML_TYPE_IQ2_S);
            if (!is_iq2) {
                can_fuse = 0; break;
            }
            if (!exp->w_gate || !exp->w_up || !exp->w_down) { can_fuse = 0; break; }

            /* Track min/max addresses to compute base and size */
            uintptr_t addrs[3] = {
                (uintptr_t)exp->w_gate, (uintptr_t)exp->w_up, (uintptr_t)exp->w_down
            };
            /* gate/up: [expert_dim, hidden_dim], down: [hidden_dim, expert_dim]
             * Byte sizes depend on quant type: IQ2_XXS=66, IQ2_S=82 per 256 elements */
            int gate_blk = (gt == TQ_GGML_TYPE_IQ2_S) ? 82 : 66;
            int up_blk   = (ut == TQ_GGML_TYPE_IQ2_S) ? 82 : 66;
            int down_blk = (dt == TQ_GGML_TYPE_IQ2_S) ? 82 : 66;
            size_t gate_bytes = (size_t)(expert_dim * (hidden_dim / 256)) * gate_blk;
            size_t up_bytes   = (size_t)(expert_dim * (hidden_dim / 256)) * up_blk;
            size_t down_bytes = (size_t)(hidden_dim * (expert_dim / 256)) * down_blk;
            size_t sizes[3] = { gate_bytes, up_bytes, down_bytes };

            for (int i = 0; i < 3; i++) {
                if (addrs[i] < min_addr) min_addr = addrs[i];
                uintptr_t end = addrs[i] + sizes[i];
                if (end > max_addr) max_addr = end;
            }
        }

        if (can_fuse && min_addr < max_addr) {
            base_ptr = (const void*)min_addr;
            base_size = (size_t)(max_addr - min_addr);

            /* Compute byte offsets from base for each active expert */
            uint64_t gate_offs[8], up_offs[8], down_offs[8];
            int expert_ids[8];
            float routing_w[8];
            int per_gate_types[8], per_up_types[8], per_down_types[8];

            for (int k = 0; k < num_active; k++) {
                int eid = state->top_experts[k];
                const tq_expert_weights_t* exp = &layer->experts[eid];
                expert_ids[k] = eid;
                routing_w[k] = state->expert_weights[k];
                gate_offs[k] = (uint64_t)((uintptr_t)exp->w_gate - min_addr);
                up_offs[k]   = (uint64_t)((uintptr_t)exp->w_up   - min_addr);
                down_offs[k] = (uint64_t)((uintptr_t)exp->w_down - min_addr);
                per_gate_types[k] = (int)exp->gate_type;
                per_up_types[k]   = (int)exp->up_type;
                per_down_types[k] = (int)exp->down_type;
            }

            /* Allocate buffer for GPU SwiGLU results [num_active * expert_dim] */
            float* hb_gpu = (float*)malloc((size_t)num_active * (size_t)expert_dim * sizeof(float));
            if (!hb_gpu) goto moe_cpu_fallback;

            int rc = tq_metal_moe_forward(
                input, output, hb_gpu,
                base_ptr, base_size,
                gate_offs, up_offs, down_offs,
                expert_ids, routing_w,
                num_active, expert_dim, hidden_dim,
                config->num_experts,
                0, /* weight_type=0 means use per-expert types */
                per_gate_types, per_up_types, per_down_types);

            if (rc == 1) {
                /* Hybrid: GPU did gate+up+SwiGLU, we do down+accum on CPU */
                for (int k = 0; k < num_active; k++) {
                    int eid = state->top_experts[k];
                    float w = state->expert_weights[k];
                    const tq_expert_weights_t* exp = &layer->experts[eid];
                    float* hb_k = hb_gpu + k * expert_dim;

                    /* down = hb_k @ w_down^T -> [hidden_dim] */
                    tq_matmul_gguf(state->expert_out, hb_k,
                                   exp->w_down, exp->down_type,
                                   hidden_dim, expert_dim);

                    /* Weighted accumulation: output += weight * down_proj */
                    for (int i = 0; i < hidden_dim; i++)
                        output[i] += w * state->expert_out[i];
                }
                free(hb_gpu);
                goto moe_shared_expert;
            }
            free(hb_gpu);
            if (rc == 0) {
                /* Full GPU success (unlikely in hybrid mode, but handle it) */
                goto moe_shared_expert;
            }
            /* else: rc == -1, fall through to per-expert CPU path */
        }
    }
#endif /* TQ_HAS_METAL */

#ifdef TQ_HAS_METAL
moe_cpu_fallback: ;
#endif
    /* ============================================================
     * Cross-expert parallelism: run all N active experts in parallel.
     *
     * Before: outer serial loop over experts, inner tq_matmul_gguf
     *   uses thread pool (8 threads) per matmul. Each expert = ~16 thread
     *   dispatches × ~181 μs overhead = 2.9 ms per expert × 8 experts = 23 ms
     *   of pure overhead per layer × 30 layers = 690 ms per token.
     *
     * After: outer parallel loop (N experts → N threads, each with its own
     *   single-threaded matmul via tq_tls_force_serial_matmul). One barrier
     *   per layer instead of 16 per expert.
     *
     * Requires: GGUF dequant path (not Q4-converted). Each expert is
     *   memory-bound individually, but 8 experts * ~742 KB = 5.9 MB fits
     *   in M1 Pro's shared L2 (12 MB) allowing parallel execution without
     *   thrashing.
     * ============================================================ */
    extern void tq_tp_run(void* (*fn)(void*), void** args, int n_tasks);
    int n_threads = tq_get_threads();
    int parallel_experts = (num_active >= 2 && num_active <= n_threads &&
                             num_active <= TQ_TP_MAX);
    /* Pillar 1.5 R9: TQ_MOE_SERIAL=1 forces serial expert dispatch for
     * determinism testing. Parallel path was suspected source of T=0
     * non-determinism that manifests as Qwen3.6-35B ~70-token decode
     * degradation and long-context garbage. */
    if (getenv("TQ_MOE_SERIAL")) parallel_experts = 0;
    /* Only do cross-expert parallel for GGUF experts (IQ2_XXS etc).
     * Q4-converted fast path has its own parallelism. */
    {
        int first_eid = state->top_experts[0];
        if (first_eid >= 0 && first_eid < config->num_experts &&
            layer->experts[first_eid].q4_converted) {
            parallel_experts = 0;
        }
    }

    if (parallel_experts && state->expert_scratch_pool) {
        /* Parallel path: N threads × 1 expert each */
        #ifndef TQ_TP_MAX
        #define TQ_TP_MAX 16
        #endif
        expert_parallel_task_t tasks[TQ_TP_MAX];
        void* task_ptrs[TQ_TP_MAX];
        float* pool = (float*)state->expert_scratch_pool;
        /* Pool layout per task: [hb | hb2 | out] = 2*expert_dim + hidden_dim floats */
        size_t stride = (size_t)(2 * expert_dim + hidden_dim);
        for (int k = 0; k < num_active; k++) {
            tasks[k].state = state;
            tasks[k].layer = layer;
            tasks[k].config = config;
            tasks[k].input = input;
            tasks[k].k = k;
            tasks[k].expert_dim = expert_dim;
            tasks[k].hidden_dim = hidden_dim;
            tasks[k].activation_fn = activation_fn;
            tasks[k].scratch_hb  = pool + k * stride;
            tasks[k].scratch_hb2 = pool + k * stride + expert_dim;
            tasks[k].scratch_out = pool + k * stride + 2 * expert_dim;
            task_ptrs[k] = &tasks[k];
        }

        tq_tp_run(expert_parallel_worker, task_ptrs, num_active);

        /* Accumulate results. R59: NEON-vectorized — 4× FMAs per iter,
         * small but clean win since this runs serially in the main thread
         * after the parallel barrier (320 calls × 2048 floats/tok = 655K FMAs). */
        for (int k = 0; k < num_active; k++) {
            int eid = state->top_experts[k];
            if (eid < 0 || eid >= config->num_experts) continue;
            float w = state->expert_weights[k];
            float exp_scale = (layer->expert_scale) ? layer->expert_scale[eid] : 1.0f;
            float ws = w * exp_scale;
            float* eout = tasks[k].scratch_out;
            if (use_kahan) {
                /* Scalar Kahan for parallel path — NEON FMA can't easily
                 * track the compensation term, so fall back to scalar
                 * when Kahan is enabled. */
                for (int i = 0; i < hidden_dim; i++) {
                    float y = ws * eout[i] - moe_kahan_comp[i];
                    float t = output[i] + y;
                    moe_kahan_comp[i] = (t - output[i]) - y;
                    output[i] = t;
                }
            } else {
                int i = 0;
#ifdef __ARM_NEON
                float32x4_t vws = vdupq_n_f32(ws);
                for (; i + 3 < hidden_dim; i += 4) {
                    float32x4_t vo = vld1q_f32(output + i);
                    float32x4_t ve = vld1q_f32(eout + i);
                    vo = vmlaq_f32(vo, ve, vws);
                    vst1q_f32(output + i, vo);
                }
#endif
                for (; i < hidden_dim; i++)
                    output[i] += ws * eout[i];
            }
        }
        goto moe_shared_expert;
    }

    /* Serial fallback (original code path) */
    /* Step 3: For each selected expert, compute GeGLU/SwiGLU FFN and accumulate.
     * Prefetch next expert's weights to hide memory latency for mmap'd IQ3_XXS/IQ4_NL. */
    for (int k = 0; k < num_active; k++) {
        int eid = state->top_experts[k];

        /* Prefetch next expert's gate weight (first cache line) */
        if (k + 1 < num_active) {
            int next_eid = state->top_experts[k + 1];
            if (next_eid >= 0 && next_eid < config->num_experts) {
                const tq_expert_weights_t* next_exp = &layer->experts[next_eid];
                if (next_exp->w_gate) __builtin_prefetch(next_exp->w_gate, 0, 1);
                if (next_exp->w_down) __builtin_prefetch(next_exp->w_down, 0, 1);
            }
        }
        float w = state->expert_weights[k];
        if (eid < 0 || eid >= config->num_experts) continue; /* safety check */
        const tq_expert_weights_t* exp = &layer->experts[eid];

#ifdef TQ_HAS_ACCELERATE
        /* PRIMARY path on Apple: cblas_sgemv via AMX coprocessor.
         * Dequant IQ2_XXS -> FP32 once (LRU cached), then cblas_sgemv.
         * AMX is ~36x faster than manual FP32 dot for expert-sized matrices. */
        /* cblas/AMX DISABLED: IQ2→FP32 dequant cost dominates.
         * With 256 experts/layer, cache miss rate is too high.
         * Fused IQ2 dot (no dequant) is faster overall.
         * cblas would win if we could pre-dequant ALL experts at load time,
         * but 30 layers × 256 experts × 12 MB = 90 GB — impossible. */
        if (0 && g_cblas_cache && layer_idx >= 0 && layer_idx < g_cblas_n_layers
            && exp->w_gate && !exp->q4_converted) {
            cblas_cache_entry_t* ce = cblas_cache_get_or_create(layer_idx, eid, exp);
            if (ce && ce->gate_fp32 && ce->up_fp32 && ce->down_fp32) {
                /* gate: input[hidden_dim] @ gate[expert_dim, hidden_dim]^T -> hb[expert_dim]
                 * cblas_sgemv(RowMajor, NoTrans, M, N, alpha, A, lda, x, incx, beta, y, incy)
                 * A is [M x N] = [expert_dim x hidden_dim], x is [hidden_dim], y is [expert_dim] */
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            expert_dim, hidden_dim, 1.0f,
                            ce->gate_fp32, hidden_dim,
                            input, 1, 0.0f, state->expert_hb, 1);

                /* up: same layout */
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            expert_dim, hidden_dim, 1.0f,
                            ce->up_fp32, hidden_dim,
                            input, 1, 0.0f, state->expert_hb2, 1);

                /* Gated activation: GeGLU (Gemma4) or SwiGLU (Qwen) */
                activation_fn(state->expert_hb, state->expert_hb2, expert_dim);

                /* down: hb[expert_dim] @ down[hidden_dim, expert_dim]^T -> out[hidden_dim]
                 * A is [hidden_dim x expert_dim], x is [expert_dim], y is [hidden_dim] */
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            hidden_dim, expert_dim, 1.0f,
                            ce->down_fp32, expert_dim,
                            state->expert_hb, 1, 0.0f, state->expert_out, 1);

                /* Weighted accumulation: output += weight * down_proj */
                for (int i = 0; i < hidden_dim; i++)
                    output[i] += w * state->expert_out[i];
                continue;
            }
        }
#endif /* TQ_HAS_ACCELERATE */

        /* Historical note: a Q8 LRU cache for IQ2 experts (IQ2→FP32→Q8
         * on miss, fused_dot_q8_0 on hit) was prototyped here but
         * empirically lost to direct fused_dot_iq2_xxs_neon — the
         * cache miss conversion cost exceeds the direct-path compute
         * cost when expert reuse rate is low. Round 13 (2026-04-19)
         * removed the dead dispatch site. See git history of this
         * file for the prior implementation. */

        if (exp->q4_converted) {
            /* Fast Q4 matmul path — pre-converted expert weights (shared expert)
             * tq_matmul_q4(out, x, w_qs, w_scales, n=out_rows, d=in_cols) */
            tq_matmul_q4(state->expert_hb, input,
                         exp->gate_q4_qs, exp->gate_q4_scales,
                         expert_dim, hidden_dim);
            tq_matmul_q4(state->expert_hb2, input,
                         exp->up_q4_qs, exp->up_q4_scales,
                         expert_dim, hidden_dim);

            /* Gated activation: GeGLU (Gemma4) or SwiGLU (Qwen) */
            activation_fn(state->expert_hb, state->expert_hb2, expert_dim);

            tq_matmul_q4(state->expert_out, state->expert_hb,
                         exp->down_q4_qs, exp->down_q4_scales,
                         hidden_dim, expert_dim);
        } else {
            /* On-the-fly GGUF dequant path (IQ2_XXS/IQ2_S/IQ4_NL etc).
             * For fused gate_up_exps (Gemma 4), gate and up weights are
             * CONTIGUOUS: w_gate points to first expert_dim rows, w_up to
             * the next. One matmul with 2×expert_dim rows saves a dispatch. */
            if (exp->gate_type == exp->up_type &&
                (const uint8_t*)exp->w_up == (const uint8_t*)exp->w_gate +
                    tq_ggml_type_size(exp->gate_type) *
                    ((size_t)expert_dim * hidden_dim / tq_ggml_type_blck(exp->gate_type))) {
                /* Fused gate+up: single matmul, output is [gate | up].
                 * Stack buffer for the 2× output (max 16K floats = 64KB). */
                float fused_out[16384];
                if (2 * expert_dim <= 16384) {
                    tq_matmul_gguf(fused_out, input,
                                   exp->w_gate, exp->gate_type,
                                   2 * expert_dim, hidden_dim);
                    memcpy(state->expert_hb,  fused_out,
                           (size_t)expert_dim * sizeof(float));
                    memcpy(state->expert_hb2, fused_out + expert_dim,
                           (size_t)expert_dim * sizeof(float));
                } else {
                    /* Fallback: too large for stack */
                    tq_matmul_gguf(state->expert_hb, input,
                                   exp->w_gate, exp->gate_type,
                                   expert_dim, hidden_dim);
                    tq_matmul_gguf(state->expert_hb2, input,
                                   exp->w_up, exp->up_type,
                                   expert_dim, hidden_dim);
                }
            } else {
                /* Separate gate and up matmuls */
                tq_matmul_gguf(state->expert_hb, input,
                               exp->w_gate, exp->gate_type,
                               expert_dim, hidden_dim);
                tq_matmul_gguf(state->expert_hb2, input,
                               exp->w_up, exp->up_type,
                               expert_dim, hidden_dim);
            }

            /* Gated activation: GeGLU (Gemma4) or SwiGLU (Qwen) */
            activation_fn(state->expert_hb, state->expert_hb2, expert_dim);

            /* down = hb @ w_down^T   -> [hidden_dim] */
            tq_matmul_gguf(state->expert_out, state->expert_hb,
                           exp->w_down, exp->down_type,
                           hidden_dim, expert_dim);
        }

        /* Weighted accumulation: output += weight * scale * down_proj
         * Gemma 4 has per-expert output scaling (ffn_down_exps.scale). */
        float exp_scale = (layer->expert_scale) ? layer->expert_scale[eid] : 1.0f;
        float ws = w * exp_scale;
        if (use_kahan) {
            for (int i = 0; i < hidden_dim; i++) {
                float y = ws * state->expert_out[i] - moe_kahan_comp[i];
                float t = output[i] + y;
                moe_kahan_comp[i] = (t - output[i]) - y;
                output[i] = t;
            }
        } else {
            for (int i = 0; i < hidden_dim; i++)
                output[i] += ws * state->expert_out[i];
        }
    }

#ifdef TQ_HAS_METAL
moe_shared_expert:
#endif
    /* Step 4: Shared expert (always-active, if present) */
    if (config->has_shared_expert) {
        int shared_dim = config->shared_expert_intermediate_dim;
        if (shared_dim == 0) shared_dim = expert_dim;

        /* Optional shared expert gating (sigmoid scalar gate) */
        float shared_gate_val = 1.0f;
        if (layer->shared_gate) {
            float dot = 0.0f;
            for (int j = 0; j < hidden_dim; j++)
                dot += input[j] * layer->shared_gate[j];
            shared_gate_val = 1.0f / (1.0f + fast_expf_moe(-dot)); /* sigmoid */
        }

        if (layer->shared_expert.q4_converted) {
            /* Fast Q4 path for shared expert
             * tq_matmul_q4(out, x, w_qs, w_scales, n=out_rows, d=in_cols) */
            tq_matmul_q4(state->expert_hb, input,
                         layer->shared_expert.gate_q4_qs, layer->shared_expert.gate_q4_scales,
                         shared_dim, hidden_dim);
            tq_matmul_q4(state->expert_hb2, input,
                         layer->shared_expert.up_q4_qs, layer->shared_expert.up_q4_scales,
                         shared_dim, hidden_dim);

            activation_fn(state->expert_hb, state->expert_hb2, shared_dim);

            tq_matmul_q4(state->expert_out, state->expert_hb,
                         layer->shared_expert.down_q4_qs, layer->shared_expert.down_q4_scales,
                         hidden_dim, shared_dim);
        } else {
            /* Fallback: on-the-fly GGUF dequant */
            /* Gate+up batched by layer-level batch scope in tq_forward() */
            tq_matmul_gguf(state->expert_hb, input,
                           layer->shared_expert.w_gate, layer->shared_expert.gate_type,
                           shared_dim, hidden_dim);

            tq_matmul_gguf(state->expert_hb2, input,
                           layer->shared_expert.w_up, layer->shared_expert.up_type,
                           shared_dim, hidden_dim);

            tq_metal_batch_flush_if_available();

            activation_fn(state->expert_hb, state->expert_hb2, shared_dim);

            tq_matmul_gguf(state->expert_out, state->expert_hb,
                           layer->shared_expert.w_down, layer->shared_expert.down_type,
                           hidden_dim, shared_dim);
            /* Flush w_down before CPU reads expert_out */
            tq_metal_batch_flush_if_available();
        }

        if (use_kahan) {
            for (int i = 0; i < hidden_dim; i++) {
                float y = shared_gate_val * state->expert_out[i] - moe_kahan_comp[i];
                float t = output[i] + y;
                moe_kahan_comp[i] = (t - output[i]) - y;
                output[i] = t;
            }
        } else {
            for (int i = 0; i < hidden_dim; i++)
                output[i] += shared_gate_val * state->expert_out[i];
        }
    }
}

/* ============================================================
 * Mission A Step 3d: Batched MoE forward for prefill.
 *
 * The idea: for N prompt tokens, route each token independently (small
 * cost), then group tokens by expert. For each active expert e with M_e
 * tokens routed to it, do ONE batched gate/up/down matmul across its
 * M_e token subset instead of M_e separate per-token matmuls. The win
 * comes from amortizing expert weight reads (which dominate memory
 * bandwidth for IQ3/IQ4 experts on M1 Pro).
 *
 * Phase 1: Route — tq_moe_route × N
 * Phase 2: Inverse index — tokens_of[e] = [list of (token_idx, slot_k, weight)]
 * Phase 3: Per active expert: gather X subset → batched matmul → SwiGLU
 *          → down batched matmul → scatter weighted accumulate
 *
 * If TQ_MOE_BATCH=1 is not set in the environment, falls back to N ×
 * tq_moe_forward. This is an opt-in optimization.
 *
 * Sanity mode: TQ_MOE_BATCH_SANITY=1 runs BOTH paths and checks
 * max-abs-diff < 1e-3 per layer.
 * ============================================================ */

/* External public batched matmul wrappers (tq_gguf_quants.c) */
extern int tq_batched_matmul_iq3_xxs(float* out, const void* weight,
                                      const float* x, int out_dim, int in_dim, int N);
extern int tq_batched_matmul_iq3_s(float* out, const void* weight,
                                    const float* x, int out_dim, int in_dim, int N);
extern int tq_batched_matmul_iq4_xs(float* out, const void* weight,
                                     const float* x, int out_dim, int in_dim, int N);
extern int tq_batched_matmul_q3_k(float* out, const void* weight,
                                   const float* x, int out_dim, int in_dim, int N);
extern void tq_batched_matmul_q8_0(float* out, const void* w_blocks,
                                    const float* x, int n_rows, int d, int N);

/* Dispatch ONE batched matmul for the [M_e, out_dim] = X[M_e, in_dim] @ W.
 * Returns 0 on success, -1 if no batched kernel available (caller falls
 * back to M_e × tq_matmul_gguf). */
static int moe_batched_dispatch(
    float* out, const tq_expert_weights_t* exp,
    int which, /* 0=gate, 1=up, 2=down */
    const float* x, int out_dim, int in_dim, int M_e)
{
    const void* w = NULL;
    int wt = 0;
    if (which == 0) { w = exp->w_gate;  wt = (int)exp->gate_type; }
    else if (which == 1) { w = exp->w_up;    wt = (int)exp->up_type; }
    else { w = exp->w_down;  wt = (int)exp->down_type; }

    if (exp->q4_converted && which != 2) {
        /* Q4 converted gate/up */
        const uint8_t* qs = (which == 0) ? exp->gate_q4_qs : exp->up_q4_qs;
        const float*   sc = (which == 0) ? exp->gate_q4_scales : exp->up_q4_scales;
        if (qs && sc) {
            tq_batched_matmul_q4(out, qs, sc, x, out_dim, in_dim, M_e, NULL);
            return 0;
        }
    }
    if (exp->q4_converted && which == 2) {
        if (exp->down_q4_qs && exp->down_q4_scales) {
            tq_batched_matmul_q4(out, exp->down_q4_qs, exp->down_q4_scales,
                                 x, out_dim, in_dim, M_e, NULL);
            return 0;
        }
    }

    if (!w) return -1;
    if (M_e > 64) return -1; /* kernel safety cap */
    if (in_dim % 256 != 0) return -1;

    /* Batched kernel dispatch for types with proven-equivalent NEON kernels.
     * Sanity: all 4 Qwen3.6 tiers (IQ2_XXS/IQ3_XXS/IQ4_XS/Q3_K_S) pass
     * max-abs-diff < 1e-7 vs per-token tq_moe_forward. */
    if (wt == TQ_GGML_TYPE_IQ3_XXS) {
        return tq_batched_matmul_iq3_xxs(out, w, x, out_dim, in_dim, M_e);
    } else if (wt == TQ_GGML_TYPE_IQ3_S) {
        return tq_batched_matmul_iq3_s(out, w, x, out_dim, in_dim, M_e);
    } else if (wt == TQ_GGML_TYPE_IQ4_XS) {
        return tq_batched_matmul_iq4_xs(out, w, x, out_dim, in_dim, M_e);
    } else if (wt == TQ_GGML_TYPE_Q3_K) {
        return tq_batched_matmul_q3_k(out, w, x, out_dim, in_dim, M_e);
    } else if (wt == TQ_GGML_TYPE_Q8_0) {
        tq_batched_matmul_q8_0(out, w, x, out_dim, in_dim, M_e);
        return 0;
    }
    return -1;
}

/* Per-token fallback: used when no batched kernel for this expert's type. */
static void moe_per_token_fallback(
    const tq_moe_layer_t* layer,
    const tq_moe_config_t* config,
    tq_moe_state_t* state,
    const float* hidden, float* output,
    int N, int hidden_dim, int layer_idx,
    int* top_experts_all, float* expert_weights_all)
{
    int num_active = config->num_active;
    for (int n = 0; n < N; n++) {
        /* Copy precomputed routing into state and set flag */
        memcpy(state->top_experts, top_experts_all + n * num_active,
               (size_t)num_active * sizeof(int));
        memcpy(state->expert_weights, expert_weights_all + n * num_active,
               (size_t)num_active * sizeof(float));
        state->routing_precomputed = 1;

        float* out_row = output + (size_t)n * hidden_dim;
        /* tq_moe_forward zeros output internally, so this is fine. */
        tq_moe_forward(layer, config, state, hidden + (size_t)n * hidden_dim,
                       out_row, hidden_dim, layer_idx);
    }
}

/* ============================================================
 * Mission A Step 3f: cross-expert parallel worker for tq_moe_forward_batch.
 *
 * Each task = ONE expert's entire [M_e] token batch: gather → batched
 * gate/up matmul → SwiGLU/GeGLU → batched down matmul → scatter into a
 * private per-worker output buffer. After all workers finish, the main
 * thread serially reduces per-worker outputs into the shared output.
 *
 * Why private output buffers? Multiple experts may share a token (top-K
 * routing with K>1 selects K experts per token), so concurrent scatter
 * into output[tok] would race. Private accumulators eliminate the race
 * without atomics. Memory cost: n_threads × N × hidden_dim floats
 * (e.g. 8 × 256 × 2560 × 4 bytes ≈ 20 MB — fits easily).
 *
 * Each worker sets tq_tls_force_serial_matmul=1 to prevent nested pool
 * dispatch inside tq_batched_matmul_* (the outer pool is already saturated
 * by the expert-level workers).
 * ============================================================ */
typedef struct {
    /* Per-expert inputs (shared read state) */
    const tq_moe_layer_t*       layer;
    const tq_moe_config_t*      config;
    const float*                hidden;          /* [N_total, hidden_dim] */
    int                         N_total;
    int                         hidden_dim;
    int                         expert_dim;
    int                         eid;             /* global expert id */
    int                         M_e;             /* tokens routed to this expert */
    const int*                  tokens_idx;      /* [M_e] token indices */
    const float*                tokens_w;        /* [M_e] routing weights */
    float                       exp_scale;       /* per-expert output scale (Gemma 4) */
    void (*activation_fn)(float* restrict, const float* restrict, int);
    /* Wave mode: per-task pre-baked pointers (output_private != NULL).
     * Dynamic (Step 3g) mode: output_private == NULL → worker picks its
     * own scratch slab + private output from scratch_all using
     * tq_tls_worker_id. shared_output is the single destination of
     * worker id 0 (replaces its private slab). */
    float*                      X_sub;           /* [64, hidden_dim] */
    float*                      gate_out;        /* [64, expert_dim] */
    float*                      up_out;          /* [64, expert_dim] */
    float*                      down_out;        /* [64, hidden_dim] */
    float*                      output_private;  /* [N_total, hidden_dim] — this worker's accumulator (wave mode) */
    /* Dynamic-mode shared slabs (only read when output_private == NULL) */
    float*                      scratch_all;     /* n_workers × per_worker_total */
    float*                      shared_output;   /* == `output`, used by worker id 0 */
    size_t                      per_worker_total;
    int                         max_M_cap;
    int                         n_workers;       /* bound for TLS id clamp */
} moe_batch_expert_task_t;

static void* moe_batch_expert_worker(void* arg) {
    moe_batch_expert_task_t* t = (moe_batch_expert_task_t*)arg;
    const tq_expert_weights_t* exp = &t->layer->experts[t->eid];
    int hidden_dim = t->hidden_dim;
    int expert_dim = t->expert_dim;
    int M_e = t->M_e;
    if (M_e <= 0) return NULL;

    /* Resolve scratch + output buffers.
     *   Wave mode: t->output_private is pre-baked, use t->X_sub/gate_out/etc.
     *   Dynamic mode (Step 3g): output_private == NULL → pick per-worker
     *   slab by TLS worker id. Worker id 0 writes through to shared_output
     *   (== outer `output`), others use their private slab inside scratch_all. */
    float* X_sub; float* gate_out; float* up_out; float* down_out; float* output_private;
    if (t->output_private) {
        X_sub          = t->X_sub;
        gate_out       = t->gate_out;
        up_out         = t->up_out;
        down_out       = t->down_out;
        output_private = t->output_private;
    } else {
        int wid = tq_tls_worker_id;
        if (wid < 0) wid = 0;
        if (wid >= t->n_workers) wid = t->n_workers - 1;
        int max_M_cap = t->max_M_cap;
        float* base = t->scratch_all + (size_t)wid * t->per_worker_total;
        X_sub    = base + 0;
        gate_out = base + (size_t)max_M_cap * hidden_dim;
        up_out   = base + (size_t)max_M_cap * hidden_dim
                        + (size_t)max_M_cap * expert_dim;
        down_out = base + (size_t)max_M_cap * hidden_dim
                        + 2 * (size_t)max_M_cap * expert_dim;
        size_t per_worker_scratch =
            (size_t)max_M_cap * hidden_dim
          + (size_t)max_M_cap * expert_dim
          + (size_t)max_M_cap * expert_dim
          + (size_t)max_M_cap * hidden_dim;
        output_private = (wid == 0) ? t->shared_output : (base + per_worker_scratch);
    }

    /* Force inner matmuls to run single-threaded (outer pool is busy) */
    int prev_flag = tq_tls_force_serial_matmul;
    tq_tls_force_serial_matmul = 1;

    /* Chunk to fit kernel M<=64 cap */
    int chunks = (M_e + 63) / 64;
    for (int ci = 0; ci < chunks; ci++) {
        int chunk_start = ci * 64;
        int chunk_end = chunk_start + 64;
        if (chunk_end > M_e) chunk_end = M_e;
        int M_c = chunk_end - chunk_start;

        /* Gather X_sub rows for this chunk */
        for (int m = 0; m < M_c; m++) {
            int tok = t->tokens_idx[chunk_start + m];
            memcpy(X_sub + (size_t)m * hidden_dim,
                   t->hidden + (size_t)tok * hidden_dim,
                   (size_t)hidden_dim * sizeof(float));
        }

        /* Gate + up projections */
        int rc_gate = moe_batched_dispatch(gate_out, exp, 0,
                                            X_sub, expert_dim, hidden_dim, M_c);
        int rc_up   = moe_batched_dispatch(up_out, exp, 1,
                                            X_sub, expert_dim, hidden_dim, M_c);
        if (rc_gate != 0 || rc_up != 0) {
            for (int m = 0; m < M_c; m++) {
                tq_matmul_gguf(gate_out + (size_t)m * expert_dim,
                               X_sub + (size_t)m * hidden_dim,
                               exp->w_gate, exp->gate_type, expert_dim, hidden_dim);
                tq_matmul_gguf(up_out + (size_t)m * expert_dim,
                               X_sub + (size_t)m * hidden_dim,
                               exp->w_up, exp->up_type, expert_dim, hidden_dim);
            }
        }

        /* Activation (SwiGLU/GeGLU) per row */
        for (int m = 0; m < M_c; m++) {
            t->activation_fn(gate_out + (size_t)m * expert_dim,
                             up_out + (size_t)m * expert_dim, expert_dim);
        }

        /* Down projection */
        int rc_down = moe_batched_dispatch(down_out, exp, 2,
                                            gate_out, hidden_dim, expert_dim, M_c);
        if (rc_down != 0) {
            for (int m = 0; m < M_c; m++) {
                tq_matmul_gguf(down_out + (size_t)m * hidden_dim,
                               gate_out + (size_t)m * expert_dim,
                               exp->w_down, exp->down_type, hidden_dim, expert_dim);
            }
        }

        /* Scatter weighted accumulate into this worker's PRIVATE output buffer */
        float exp_scale = t->exp_scale;
        for (int m = 0; m < M_c; m++) {
            int tok = t->tokens_idx[chunk_start + m];
            float w = t->tokens_w[chunk_start + m] * exp_scale;
            float* out_row = output_private + (size_t)tok * hidden_dim;
            const float* dout = down_out + (size_t)m * hidden_dim;
#if TQ_MOE_HAS_NEON
            int i = 0;
            float32x4_t vw = vdupq_n_f32(w);
            for (; i + 3 < hidden_dim; i += 4) {
                float32x4_t vo = vld1q_f32(out_row + i);
                float32x4_t vd = vld1q_f32(dout + i);
                vst1q_f32(out_row + i, vfmaq_f32(vo, vw, vd));
            }
            for (; i < hidden_dim; i++) out_row[i] += w * dout[i];
#else
            for (int i = 0; i < hidden_dim; i++) out_row[i] += w * dout[i];
#endif
        }
    }

    tq_tls_force_serial_matmul = prev_flag;
    return NULL;
}

void tq_moe_forward_batch(const tq_moe_layer_t* layer,
                          const tq_moe_config_t* config,
                          tq_moe_state_t* state,
                          const float* hidden, float* output,
                          int N, int hidden_dim, int layer_idx)
{
    if (N <= 0) return;
    int num_active = config->num_active;
    int num_experts = config->num_experts;
    int expert_dim = config->expert_intermediate_dim;

    void (*activation_fn)(float* restrict, const float* restrict, int) =
        config->use_gelu ? geglu_fused : swiglu_fused;

    int debug = (getenv("TQ_DEBUG_MOE_BATCH") != NULL);
    /* Internal sanity check: save ref from per-token path, compare against
     * batched output. Gated by TQ_MOE_BATCH_KERNEL_SANITY to avoid
     * colliding with the outer hybrid driver's TQ_MOE_BATCH_SANITY. */
    int sanity = (getenv("TQ_MOE_BATCH_KERNEL_SANITY") != NULL);

    /* Phase 1: Route all N tokens — N × tq_moe_route.
     * Allocate per-token arrays for top_experts and expert_weights. */
    int* top_experts_all = (int*)malloc((size_t)N * num_active * sizeof(int));
    float* expert_weights_all = (float*)malloc((size_t)N * num_active * sizeof(float));
    if (!top_experts_all || !expert_weights_all) {
        free(top_experts_all); free(expert_weights_all);
        /* Fallback: per-token, no precomputed routing */
        for (int n = 0; n < N; n++) {
            state->routing_precomputed = 0;
            tq_moe_forward(layer, config, state, hidden + (size_t)n * hidden_dim,
                           output + (size_t)n * hidden_dim, hidden_dim, layer_idx);
        }
        return;
    }

    float scaled_input_buf[4096];
    for (int n = 0; n < N; n++) {
        const float* h_n = hidden + (size_t)n * hidden_dim;
        const float* route_input = h_n;
        if (layer->router_input_scale && hidden_dim <= 4096) {
            float inv_sqrt_dim = 1.0f / sqrtf((float)hidden_dim);
            for (int i = 0; i < hidden_dim; i++) {
                scaled_input_buf[i] = h_n[i] * inv_sqrt_dim * layer->router_input_scale[i];
            }
            route_input = scaled_input_buf;
        }
        tq_moe_route(route_input, layer->router_weight,
                     num_experts, num_active, hidden_dim,
                     top_experts_all + n * num_active,
                     expert_weights_all + n * num_active);
    }

    /* Optional sanity mode: save a reference copy from per-token path. */
    float* ref_output = NULL;
    if (sanity) {
        ref_output = (float*)malloc((size_t)N * hidden_dim * sizeof(float));
        if (ref_output) {
            memset(ref_output, 0, (size_t)N * hidden_dim * sizeof(float));
            moe_per_token_fallback(layer, config, state, hidden, ref_output,
                                    N, hidden_dim, layer_idx,
                                    top_experts_all, expert_weights_all);
        }
    }

    /* Phase 2: Inverse index — for each expert e, build list of (token_idx,
     * slot_k, weight). Use compact arrays indexed by active_experts[]. */
    int* tokens_per_expert_count = (int*)calloc((size_t)num_experts, sizeof(int));
    if (!tokens_per_expert_count) {
        free(top_experts_all); free(expert_weights_all); free(ref_output);
        return;
    }
    /* Count pass */
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < num_active; k++) {
            int eid = top_experts_all[n * num_active + k];
            if (eid >= 0 && eid < num_experts) tokens_per_expert_count[eid]++;
        }
    }

    /* Per-expert arrays */
    int** tokens_per_expert_idx = (int**)calloc((size_t)num_experts, sizeof(int*));
    float** tokens_per_expert_w = (float**)calloc((size_t)num_experts, sizeof(float*));
    int* tokens_per_expert_pos = (int*)calloc((size_t)num_experts, sizeof(int));
    if (!tokens_per_expert_idx || !tokens_per_expert_w || !tokens_per_expert_pos) {
        free(tokens_per_expert_count);
        free(tokens_per_expert_idx); free(tokens_per_expert_w); free(tokens_per_expert_pos);
        free(top_experts_all); free(expert_weights_all); free(ref_output);
        return;
    }
    int n_active_experts = 0;
    int max_M = 0, total_M = 0;
    for (int e = 0; e < num_experts; e++) {
        int c = tokens_per_expert_count[e];
        if (c > 0) {
            tokens_per_expert_idx[e] = (int*)malloc((size_t)c * sizeof(int));
            tokens_per_expert_w[e] = (float*)malloc((size_t)c * sizeof(float));
            if (!tokens_per_expert_idx[e] || !tokens_per_expert_w[e]) {
                /* Clean-up and bail to fallback. */
                for (int ee = 0; ee < num_experts; ee++) {
                    free(tokens_per_expert_idx[ee]);
                    free(tokens_per_expert_w[ee]);
                }
                free(tokens_per_expert_idx); free(tokens_per_expert_w);
                free(tokens_per_expert_pos); free(tokens_per_expert_count);
                free(top_experts_all); free(expert_weights_all); free(ref_output);
                /* Zero + per-token fallback */
                memset(output, 0, (size_t)N * hidden_dim * sizeof(float));
                for (int n = 0; n < N; n++) {
                    state->routing_precomputed = 0;
                    tq_moe_forward(layer, config, state,
                                   hidden + (size_t)n * hidden_dim,
                                   output + (size_t)n * hidden_dim,
                                   hidden_dim, layer_idx);
                }
                return;
            }
            n_active_experts++;
            if (c > max_M) max_M = c;
            total_M += c;
        }
    }
    /* Fill pass */
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < num_active; k++) {
            int eid = top_experts_all[n * num_active + k];
            if (eid < 0 || eid >= num_experts) continue;
            int p = tokens_per_expert_pos[eid]++;
            tokens_per_expert_idx[eid][p] = n;
            tokens_per_expert_w[eid][p] = expert_weights_all[n * num_active + k];
        }
    }

    if (debug) {
        float avg_M = n_active_experts > 0 ? (float)total_M / (float)n_active_experts : 0.0f;
        fprintf(stderr, "[moe_batch L%d] N=%d active_experts=%d max_M=%d avg_M=%.1f\n",
                layer_idx, N, n_active_experts, max_M, (double)avg_M);
    }

    /* Zero output (caller pre-zero is also fine; do it defensively). */
    memset(output, 0, (size_t)N * hidden_dim * sizeof(float));

    /* Phase 3: Per active expert — gather → batched matmul → activation →
     * down matmul → scatter weighted accumulate.
     *
     * Step 3f: cross-expert parallel dispatch. Build a compact list of
     * active experts (those with M_e > 0), then dispatch one task per
     * expert via the shared thread pool. Each worker owns private
     * scratch + a private output accumulator to avoid scatter conflicts.
     *
     * Bounded by TQ_TP_MAX=16 — if n_active_experts > TQ_TP_MAX we process
     * in waves of TQ_TP_MAX experts. In practice for Qwen3.6 (N<=256,
     * num_active=8), n_active_experts stays ≤ min(N*8, 128) ≤ 128 at
     * large N but typical prefill (N=32..128) yields 30-60 active experts.
     */
    #ifndef TQ_TP_MAX
    #define TQ_TP_MAX 16
    #endif
    int max_M_cap = max_M > 64 ? 64 : max_M;

    /* Thread/worker count: cap by active experts count, hw threads, TQ_TP_MAX */
    extern void tq_tp_run(void* (*fn)(void*), void** args, int n_tasks);
    extern void tq_tp_run_dynamic(void* (*fn)(void*), void** args, int n_tasks);
    int n_threads = tq_get_threads();
    if (n_threads > TQ_TP_MAX) n_threads = TQ_TP_MAX;
    if (n_threads < 1) n_threads = 1;
    /* Don't spawn more workers than active experts */
    int n_workers = n_threads;
    if (n_workers > n_active_experts) n_workers = n_active_experts;
    if (n_workers < 1) n_workers = 1;
    /* Already holding the pool (nested call)? Fall back to serial. */
    if (tq_tls_force_serial_matmul) n_workers = 1;

    /* Step 3g: FCFS dynamic dispatch — all active experts queued at once,
     * workers grab via atomic counter. Default ON (Round 10: 12/12 regression
     * PASS, +17% prefill measured on Qwen3.6-UD-Q3_K_S); opt-out via
     * TQ_NO_MOE_BATCH_DYNAMIC. Serial fallback (n_workers==1) always uses
     * the wave path. */
    int use_dynamic = 0;
    if (n_workers > 1 && n_active_experts > n_workers) {
        use_dynamic = !getenv("TQ_NO_MOE_BATCH_DYNAMIC");
    }

    /* Per-worker scratch: X_sub [64 × hidden_dim] + gate_out/up_out [64 × expert_dim]
     * + down_out [64 × hidden_dim] + private output [N × hidden_dim].
     * Total per worker ≈ 64*(hidden+2*expert+hidden) + N*hidden floats.
     * For Qwen3.6 hidden=2560 expert=768 N≤256: ≈ (64*6656 + 256*2560) floats
     * = 1.08 M floats = 4.3 MB per worker × 8 workers = 35 MB — fits 16 GB. */
    size_t per_worker_scratch =
        (size_t)max_M_cap * hidden_dim         /* X_sub */
      + (size_t)max_M_cap * expert_dim         /* gate_out */
      + (size_t)max_M_cap * expert_dim         /* up_out */
      + (size_t)max_M_cap * hidden_dim;        /* down_out */
    size_t per_worker_output = (size_t)N * hidden_dim;
    size_t per_worker_total = per_worker_scratch + per_worker_output;

    float* scratch_all = (float*)malloc((size_t)n_workers * per_worker_total * sizeof(float));
    if (!scratch_all) goto cleanup_and_fallback;
    /* Zero private outputs (scratch can stay uninitialized).
     * Worker 0 reuses the shared `output` buffer (already zeroed above) to save
     * both its memset and its reduce pass — one fewer N×hidden_dim buffer
     * touched per layer. Workers 1..n_workers-1 use the private scratch. */
    for (int w = 1; w < n_workers; w++) {
        float* base = scratch_all + (size_t)w * per_worker_total;
        float* priv_out = base + per_worker_scratch;
        memset(priv_out, 0, per_worker_output * sizeof(float));
    }

    /* Build compact active-expert list */
    int* active_eids = (int*)malloc((size_t)n_active_experts * sizeof(int));
    if (!active_eids) {
        free(scratch_all);
        goto cleanup_and_fallback;
    }
    {
        int ap = 0;
        for (int e = 0; e < num_experts; e++) {
            if (tokens_per_expert_count[e] > 0) active_eids[ap++] = e;
        }
    }

    if (use_dynamic) {
        if (debug) {
            fprintf(stderr, "[moe_batch L%d] DYNAMIC n_active=%d n_workers=%d tp_n=%d\n",
                    layer_idx, n_active_experts, n_workers, tq_get_threads());
        }
        /* Step 3g dynamic path: one task per active expert, all queued
         * at once. Workers grab tasks via FCFS atomic counter; each
         * worker resolves its private scratch/output from scratch_all
         * using tq_tls_worker_id. Long-M_e experts no longer block the
         * wave boundary. */
        moe_batch_expert_task_t* tasks_d =
            (moe_batch_expert_task_t*)malloc(
                (size_t)n_active_experts * sizeof(moe_batch_expert_task_t));
        void** task_ptrs_d = (void**)malloc((size_t)n_active_experts * sizeof(void*));
        if (!tasks_d || !task_ptrs_d) {
            free(tasks_d); free(task_ptrs_d);
            /* Fall back to wave path below */
            use_dynamic = 0;
        } else {
            for (int i = 0; i < n_active_experts; i++) {
                int eid = active_eids[i];
                tasks_d[i].layer = layer;
                tasks_d[i].config = config;
                tasks_d[i].hidden = hidden;
                tasks_d[i].N_total = N;
                tasks_d[i].hidden_dim = hidden_dim;
                tasks_d[i].expert_dim = expert_dim;
                tasks_d[i].eid = eid;
                tasks_d[i].M_e = tokens_per_expert_count[eid];
                tasks_d[i].tokens_idx = tokens_per_expert_idx[eid];
                tasks_d[i].tokens_w = tokens_per_expert_w[eid];
                tasks_d[i].exp_scale = (layer->expert_scale) ? layer->expert_scale[eid] : 1.0f;
                tasks_d[i].activation_fn = activation_fn;
                /* Dynamic-mode: worker resolves scratch via TLS id. */
                tasks_d[i].X_sub          = NULL;
                tasks_d[i].gate_out       = NULL;
                tasks_d[i].up_out         = NULL;
                tasks_d[i].down_out       = NULL;
                tasks_d[i].output_private = NULL;  /* sentinel for dynamic mode */
                tasks_d[i].scratch_all      = scratch_all;
                tasks_d[i].shared_output    = output;
                tasks_d[i].per_worker_total = per_worker_total;
                tasks_d[i].max_M_cap        = max_M_cap;
                tasks_d[i].n_workers        = n_workers;
                task_ptrs_d[i] = &tasks_d[i];
            }
            tq_tp_run_dynamic(moe_batch_expert_worker, task_ptrs_d, n_active_experts);
            free(tasks_d);
            free(task_ptrs_d);
        }
    }

    if (!use_dynamic) {
        /* Wave path (default): process active experts in waves of n_workers.
         * Each worker owns one expert per wave and scatters into its
         * private output buffer (pre-baked per-task). */
        moe_batch_expert_task_t tasks[TQ_TP_MAX];
        void* task_ptrs[TQ_TP_MAX];

        for (int wave_start = 0; wave_start < n_active_experts; wave_start += n_workers) {
            int wave_end = wave_start + n_workers;
            if (wave_end > n_active_experts) wave_end = n_active_experts;
            int wave_count = wave_end - wave_start;

            for (int w = 0; w < wave_count; w++) {
                int eid = active_eids[wave_start + w];
                float* base = scratch_all + (size_t)w * per_worker_total;
                tasks[w].layer = layer;
                tasks[w].config = config;
                tasks[w].hidden = hidden;
                tasks[w].N_total = N;
                tasks[w].hidden_dim = hidden_dim;
                tasks[w].expert_dim = expert_dim;
                tasks[w].eid = eid;
                tasks[w].M_e = tokens_per_expert_count[eid];
                tasks[w].tokens_idx = tokens_per_expert_idx[eid];
                tasks[w].tokens_w = tokens_per_expert_w[eid];
                tasks[w].exp_scale = (layer->expert_scale) ? layer->expert_scale[eid] : 1.0f;
                tasks[w].activation_fn = activation_fn;
                tasks[w].X_sub    = base + 0;
                tasks[w].gate_out = base + (size_t)max_M_cap * hidden_dim;
                tasks[w].up_out   = base + (size_t)max_M_cap * hidden_dim
                                         + (size_t)max_M_cap * expert_dim;
                tasks[w].down_out = base + (size_t)max_M_cap * hidden_dim
                                         + 2 * (size_t)max_M_cap * expert_dim;
                /* Worker 0 scatters directly into shared output (skip memset+reduce). */
                tasks[w].output_private = (w == 0) ? output : (base + per_worker_scratch);
                /* Dynamic-mode fields unused in wave path */
                tasks[w].scratch_all = NULL;
                tasks[w].shared_output = NULL;
                tasks[w].per_worker_total = 0;
                tasks[w].max_M_cap = 0;
                tasks[w].n_workers = 0;
                task_ptrs[w] = &tasks[w];
            }

            if (wave_count == 1) {
                /* Serial: run directly on this thread. */
                moe_batch_expert_worker(task_ptrs[0]);
            } else {
                tq_tp_run(moe_batch_expert_worker, task_ptrs, wave_count);
            }
        }
    }

    /* Reduce per-worker outputs into the shared output, serially.
     * Each worker accumulated different tokens' contributions; we simply
     * sum them (float add is commutative-associative up to rounding).
     * Worker 0 wrote directly into `output` so we skip it. */
    for (int w = 1; w < n_workers; w++) {
        float* base = scratch_all + (size_t)w * per_worker_total;
        const float* priv_out = base + per_worker_scratch;
#if TQ_MOE_HAS_NEON
        size_t total = (size_t)N * hidden_dim;
        size_t i = 0;
        for (; i + 15 < total; i += 16) {
            float32x4_t a0 = vld1q_f32(output + i + 0);
            float32x4_t a1 = vld1q_f32(output + i + 4);
            float32x4_t a2 = vld1q_f32(output + i + 8);
            float32x4_t a3 = vld1q_f32(output + i + 12);
            float32x4_t b0 = vld1q_f32(priv_out + i + 0);
            float32x4_t b1 = vld1q_f32(priv_out + i + 4);
            float32x4_t b2 = vld1q_f32(priv_out + i + 8);
            float32x4_t b3 = vld1q_f32(priv_out + i + 12);
            vst1q_f32(output + i + 0,  vaddq_f32(a0, b0));
            vst1q_f32(output + i + 4,  vaddq_f32(a1, b1));
            vst1q_f32(output + i + 8,  vaddq_f32(a2, b2));
            vst1q_f32(output + i + 12, vaddq_f32(a3, b3));
        }
        for (; i < total; i++) output[i] += priv_out[i];
#else
        size_t total = (size_t)N * hidden_dim;
        for (size_t i = 0; i < total; i++) output[i] += priv_out[i];
#endif
    }

    free(active_eids);
    free(scratch_all);

    /* Shared expert: always-active. Mission A Step 3h — batched dispatch.
     *
     * Q4-converted path (Qwen3.6 typical): one tq_batched_matmul_q4 call
     * each for gate/up/down across all N tokens. Amortizes expert weight
     * reads (76 MB per layer for shared expert alone) and uses the matmul
     * thread pool directly (outer context, tq_tls_force_serial_matmul=0).
     *
     * GGUF-native path (non-Q4): per-token fallback kept — TODO add
     * batched tq_matmul_gguf later.
     *
     * Shared gate (sigmoid scalar per token) computed per-token; cheap
     * compared to the matmul.
     */
    if (config->has_shared_expert) {
        int shared_dim = config->shared_expert_intermediate_dim;
        if (shared_dim == 0) shared_dim = expert_dim;

        int did_batched = 0;
        if (layer->shared_expert.q4_converted) {
            /* Batched Q4 path: three batched matmuls across all N tokens. */
            float* gate_batch = (float*)malloc((size_t)N * shared_dim * sizeof(float));
            float* up_batch   = (float*)malloc((size_t)N * shared_dim * sizeof(float));
            float* down_batch = (float*)malloc((size_t)N * hidden_dim * sizeof(float));
            if (gate_batch && up_batch && down_batch) {
                /* gate_batch[N, shared_dim] = hidden[N, hidden_dim] @ W_gate^T */
                tq_batched_matmul_q4(gate_batch,
                                     layer->shared_expert.gate_q4_qs,
                                     layer->shared_expert.gate_q4_scales,
                                     hidden, shared_dim, hidden_dim, N, NULL);
                /* up_batch[N, shared_dim] = hidden[N, hidden_dim] @ W_up^T */
                tq_batched_matmul_q4(up_batch,
                                     layer->shared_expert.up_q4_qs,
                                     layer->shared_expert.up_q4_scales,
                                     hidden, shared_dim, hidden_dim, N, NULL);

                /* Per-token activation (SwiGLU/GeGLU) on gate/up rows. */
                for (int n = 0; n < N; n++) {
                    activation_fn(gate_batch + (size_t)n * shared_dim,
                                  up_batch   + (size_t)n * shared_dim,
                                  shared_dim);
                }

                /* down_batch[N, hidden_dim] = gate_batch[N, shared_dim] @ W_down^T */
                tq_batched_matmul_q4(down_batch,
                                     layer->shared_expert.down_q4_qs,
                                     layer->shared_expert.down_q4_scales,
                                     gate_batch, hidden_dim, shared_dim, N, NULL);

                /* Scatter with optional sigmoid gate (per-token scalar). */
                for (int n = 0; n < N; n++) {
                    const float* h_n = hidden + (size_t)n * hidden_dim;
                    float shared_gate_val = 1.0f;
                    if (layer->shared_gate) {
                        float dot = 0.0f;
                        for (int j = 0; j < hidden_dim; j++)
                            dot += h_n[j] * layer->shared_gate[j];
                        shared_gate_val = 1.0f / (1.0f + fast_expf_moe(-dot));
                    }
                    float* out_row = output + (size_t)n * hidden_dim;
                    const float* down_row = down_batch + (size_t)n * hidden_dim;
                    for (int i = 0; i < hidden_dim; i++)
                        out_row[i] += shared_gate_val * down_row[i];
                }
                did_batched = 1;
            }
            free(gate_batch);
            free(up_batch);
            free(down_batch);
        }

        if (!did_batched) {
            /* Fallback: per-token.
             *   - GGUF-native shared expert (!q4_converted): TODO batched
             *     tq_matmul_gguf for N tokens in a future round.
             *   - Allocation failure on Q4 path: same per-token recovery. */
            for (int n = 0; n < N; n++) {
                const float* h_n = hidden + (size_t)n * hidden_dim;
                float shared_gate_val = 1.0f;
                if (layer->shared_gate) {
                    float dot = 0.0f;
                    for (int j = 0; j < hidden_dim; j++)
                        dot += h_n[j] * layer->shared_gate[j];
                    shared_gate_val = 1.0f / (1.0f + fast_expf_moe(-dot));
                }
                if (layer->shared_expert.q4_converted) {
                    tq_matmul_q4(state->expert_hb, h_n,
                                 layer->shared_expert.gate_q4_qs, layer->shared_expert.gate_q4_scales,
                                 shared_dim, hidden_dim);
                    tq_matmul_q4(state->expert_hb2, h_n,
                                 layer->shared_expert.up_q4_qs, layer->shared_expert.up_q4_scales,
                                 shared_dim, hidden_dim);
                    activation_fn(state->expert_hb, state->expert_hb2, shared_dim);
                    tq_matmul_q4(state->expert_out, state->expert_hb,
                                 layer->shared_expert.down_q4_qs, layer->shared_expert.down_q4_scales,
                                 hidden_dim, shared_dim);
                } else {
                    tq_matmul_gguf(state->expert_hb, h_n,
                                   layer->shared_expert.w_gate, layer->shared_expert.gate_type,
                                   shared_dim, hidden_dim);
                    tq_matmul_gguf(state->expert_hb2, h_n,
                                   layer->shared_expert.w_up, layer->shared_expert.up_type,
                                   shared_dim, hidden_dim);
                    activation_fn(state->expert_hb, state->expert_hb2, shared_dim);
                    tq_matmul_gguf(state->expert_out, state->expert_hb,
                                   layer->shared_expert.w_down, layer->shared_expert.down_type,
                                   hidden_dim, shared_dim);
                }
                float* out_row = output + (size_t)n * hidden_dim;
                for (int i = 0; i < hidden_dim; i++)
                    out_row[i] += shared_gate_val * state->expert_out[i];
            }
        }
    }

    /* Sanity check: compare output vs ref_output */
    if (sanity && ref_output) {
        float max_diff = 0.0f;
        for (size_t i = 0; i < (size_t)N * hidden_dim; i++) {
            float d = output[i] - ref_output[i];
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
        }
        fprintf(stderr, "[moe_batch_sanity L%d] N=%d max_abs_diff=%.6g\n",
                layer_idx, N, (double)max_diff);
        if (max_diff > 1e-3f) {
            fprintf(stderr, "[moe_batch_sanity L%d] FAIL: diff exceeds 1e-3\n", layer_idx);
        }
    }
    free(ref_output);

    /* Cleanup */
    for (int e = 0; e < num_experts; e++) {
        free(tokens_per_expert_idx[e]);
        free(tokens_per_expert_w[e]);
    }
    free(tokens_per_expert_idx);
    free(tokens_per_expert_w);
    free(tokens_per_expert_pos);
    free(tokens_per_expert_count);
    free(top_experts_all);
    free(expert_weights_all);

#ifdef TQ_HAS_ACCELERATE
    g_cblas_token += N;
#endif
    return;

cleanup_and_fallback:
    /* Couldn't allocate scratch for batched path — fall back per-token.
     * First clean up what we have. */
    for (int e = 0; e < num_experts; e++) {
        free(tokens_per_expert_idx[e]);
        free(tokens_per_expert_w[e]);
    }
    free(tokens_per_expert_idx);
    free(tokens_per_expert_w);
    free(tokens_per_expert_pos);
    free(tokens_per_expert_count);
    free(ref_output);
    memset(output, 0, (size_t)N * hidden_dim * sizeof(float));
    for (int n = 0; n < N; n++) {
        memcpy(state->top_experts, top_experts_all + n * num_active,
               (size_t)num_active * sizeof(int));
        memcpy(state->expert_weights, expert_weights_all + n * num_active,
               (size_t)num_active * sizeof(float));
        state->routing_precomputed = 1;
        tq_moe_forward(layer, config, state,
                       hidden + (size_t)n * hidden_dim,
                       output + (size_t)n * hidden_dim,
                       hidden_dim, layer_idx);
    }
    free(top_experts_all);
    free(expert_weights_all);
}

/* ============================================================
 * Expert memory advise (madvise hints for paging)
 * ============================================================ */

void tq_moe_advise(const tq_moe_layer_t* layer,
                   const int* active_ids, int n_active,
                   int num_experts)
{
    /*
     * TODO: Implement madvise(MADV_WILLNEED) for active experts and
     *       madvise(MADV_DONTNEED) for inactive experts once tensor
     *       size information is available in tq_expert_weights_t.
     *
     * The idea is:
     *   - For each active expert, call madvise(MADV_WILLNEED) on
     *     w_gate, w_up, w_down data regions to prefetch pages.
     *   - For inactive experts, optionally call madvise(MADV_DONTNEED)
     *     to allow the OS to reclaim those pages.
     *
     * This requires knowing the byte size of each weight tensor,
     * which currently isn't stored in tq_expert_weights_t.
     */
    (void)layer;
    (void)active_ids;
    (void)n_active;
    (void)num_experts;
}
