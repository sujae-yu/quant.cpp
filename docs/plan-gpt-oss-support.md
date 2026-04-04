# Plan: gpt-oss-20b (OpenAI MoE) Architecture Support

## Background

The `gpt-oss-20b` model (marketed as `chatgpt-20b`) is OpenAI's first open-weight release.
In llama.cpp it is registered as `LLM_ARCH_OPENAI_MOE` with GGUF arch string `"gpt-oss"`.
It uses an ISWA (Interleaved Sliding Window Attention) variant with MoE.

### Architecture Summary

| Parameter | Value (20B) | Value (120B) |
|-----------|-------------|--------------|
| Layers | 24 | 36 |
| Hidden dim | ~4096 (TBD) | ~8192 (TBD) |
| Experts | 32 total, 4 active | 32 total, 4 active |
| Attention | GQA with RoPE | GQA with RoPE |
| Activation | SwiGLU-OAI (clamped, alpha=1.702) | same |
| Norm | RMSNorm (pre-attn + post-attn) | same |
| QKV bias | Yes (all of Q, K, V, O) | same |
| Expert bias | Yes (gate, up, down + router) | same |
| Sliding window | Yes, alternating pattern (period=2) | same |
| Attention sinks | Per-head learned bias [n_heads] | same |

### Key Differences from Existing Architectures

| Feature | Llama/Qwen | Qwen2-MoE | Gemma 4 | gpt-oss |
|---------|-----------|-----------|---------|---------|
| QKV bias | No | No | No | **Yes** |
| Expert bias | No | No | No | **Yes** |
| Activation | SwiGLU | SwiGLU | GeGLU | **SwiGLU-OAI** |
| Post-attn norm | No | No | Yes | **Yes** |
| Attn sinks | No | No | No | **Yes** |
| SWA pattern | No | No | Fixed | **Alternating (period=2)** |
| Router gating | Softmax+renorm | Softmax+renorm | Softmax+renorm | **Softmax-weight** (select then softmax) |
| expert_weights_scale | No | No | No | **Yes** |

## GGUF Tensor Layout

Based on `refs/llama.cpp/src/llama-model.cpp` lines 6829-6868 and `refs/llama.cpp/src/llama-arch.cpp`:

### Global Tensors
```
token_embd.weight          — [n_embd, n_vocab]
output_norm.weight         — [n_embd]
output.weight              — [n_embd, n_vocab]
```

### Per-Layer Tensors (`blk.%d.`)
```
attn_norm.weight                — [n_embd]           RMSNorm pre-attention
post_attention_norm.weight      — [n_embd]           RMSNorm post-attention (NEW)

attn_q.weight                   — [n_embd, n_heads * head_dim]
attn_q.bias                     — [n_heads * head_dim]        (NEW)
attn_k.weight                   — [n_embd, n_kv_heads * head_dim]
attn_k.bias                     — [n_kv_heads * head_dim]     (NEW)
attn_v.weight                   — [n_embd, n_kv_heads * head_dim]
attn_v.bias                     — [n_kv_heads * head_dim]     (NEW)
attn_output.weight              — [n_heads * head_dim, n_embd]
attn_output.bias                — [n_embd]                    (NEW)

attn_sinks.weight               — [n_heads]                   (NEW)

ffn_gate_inp.weight             — [n_embd, n_expert]          Router
ffn_gate_inp.bias               — [n_expert]                  (NEW)
ffn_gate_exps.weight            — [n_embd, n_ff_exp, n_expert]
ffn_gate_exps.bias              — [n_ff_exp, n_expert]        (NEW)
ffn_up_exps.weight              — [n_embd, n_ff_exp, n_expert]
ffn_up_exps.bias                — [n_ff_exp, n_expert]        (NEW)
ffn_down_exps.weight            — [n_ff_exp, n_embd, n_expert]
ffn_down_exps.bias              — [n_embd, n_expert]          (NEW)
```

### GGUF Metadata Keys
```
gpt-oss.block_count
gpt-oss.embedding_length
gpt-oss.attention.head_count
gpt-oss.attention.head_count_kv
gpt-oss.attention.layer_norm_rms_epsilon
gpt-oss.expert_count                     — 32
gpt-oss.expert_used_count                — 4
gpt-oss.expert_feed_forward_length
gpt-oss.attention.sliding_window
gpt-oss.attention.sliding_window_pattern — 2 (alternating)
gpt-oss.rope.freq_base
gpt-oss.rope.freq_base_swa              — SWA layers use different RoPE base
gpt-oss.expert_weights_scale             — float, applied to expert output
```

## Implementation Plan

### Phase 1: Struct Changes (tq_engine.h, tq_gguf.h)

**File: `include/turboquant/tq_engine.h`**

1. Add QKV bias fields to `tq_layer_weights_t` (after line 81):
```c
/* Attention bias (gpt-oss) — NULL if not present */
float* bq;            /* [n_heads * head_dim] Q bias */
float* bk;            /* [n_kv_heads * head_dim] K bias */
float* bv;            /* [n_kv_heads * head_dim] V bias */
float* bo;            /* [hidden_dim] output bias */
```

2. Add attention sinks to `tq_layer_weights_t`:
```c
/* Attention sink bias (gpt-oss) — per-head learned bias [n_heads], NULL if not used */
float* attn_sinks;
```

3. Add fields to `tq_model_config_t` (around line 47):
```c
int is_gpt_oss;          /* 1 if gpt-oss architecture */
int swa_pattern;         /* SWA layer pattern period (2 = alternating, 0 = disabled) */
float expert_weights_scale; /* per-expert output scaling (0.0 = disabled) */
```

**File: `include/turboquant/tq_gguf.h`**

4. Add bias pointers to `tq_expert_weights_t` (after line 246):
```c
/* Expert biases (gpt-oss) — NULL if not present */
const float* gate_bias;   /* [expert_inter] */
const float* up_bias;     /* [expert_inter] */
const float* down_bias;   /* [hidden_dim] */
```

5. Add router bias to `tq_moe_layer_t` (after line 252):
```c
float* router_bias;       /* [num_experts] router gate bias (NULL if not used) */
```

6. Add activation type to `tq_moe_config_t` (replace `use_gelu` with enum):
```c
int activation_type;  /* 0=SwiGLU, 1=GeGLU, 2=SwiGLU-OAI */
```

### Phase 2: Model Loading (tq_model.c)

**File: `src/engine/tq_model.c`**

7. **Architecture detection** (around line 2991, the model_type detection block):
```c
} else if (strstr(gguf->arch, "gpt-oss") != NULL) {
    c->model_type = 3; /* gpt-oss */
    c->is_gpt_oss = 1;
    /* gpt-oss uses post-attention norm (2 norms per block for dense path) */
    c->n_norms_per_block = 2;
}
```

8. **Read gpt-oss-specific metadata** (after MoE config, around line 2988):
```c
/* gpt-oss specific: SWA pattern and expert_weights_scale */
if (c->is_gpt_oss) {
    c->swa_pattern = tq_gguf_get_i32(gguf, GGUF_KEY("attention.sliding_window_pattern"), 2);
    c->expert_weights_scale = tq_gguf_get_f32(gguf, GGUF_KEY("expert_weights_scale"), 0.0f);
    /* SWA layers use different RoPE base */
    c->rope_local_base_freq = tq_gguf_get_f32(gguf, GGUF_KEY("rope.freq_base_swa"),
                               c->rope_freq_base);
}
```

9. **Set up layer_is_sliding for alternating SWA** (after line 3520):
```c
/* gpt-oss: alternating SWA pattern (every swa_pattern-th layer is full/global) */
if (c->is_gpt_oss && c->sliding_window > 0 && c->swa_pattern > 0) {
    model->layer_is_sliding = (int*)calloc((size_t)c->n_layers, sizeof(int));
    for (int l = 0; l < c->n_layers; l++) {
        /* Period=2: odd layers are sliding, even layers are full (or vice versa) */
        model->layer_is_sliding[l] = (l % c->swa_pattern != 0) ? 1 : 0;
    }
}
```

10. **Load QKV bias tensors** (in the per-layer tensor loading loop, after line 3235):
```c
/* Attention biases (gpt-oss) */
snprintf(tname, sizeof(tname), "blk.%d.attn_q.bias", l);
t = find_gguf_tensor(gguf, tname);
if (t) layer->bq = dequant_tensor_fp32(t);
snprintf(tname, sizeof(tname), "blk.%d.attn_k.bias", l);
t = find_gguf_tensor(gguf, tname);
if (t) layer->bk = dequant_tensor_fp32(t);
snprintf(tname, sizeof(tname), "blk.%d.attn_v.bias", l);
t = find_gguf_tensor(gguf, tname);
if (t) layer->bv = dequant_tensor_fp32(t);
snprintf(tname, sizeof(tname), "blk.%d.attn_output.bias", l);
t = find_gguf_tensor(gguf, tname);
if (t) layer->bo = dequant_tensor_fp32(t);
```

11. **Load attention sinks** (same loop):
```c
snprintf(tname, sizeof(tname), "blk.%d.attn_sinks.weight", l);
t = find_gguf_tensor(gguf, tname);
if (t) layer->attn_sinks = dequant_tensor_fp32(t);
```

12. **Load post_attention_norm** for gpt-oss (around line 3122):
The code already loads `blk.%d.post_attention_norm.weight` for gemma3. Need to
also load it when `c->is_gpt_oss`:
```c
/* post_attention_norm: Gemma family OR gpt-oss */
if (c->model_type == 1 || c->is_gpt_oss) {
    snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
    ...
}
```

13. **Load expert biases** (in the MoE loading section, after line 3400):
```c
/* Expert biases (gpt-oss) */
if (c->is_gpt_oss) {
    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_exps.bias", l);
    const tq_gguf_tensor_t* gb = find_gguf_tensor(gguf, tname);
    snprintf(tname, sizeof(tname), "blk.%d.ffn_up_exps.bias", l);
    const tq_gguf_tensor_t* ub = find_gguf_tensor(gguf, tname);
    snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.bias", l);
    const tq_gguf_tensor_t* db = find_gguf_tensor(gguf, tname);
    if (gb && ub && db) {
        int exp_inter = c->expert_intermediate_dim;
        size_t gate_bias_size = (size_t)exp_inter * sizeof(float);
        size_t down_bias_size = (size_t)c->hidden_dim * sizeof(float);
        for (int e = 0; e < c->num_experts; e++) {
            moe->experts[e].gate_bias = (const float*)gb->data + e * exp_inter;
            moe->experts[e].up_bias   = (const float*)ub->data + e * exp_inter;
            moe->experts[e].down_bias = (const float*)db->data + e * c->hidden_dim;
        }
    }
    /* Router bias */
    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.bias", l);
    t = find_gguf_tensor(gguf, tname);
    if (t) moe->router_bias = dequant_tensor_fp32(t);
}
```

14. **Set MoE activation type** (around line 3043):
```c
moe_cfg->activation_type = c->is_gemma4 ? 1 : (c->is_gpt_oss ? 2 : 0);
```

### Phase 3: Forward Pass (tq_transformer.c, tq_moe.c, tq_ops.c)

**File: `src/engine/tq_ops.c`**

15. **Add SwiGLU-OAI activation function** (after `tq_gelu_tanh` around line 1260):
```c
/**
 * SwiGLU-OAI activation: clamped SiLU with multiplicative gate
 *   x = clamp(x, -, limit)   [limit=7.0 by default]
 *   g = clamp(g, -limit, limit)
 *   out = (x * sigmoid(alpha * x)) * (1 + g)
 *   alpha = 1.702
 *
 * Reference: refs/llama.cpp/ggml/src/ggml-cuda/unary.cuh line 105-111
 */
void tq_swiglu_oai(float* restrict hb, const float* restrict hb2, int n);
```

Implementation (~30 lines with NEON):
- Clamp gate input `hb[i]` to `[-inf, 7.0]` (one-sided for x)
- Clamp up input `hb2[i]` to `[-7.0, 7.0]` (two-sided for g)
- Compute: `silu(alpha * x) * (1 + g)` where alpha=1.702

**File: `src/engine/tq_moe.c`**

16. **Add SwiGLU-OAI to expert forward dispatch** (around line 694):
```c
void (*activation_fn)(float* restrict, const float* restrict, int);
switch (config->activation_type) {
    case 1:  activation_fn = geglu_fused; break;
    case 2:  activation_fn = swiglu_oai_fused; break;
    default: activation_fn = swiglu_fused; break;
}
```

17. **Add bias support to expert FFN** (in the per-expert loop, after gate/up matmul):
```c
/* Add expert biases if present */
if (exp->gate_bias) {
    for (int i = 0; i < expert_dim; i++) state->expert_hb[i] += exp->gate_bias[i];
}
if (exp->up_bias) {
    for (int i = 0; i < expert_dim; i++) state->expert_hb2[i] += exp->up_bias[i];
}
```
And after down projection:
```c
if (exp->down_bias) {
    for (int i = 0; i < hidden_dim; i++) state->expert_out[i] += exp->down_bias[i];
}
```

18. **Add router bias to routing** (in `tq_moe_route`, after the router matmul, around line 600):
```c
/* Add router bias if present */
if (layer->router_bias) {
    for (int e = 0; e < num_experts; e++)
        logits[e] += layer->router_bias[e];
}
```
Note: `tq_moe_route` currently takes `router_weight` only. Either:
- (a) Pass `tq_moe_layer_t*` instead of just `router_weight`, or
- (b) Add a `router_bias` parameter to `tq_moe_route`.

Option (b) is cleaner:
```c
void tq_moe_route(const float* hidden, const float* router_weight,
                  const float* router_bias,  /* NEW: NULL if no bias */
                  int num_experts, int num_active, int hidden_dim,
                  int* out_expert_ids, float* out_expert_weights);
```

19. **Add expert_weights_scale** (after softmax renormalization in `tq_moe_forward`):
```c
/* Apply expert output scale (gpt-oss) */
if (config->expert_weights_scale != 0.0f) {
    for (int k = 0; k < num_active; k++)
        state->expert_weights[k] *= config->expert_weights_scale;
}
```
Add `float expert_weights_scale;` to `tq_moe_config_t`.

**File: `src/engine/tq_transformer.c`**

20. **Add QKV bias in attention forward** (after Q/K/V matmul, around lines 1050-1080):
```c
/* Apply QKV biases (gpt-oss) */
if (layer->bq) {
    for (int i = 0; i < q_dim; i++) s->q[i] += layer->bq[i];
}
if (layer->bk) {
    for (int i = 0; i < kv_dim; i++) s->k[i] += layer->bk[i];
}
if (layer->bv) {
    for (int i = 0; i < kv_dim; i++) s->v[i] += layer->bv[i];
}
```

21. **Add output bias after O projection** (after line ~1140):
```c
/* Apply output projection bias (gpt-oss) */
if (layer->bo) {
    for (int i = 0; i < dim; i++) s->xb2[i] += layer->bo[i];
}
```

22. **Add attention sinks** (in attention score computation, around line 1420):
The attn_sinks tensor is a per-head bias applied to the attention scores for
the first few (sink) positions. In quant.cpp's scalar attention:
```c
/* Apply attention sink bias (gpt-oss) — adds learned per-head bias to position 0 scores */
if (layer->attn_sinks) {
    /* attn_sinks[h] is added to the attention logit at position 0 for head h.
     * This helps the model maintain attention to BOS/initial tokens in SWA layers. */
    s->att[h * seq_len + 0] += layer->attn_sinks[h];
}
```
Note: The exact semantics of attn_sinks needs verification from the GGUF spec.
It may apply to more than just position 0 (could be the first N sink positions).
This should be confirmed during implementation by testing with the actual model.

23. **Add post-attention norm for gpt-oss** (around line 2147):
The code already handles `post_attn_norm` for gemma3. Extend the condition:
```c
if ((is_gemma3 || c->is_gpt_oss) && layer->post_attn_norm) {
    tq_rmsnorm(s->xb2, s->xb2, layer->post_attn_norm, dim, c->rms_norm_eps);
}
```

24. **Add SWA layer dispatch for gpt-oss** (in the RoPE section around line 1092):
Currently SWA RoPE base switching is gated on `model_type == 1` (gemma3).
Extend:
```c
if ((c->model_type == 1 || c->is_gpt_oss) && c->rope_local_base_freq > 0.0f &&
    model->layer_is_sliding && model->layer_is_sliding[l]) {
    rope_base = c->rope_local_base_freq;
}
```

Similarly for the attention window limit (line 1430):
```c
if ((c->model_type == 1 || c->is_gpt_oss) && c->sliding_window > 0 &&
    model->layer_is_sliding && model->layer_is_sliding[l]) {
    /* Limit attention to sliding window */
}
```

### Phase 4: Memory Cleanup (tq_model.c)

**File: `src/engine/tq_model.c`**

25. Free new bias allocations in `tq_free_model` (around line 4300+):
```c
free(layer->bq);
free(layer->bk);
free(layer->bv);
free(layer->bo);
free(layer->attn_sinks);
/* Expert biases are pointers into GGUF mmap, no free needed */
/* Router bias: free if dequanted */
if (moe) free(moe->router_bias);
```

### Phase 5: TQM Binary Format (tq_model.c)

26. Update `tq_model_header_t` to include `is_gpt_oss` flag and new config fields.
The existing save/load for TQM format (lines 4137+, 2543+) needs:
- Add `swa_pattern`, `expert_weights_scale`, `is_gpt_oss` to header
- Save/load bias tensors alongside weights

This is lower priority -- GGUF loading is the primary path.

## Reusable Code (No Changes Needed)

The following existing code works as-is for gpt-oss:

| Component | Location | Why it works |
|-----------|----------|--------------|
| RoPE | `tq_ops.c:1185` | Standard LLaMA-style RoPE, same convention |
| GQA attention | `tq_transformer.c` | Standard GQA with n_heads/n_kv_heads |
| MoE routing | `tq_moe.c:580-678` | Already does softmax-over-selected (matches SOFTMAX_WEIGHT) |
| MoE expert dispatch | `tq_moe.c:684+` | Gate/up/down structure matches |
| Expert Q4 LRU cache | `tq_moe.c` | Works for any expert weight format |
| Expert madvise | `tq_moe.c` | Works for any MoE model |
| RMSNorm | `tq_ops.c` | Same norm type |
| GGUF on-the-fly dequant | `tq_ops.c` | All GGUF quant types supported |
| KV cache + compression | `src/cache/` | Architecture-independent |
| Metal MoE dispatch | `src/backend/metal/` | Expert weight format compatible |

## Estimated Scope

| Area | Lines Added | Lines Modified |
|------|-------------|----------------|
| `tq_engine.h` — struct fields | ~15 | ~5 |
| `tq_gguf.h` — MoE bias fields | ~10 | ~5 |
| `tq_model.c` — arch detection | ~20 | ~10 |
| `tq_model.c` — metadata parsing | ~15 | ~5 |
| `tq_model.c` — SWA pattern setup | ~15 | ~5 |
| `tq_model.c` — tensor loading (bias) | ~60 | ~10 |
| `tq_model.c` — memory cleanup | ~10 | ~5 |
| `tq_ops.c` — swiglu_oai activation | ~40 | 0 |
| `tq_moe.c` — activation dispatch | ~5 | ~10 |
| `tq_moe.c` — expert bias support | ~20 | ~5 |
| `tq_moe.c` — router bias + scale | ~15 | ~10 |
| `tq_transformer.c` — QKV bias | ~25 | ~5 |
| `tq_transformer.c` — attn sinks | ~10 | ~5 |
| `tq_transformer.c` — post-attn norm | ~5 | ~5 |
| `tq_transformer.c` — SWA dispatch | ~5 | ~10 |
| **Total** | **~270** | **~95** |

Estimated total: **~365 lines** of changes across 7 files.

## Test Plan

### Unit Tests

1. **`tq_swiglu_oai` activation** — verify against reference implementation:
   - `swiglu_oai(x=3.0, g=0.5, alpha=1.702, limit=7.0)` matches CUDA kernel
   - Edge cases: x > limit (clamped), g at +/-limit boundaries
   - Zero input, negative input

2. **QKV bias application** — verify Q + bias == expected for known input

3. **Expert bias application** — verify gate/up/down bias adds correctly

4. **SWA pattern** — verify layer_is_sliding pattern for period=2:
   - Layer 0: full, Layer 1: sliding, Layer 2: full, ...

### Integration Tests

5. **GGUF loading** — load a small gpt-oss Q4 GGUF, verify:
   - All tensor names resolved
   - Bias tensors loaded (non-NULL)
   - MoE config: 32 experts, 4 active
   - layer_is_sliding pattern correct

6. **End-to-end inference** — generate tokens from chatgpt-20b-Q4_K_M.gguf:
   - Verify no NaN/Inf in output logits
   - Compare top-5 token probabilities with llama.cpp reference output

### Regression Tests

7. **Existing model tests** — verify Llama, Qwen, Gemma models unaffected:
   - All NULL bias pointers remain NULL for non-gpt-oss models
   - `tq_moe_route` signature change (added `router_bias` param) updated at all call sites

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Attention sinks semantics unclear | Medium | Test with actual model; compare with llama.cpp output |
| Expert biases stored as 3D tensors | Low | Follow llama.cpp's [n_ff_exp, n_expert] layout |
| SWA pattern != simple alternating | Low | Read metadata `sliding_window_pattern`; fallback to period=2 |
| `tq_moe_route` API change breaks callers | Low | Add NULL default for router_bias at all existing call sites |
| Performance regression from bias additions | Low | Bias add is O(dim), negligible vs matmul O(dim^2) |

## Implementation Order

1. **Struct changes** (Phase 1) — header-only, no functional change
2. **SwiGLU-OAI activation** (Phase 3, step 15) — independent, unit testable
3. **Model loading** (Phase 2) — wire up all new tensors
4. **Transformer forward** (Phase 3, steps 20-24) — QKV bias + sinks + norms
5. **MoE forward** (Phase 3, steps 16-19) — bias + activation + scale
6. **End-to-end test** with actual GGUF file
7. **TQM format** (Phase 5) — optional, only if needed

Steps 1-2 can be done first as a small PR, followed by steps 3-6 as the main PR.
