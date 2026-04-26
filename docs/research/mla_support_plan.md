# MLA Attention Support — Research Plan

> **Goal**: Add support for Multi-head Latent Attention (MLA) models (DeepSeek V2/V3, Kimi-Linear, potentially future Qwen) to unlock **MLA × our 4-bit KV = ~64× total KV compression**.
>
> **Strategic value**: "long-context engine on constrained hardware" — 128K context on 16 GB Mac.

## Why MLA matters for our project

Standard multi-head attention KV cache: `2 × n_heads × head_dim × seq_len × 4 bytes` (FP32).

MLA stores only a compressed **latent KV** vector, typically 4-8× smaller than standard KV per position. Then during attention, latent is decompressed to full K,V via small projection matrices (wk_b, wv_b).

Our existing turbo_kv_4b gives 7-8× KV compression on standard attention. Stacking:

```
MLA latent reduction: ~8×
× quant.cpp 4-bit KV: ~8×
= ~64× total KV size reduction
```

At 64× compression, 16 GB of free RAM can hold ~128K context KV for a mid-sized model. This is a UNIQUE capability — llama.cpp has MLA, but they don't aggressively quant the latent. vLLM has KV compression but is batch-serving oriented, not embeddable.

This is our **long-context moat**.

## MLA architectures in the wild (as of 2026-04)

Per `refs/llama.cpp/src/models/`:

1. **DeepSeek V2/V3** (`deepseek2.cpp`) — original MLA. `wq_a`, `wq_b`, `wkv_a`, `wkv_b` projections. Uses RoPE on compressed q.

2. **Kimi-Linear** (`kimi-linear.cpp`) — MLA without RoPE. Simpler variant.

3. **Qwen3-Next** (memory references) — may use MLA but unclear.

Each has slightly different projection layouts. We'd need per-arch dispatch.

## What would be required

### Phase 1: Loader support
- Detect MLA arch in GGUF metadata (`deepseek2`, `kimi_linear`, etc.)
- Load `wq_a/b`, `wkv_a/b` tensors instead of standard `wq/wk/wv`
- Config: `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`

### Phase 2: Forward pass
Compressed attention compute:
```
latent = input @ wkv_a              # [seq, kv_lora_rank]
q_compressed = input @ wq_a          # [seq, q_lora_rank]
q = q_compressed @ wq_b              # [seq, n_heads × head_dim]
k_full = latent @ wkv_b_k            # on-the-fly from latent
v_full = latent @ wkv_b_v
apply_rope(q_pe_split, k_pe_split)  # per DeepSeek MLA
attention(q, k_full, v_full)
```

### Phase 3: KV cache integration
- Store `latent` in KV cache instead of k,v
- Apply our turbo_kv_4b quantization to latent
- On read, dequant + decompress via wkv_b

### Phase 4: Quantization pipeline
- MLA wkv_b projection is small weight matrix — quantize as Q4 or Q8
- Latent activations are FP32 — quantize via turbo_kv_4b
- Measure MSE of decompressed K,V vs FP32 reference

### Phase 5: Benchmark and release
- Support DeepSeek-V2-Lite 16B as first target (widely available)
- Demo: 128K context on 16 GB Mac via Gradio
- Tech report with compression vs PPL curves

## Engineering effort estimate

- Phase 1 (loader): 2-3 days. Straightforward GGUF parsing.
- Phase 2 (forward pass): 5-7 days. Non-trivial — MLA has multiple nope/rope splits, ROPE on subset of head_dim.
- Phase 3 (KV cache): 3-5 days. Our KV infrastructure is good but wasn't designed for latent-then-decompress.
- Phase 4 (quant): 2-3 days. Mostly reuses turbo_kv_4b + small wb quant.
- Phase 5 (release): 3-5 days. Benchmarks, demo, docs.

**Total: 15-23 days of focused engineering.** Multi-session, but tractable.

## What this session has (2026-04-25)

- Recognition of MLA as strategic direction (this doc)
- `refs/llama.cpp` contains working reference (deepseek2.cpp, kimi-linear.cpp)
- Our turbo_kv_4b already achieves 7-8× KV compression on standard attention (validated)
- Measurement infrastructure (coh_bench.sh) ready to evaluate MLA models when loader lands

## Non-goals

- NOT implementing MLA attention in general (llama.cpp has it, vLLM has it). We implement specifically with **latent KV compression** applied on top.
- NOT targeting training. We handle inference only.
- NOT targeting MoE + MLA hybrid (DeepSeek V3) in Phase 1. That's Phase 6+.

## First step when resuming

1. Download DeepSeek-V2-Lite 16B (Q4_K_M) from HF. ~8 GB.
2. Run `llama.cpp` to establish reference output. Capture perf + quality.
3. Attempt to load with our engine → expect load failure on MLA tensors.
4. Implement Phase 1 loader to get the model loaded.
5. Decide: build forward pass from scratch, or adapt our existing hybrid-attention path?

Owner: future session(s). This plan is the handoff.

## Phase 1 results (2026-04-26)

Downloaded `bartowski/DeepSeek-Coder-V2-Lite-Instruct-IQ3_XXS.gguf` (6.96 GB, 16B-MoE/2.4B-active, MLA + YaRN, 27 layers).

**Load: SUCCESS, no engine changes required.**
- arch detected as `deepseek2` (already passes through our generic loader)
- 27 blocks parsed, MoE config (64 experts × 6 active, expert_dim=1408, shared=2) recognised
- IQ3_XXS dequant fired correctly (existing R3 infrastructure)
- 6.5 GB mlock — fits cleanly in 16 GB RAM, no paging
- Decode: **5.1 tok/s** on M1 Pro single-thread — order-of-magnitude faster than 27B-dense Tier 3 attempts

**Forward pass: BROKEN, as expected.**
- Output on `Hello` prompt: `肯定里<\/ Includingatha` — multilingual garbage
- Our generic transformer treats `attn_q`/`attn_kv_a_mqa`/`attn_kv_b` as if they were `wq`/`wk`/`wv`. Shape mismatch produces nonsensical attention scores.
- MLA invariants the loader does NOT yet honour:
  - `attn_kv_a_mqa` shape `[2048, 576]` = 512 latent + 64 RoPE-K (down-proj, single head, MQA-style)
  - `attn_kv_a_norm` shape `[512]` (latent norm before up-proj)
  - `attn_kv_b` shape `[512, 4096]` (latent → full K|V split, 16 heads × (128 nope-K + 128 V))
  - `attn_q` shape `[2048, 3072]` = 16 heads × (64 nope-Q + 128 rope-Q) — Q has its own RoPE/no-RoPE split
  - `kv_lora_rank=512`, `key_length=192`, `value_length=128`
- KV-cache layout would change from `2 × n_heads × head_dim × seq_len` to `(kv_lora_rank + qk_rope_head_dim) × seq_len = 576 × seq_len`
  → **10.7× compression at the architectural level**, before our turbo_kv_4b's 8× → **~85× total** target (MLA × turbo_kv_4b), unblocking 256 K context on 16 GB.
- YaRN RoPE scaling (factor 40, original ctx 4096, log_multiplier 0.0707) needs distinct math from standard RoPE.

**Phase 1 verdict**: loader path through existing GGUF reader works without modification — the architectural change is entirely in the forward pass. Move directly to Phase 2 (MLA attention compute) when next picked up.

**Phase 2 entry plan**:
1. Detect `gctx->arch == "deepseek2"` and dispatch to a new `deepseek2_attention_forward` instead of the standard hybrid path.
2. Allocate latent KV cache `[max_seq, kv_lora_rank + qk_rope_head_dim]` per layer.
3. Implement compressed forward: latent = x @ kv_a_mqa → split → norm latent → cache; on read, K|V = latent @ kv_b → split → standard attention with Q from `attn_q`.
4. YaRN RoPE for the rope-Q / rope-K subset.
5. Validate: bit-exact L0 attn output vs llama.cpp on the same prompt before going to multi-layer.
