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
