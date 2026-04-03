# quant.cpp — Session State

**Last updated**: 2026-04-02
**Score**: 98.1% (structure 90%, correctness 100%, quality 100%, perf 100%, integration 100%)
**Tests**: 31/31 pass, 0 warnings, ASan clean

## What Works

### Core Engine
- 15,000+ lines C, self-built inference engine (not a fork/wrapper)
- Gemma 3 (4B, 270M) + Qwen3.5 (0.8B) + Qwen3.5-35B-A3B MoE
- TQM format (instant mmap loading) + GGUF v3 (Q8_0, Q4_K_M, IQ2_XXS)
- DeltaNet hybrid + sliding window + GQA + MoE (256 experts, top-8, shared)
- GGUF tokenizer loading from metadata

### quant.cpp KV Compression (핵심 가치)
- 12 KV quantization types (RHT + Lloyd-Max + QJL)
- 1-bit K: byte-identical output on ALL verified models (270M~35B)
- Gemma 4B PPL: +0.03% with 1-bit K + Q4 V (4.9x compression)
- Q4 V fused attention (packed nibble direct accumulation)
- Adaptive: per-layer bit recommendation, codebook calibration (49.7% MSE gain)
- Formal: unbiasedness < 0.2% (100K samples), cosine = 2/pi (theoretical limit)

### Verification
- 31 test suites: PPL, unbiasedness, attention distribution, codebook theory, NEON consistency, edge cases, rate-distortion, cumulative error
- ASan + UBSan clean
- Activation profiling (--profile-kv), perplexity (--ppl), layer recommendation (--recommend)

### GPU
- Metal: matmul shaders (IQ2_XXS, Q8_0, Q4_K), runtime compile, batched dispatch
- Metal functional but not faster than CPU yet (dispatch overhead)

## Speed

```
Model                  Format       CPU (6T)    Metal
Gemma 270M (TQM)      Q4           176 tok/s   -
Qwen 0.8B (TQM)       Q4           80.1 tok/s  -
Qwen 0.8B (GGUF)      Q8_0         7.4 tok/s   3.7 tok/s
Qwen 0.8B (GGUF)      Q4_K_M       4.8 tok/s   3.1 tok/s
Gemma 4B (TQM)        Q4           20.2 tok/s  -
35B MoE (GGUF)        IQ2_XXS      ~1.0 tok/s  ~0.7 tok/s
```

## What Needs Work (Priority Order)

### P0: GPU Performance (biggest user-facing gap)
- Metal per-matmul dispatch overhead → need full forward GPU graph
- 35B: 1 tok/s CPU vs llama.cpp ~8-12 tok/s Metal — 10x gap
- GGUF matmul on GPU needs architecture redesign (batch entire layers)

### P1: GGUF Speed Parity
- GGUF path 10x slower than TQM path (7.4 vs 80.1 tok/s on 0.8B)
- Root cause: on-the-fly dequant per matmul, no weight pre-processing
- Solution: pre-dequant to Q8 at load time (like TQM Q4 path)

### P2: Model Coverage
- No Llama/Phi/Mistral support
- No 8B model verified (gap between 4B and 35B)
- Need Qwen3.5-7B or Llama-3.1-8B for mid-range PPL verification

### P3: GGUF Tokenizer for PPL
- --ppl works on TQM (embedded tokenizer) but not GGUF
- GGUF tokenizer loads for generation but not for PPL path
- Need to wire GGUF tokenizer into PPL mode

### P4: Metal Architecture Redesign
- Current: individual matmul GPU dispatch
- Need: compute graph (encode all layer ops, commit once per layer)
- This is a major refactor but the only path to competitive GPU speed

### P5: Documentation Sync
- state.md was stale (from v0.9.3, now updated)
- docs/RELEASE_NOTES.md needs v0.4.0 entry for GGUF/MoE/Metal
- WBS progress needs update (score.sh structure dimension)

### P6: IQ2_XXS Quality on 35B
- Simple factual queries work ("Paris"), complex reasoning fails
- This is IQ2 weight limitation, not quant.cpp issue
- Need Q4_K or Q5_K 35B model (~15GB) to prove quality
