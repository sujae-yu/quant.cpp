## quant.cpp v0.1.0 — First Release

Multi-architecture LLM inference engine in pure C with KV cache compression.

### Highlights

- **3 models supported**: Gemma 3 4B, Qwen3.5-0.8B, Gemma 3 270M
- **3.8x KV cache compression** — at 32K context: 1.2 GB vs llama.cpp's 4.4 GB
- **llama.cpp parity**: 51 tok/s single-thread (vs 50.7 tok/s)
- **Multi-shard safetensors**: loads sharded models (Gemma 4B = 2 shards)
- **Dual tokenizer**: GPT2 byte-level BPE + SentencePiece auto-detect
- **TQM format**: pre-quantized mmap binary, instant loading
- **Zero dependencies**: libc only, ~1MB binary

### Supported Models

| Model | Speed (Q4, 6T) | Quality |
|-------|----------------|---------|
| Gemma 3 4B | 5.2 tok/s | "capital of France" → "Paris" |
| Qwen3.5-0.8B | 82 tok/s | 0.999 cosine vs PyTorch |
| Gemma 3 270M | 176 tok/s | per-layer exact match |

### KV Cache Memory Savings

```
Gemma 3 4B at 32K context:
  llama.cpp (FP16 KV):    4,352 MB
  quant.cpp (Q4 KV):     1,156 MB  ← 3.8x compression
```

### Quick Start

```bash
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

### What's Inside

- 9,000+ lines of pure C — complete inference engine
- 8 quantization types: Uniform, Mixed, PolarQuant, QJL, quant.cpp
- Architecture dispatch: Qwen3.5 (DeltaNet + Attention) + Gemma 3 (Sliding Window + GQA)
- Q4 weight quantization with NEON 2-row batching + thread pool
- Integer Q4×Q8 attention via ARM vdotq_s32
- 20 test suites, 70+ tests
- Python bindings (ctypes), llama.cpp/vLLM integration stubs

### References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — KV cache compression
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1-bit quantized JL transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar coordinate quantization
