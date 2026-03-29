# TurboQuant.cpp — Session State

**Last updated**: 2026-03-29 (grow round 8)
**Last commit**: d3e02cd
**Score**: 99.7%

## Current Status

### What Works
- ✅ **Self-contained LLM inference engine** (pure C, 0 dependencies)
- ✅ **15.6 tok/s** on CPU (Qwen3.5-0.8B, 4 threads, Q8 weights)
- ✅ **17x faster than PyTorch CPU**, 1.5x faster than PyTorch+GPU
- ✅ Q4 weight quantization: 2.1 GB → ~280 MB (7x savings), `-q q4` flag (default)
- ✅ Q8 weight quantization: 2.1 GB → 533 MB (4x savings), `-q q8` flag
- ✅ Streaming BF16: embed/lm_head mmap'd, ~1 GB saved
- ✅ Multi-threaded matmul: pthread, 4 threads, NEON optimized
- ✅ DeltaNet + Self-Attention hybrid forward pass (Qwen3.5)
- ✅ HuggingFace BPE tokenizer (248K vocab)
- ✅ KV cache quantization in inference (Q4, 7.5x compression)
- ✅ Integer Q4×Q8 attention (2.9x faster than FP32)
- ✅ tq_chat.py uses native C engine (not PyTorch)
- ✅ 19 C++ test suites (48+ sub-tests), 22 Python tests
- ✅ CLI: tq_run, tq, tq_chat (native + pytorch fallback)

### What Needs Work (Priority Order)
1. Metal GPU matmul — Apple GPU for further speed
2. Value cache quantization — currently keys only
3. More models — Llama, Phi architecture support

### Key Metrics
| Metric | Value |
|--------|-------|
| CPU inference (4 threads, Q8) | 15.6 tok/s |
| CPU inference (1 thread) | 7.8 tok/s |
| PyTorch CPU | 0.8 tok/s (17-20x slower) |
| PyTorch MPS | 10 tok/s (1.5x slower than our CPU) |
| Weight memory (Q4) | ~280 MB (7x savings) |
| Weight memory (Q8) | 533 MB (4x savings) |
| KV compression | 7.5x (uniform_4b) |
| Integer attention | 2.9-4.8x faster than FP32 |
| Logits cosine vs PyTorch | 0.999 |
| Tests | 19 C++ + 22 Python = 70+ |
| Code | 8,500+ lines C, 191 files |
| Commits | 27 |
