## TurboQuant.cpp v0.1.0 — First Release

Pure C LLM inference engine with KV cache compression. Matches llama.cpp single-thread speed.

### Highlights

- **82 tok/s peak** on Qwen3.5-0.8B (Q4, CPU-only, Apple Silicon)
- **51 tok/s single-thread** — on par with llama.cpp (50.7 tok/s)
- **7.5x KV cache compression** with 0.999 cosine similarity
- **8 quantization types**: Uniform, Mixed, PolarQuant, QJL, TurboQuant
- **TQM format**: pre-quantized binary model, mmap instant load (0.3s)
- **Zero dependencies**: libc only, ~1MB binary
- **One-command quickstart**: `bash scripts/quickstart.sh`

### What's Included

- Complete inference engine: DeltaNet + Self-Attention hybrid (Qwen3.5)
- BPE tokenizer (248K vocab, embedded in TQM)
- Q4 weight quantization with NEON 2-row batching
- Thread pool with zero-overhead dispatch
- Integer Q4×Q8 attention (ARM vdotq_s32)
- 19 test suites, 135 tests
- Python bindings (ctypes)
- llama.cpp / vLLM integration stubs

### Quick Start

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

### References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025)
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
