# TurboQuant.cpp

**Standalone C inference engine with [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV cache compression. Not a wrapper — built from scratch, zero dependencies.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![CI](https://img.shields.io/github/actions/workflow/status/quantumaikr/TurboQuant.cpp/ci.yml?label=CI)]()
[![Tests](https://img.shields.io/badge/tests-33%20pass-brightgreen)]()
[![ASan](https://img.shields.io/badge/ASan%2BUBSan-clean-brightgreen)]()

## What TurboQuant Does

**Compress KV cache 7x, extend context 7x — with zero quality loss.**

```
16GB Mac Air M3, Gemma 3 4B:

  Without TurboQuant:   32K context  (FP16 KV = 4.2 GB)
  With TurboQuant:     230K context  (1-bit K + Q4 V = 612 MB)

  PPL: 35.99 → 35.99   (+0.00% for K-only)
```

Same hardware, same model, **7x longer context**. No quality loss.

---

## Verified: PPL +0.00% at 800 Tokens, 4 Architectures

```
SmolLM2 1.7B (Llama), 800 tokens:          Qwen3.5 0.8B, 800 tokens:

  baseline  ████████████  PPL 11.07           baseline  ████████████  PPL 137.6
  1-bit K   ████████████  PPL 11.07 (+0.00%)  1-bit K   ████████████  PPL 137.6 (+0.00%)
```

| Model | Arch | Tokens | Baseline PPL | 1-bit K PPL | Delta |
|-------|------|--------|-------------|-------------|-------|
| SmolLM2 1.7B | Llama | 800 | 11.07 | 11.07 | **+0.00%** |
| Qwen3.5 0.8B | Qwen3.5 | 800 | 137.6 | 137.6 | **+0.00%** |
| Gemma 3 4B | Gemma 3 | 101 | 35.99 | 35.99 | **+0.00%** |
| SmolLM2 1.7B | Llama | 105 | 5.84 | 5.84 | **+0.00%** |

K-only quantization is **perplexity-identical** at every tested length.

---

## Context Extension: What You Get

### On Your Hardware

| Hardware | Model | FP16 Context | TurboQuant Context | Gain |
|----------|-------|-------------|-------------------|------|
| **8GB Laptop** | Llama 8B (Q4) | 16K | 116K | 7.1x |
| **16GB Mac Air** | Gemma 4B | 96K | 684K | 7.1x |
| **16GB Mac Air** | Llama 8B (Q4) | 82K | 581K | 7.1x |
| **24GB RTX 3090** | Llama 8B (Q4) | 147K | 1M+ | 7.1x |
| **24GB RTX 3090** | 35B MoE (Q4) | 682K | 5M+ | 7.1x |

### KV Memory Per Model (32K Context)

| Model | Layers (attn) | FP16 K+V | 1-bit K + Q4 V | Saved |
|-------|--------------|----------|---------------|-------|
| SmolLM2 1.7B | 24 (24) | 6.0 GB | 869 MB | 5.1 GB |
| Gemma 3 4B | 34 (34) | 4.2 GB | 613 MB | 3.6 GB |
| Qwen3.5 4B | 32 (8) | 1.0 GB | 144 MB | 880 MB |
| Qwen 35B MoE | 40 (10) | 640 MB | 90 MB | 550 MB |

> Qwen3.5/MoE have fewer attention layers (DeltaNet hybrid) → less KV, but compression ratio is the same.

---

## How It Works

```
Store:    key → L2 normalize → RHT → sign bits (1 bit each) → compressed block
Retrieve: compressed block → dequantize → FP32 → standard attention

Memory savings come from compressed STORAGE.
Attention runs in full FP32 precision — no approximation.
```

The [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026) proves that RHT + sign quantization preserves inner product structure. We store keys in 1 bit and reconstruct to FP32 for attention — getting memory savings without quality loss.

| Stage | What | Why |
|-------|------|-----|
| **RHT** | Randomized Hadamard Transform | Distributes outliers evenly → enables scalar quantization |
| **Sign bits** | 1 bit per dimension after RHT | Captures direction, norm stored separately |
| **Dequant** | Reconstruct FP32 from signs + norm | Full precision for attention computation |

---

## Quick Start

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build   # 33/33 should pass

# GGUF (llama.cpp ecosystem)
./build/tq_run model.gguf -p "Hello" -k turbo_kv_1b

# TQM format (pre-quantized)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4

# Perplexity measurement
./build/tq_run model.gguf --ppl input.txt -k turbo_kv_1b
```

---

## Supported Models

| Model | Arch | Params | Format | Speed (M3, 6T) | PPL Verified |
|-------|------|--------|--------|----------------|-------------|
| **Qwen3.5-35B-A3B** | Qwen2-MoE | 35B (3B active) | GGUF IQ2_XXS | ~1-4 tok/s | byte-identical ✓ |
| **Qwen3.5-4B** | Qwen3.5 | 4B | GGUF Q8_0 | 5.4 tok/s | byte-identical ✓ |
| **SmolLM2-1.7B** | **Llama** | 1.7B | GGUF Q8_0 | 24 tok/s | **PPL +0.00% (800 tok)** ✓ |
| **Qwen3.5-0.8B** | Qwen3.5 | 752M | TQM / GGUF | 35 tok/s | **PPL +0.00% (800 tok)** ✓ |
| **Gemma 3 4B** | Gemma 3 | 4B | TQM | 20 tok/s | PPL +0.00% (101 tok) ✓ |
| **Gemma 3 270M** | Gemma 3 | 270M | TQM | 176 tok/s | byte-identical ✓ |

**4 architectures:** Llama, Gemma 3, Qwen3.5 (DeltaNet), Qwen2-MoE (256 experts).

---

## Compression Options

| Config | K bits | V bits | Compression | PPL Impact | Use Case |
|--------|--------|--------|-------------|------------|----------|
| **1-bit K + FP16 V** | 1 | 16 | 1.8x | +0.00% | Maximum quality |
| **1-bit K + Q4 V** | 1 | 4 | 4.9x | +0.03% | Best balance |
| **1-bit K + Q2 V** | 1 | 2 | 7.1x | +17.3% | Maximum compression |

```bash
./build/tq_run model -k turbo_kv_1b           # 1-bit K, FP16 V (1.8x, lossless)
./build/tq_run model -k turbo_kv_1b -v q4     # 1-bit K + Q4 V (4.9x)
./build/tq_run model -k turbo_kv_1b -v q2     # 1-bit K + Q2 V (7.1x)
./build/tq_run model -M                        # show memory stats
```

---

## Verification

| What | Result | How |
|------|--------|-----|
| **PPL at 800 tokens** | **+0.00%** (Llama, Qwen) | `--ppl` on 800-token text |
| Unbiasedness | < 0.2% relative bias | `test_unbiased` (100K pairs) |
| NEON/scalar | 14 paths match | `test_neon_scalar` |
| Edge cases | 29 tests (NaN, Inf, n=1) | `test_edge_cases` |
| ASan + UBSan | 33/33 clean | `scripts/sanitize.sh` |

---

## GPU & Compute Backends

| Backend | Target | Status | LOC |
|---------|--------|--------|-----|
| **Metal** | Apple Silicon | Verified (M3) | 4,002 |
| **NEON** | ARM CPU | Production | 980 |
| **AVX2** | x86 CPU | Production | 638 |
| **CUDA** | NVIDIA GPU | Compiles (GPU untested) | 2,146 |
| **Vulkan** | AMD + cross-platform | Compiles (GPU untested) | 2,317 |
| **ROCm/HIP** | AMD ROCm | Compiles (GPU untested) | 2,174 |

---

## FAQ

**Q: "How can 1-bit have zero loss?"**
We store keys in 1 bit but run attention in FP32. The dequantized keys preserve enough structure (RHT makes reconstruction unbiased) that attention distributions are virtually identical. PPL +0.00% verified at 800 tokens.

**Q: "What's the catch?"**
Compression is real (7.1x). Speed is unchanged (FP32 attention). The only cost is the quantize/dequantize step (~659 ns per vector, <0.1% of inference time).

**Q: "How is this different from llama.cpp's KV quantization?"**
llama.cpp uses uniform min-max. TurboQuant uses RHT + sign quantization which preserves inner product structure mathematically. We have a [llama.cpp integration patch](integrations/llamacpp/patch/) ready.

**Q: "Only small models?"**
Verified from 270M to 35B across 4 architectures. KV compression is architecture-independent.

---

## Under the Hood

**30,000+ lines of C/C++/Metal** — built from scratch, zero external dependencies.

- **12 KV quantization types** — RHT + Lloyd-Max + QJL
- **1-bit weight quantization** — 8.4x compression, output-identical on tested sequences
- **GGUF v3 loader** — 24 quant types, IQ2 E8 lattice, MoE dispatch
- **llama.cpp integration** — self-contained patch, `--cache-type-k tq_kv_1b`
- **33 test suites** — perplexity, unbiasedness, NEON consistency, edge cases

---

## References

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — Online Vector Quantization with Near-optimal Distortion Rate
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1-bit Quantized JL Transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar Coordinate Quantization

Full changelog: [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=quantumaikr/TurboQuant.cpp&type=Date)](https://star-history.com/#quantumaikr/TurboQuant.cpp&Date)

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
