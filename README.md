# TurboQuant.cpp

**Pure C inference engine with [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV cache compression.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Tests](https://img.shields.io/badge/tests-31%20pass-brightgreen)]()
[![ASan](https://img.shields.io/badge/ASan%2BUBSan-clean-brightgreen)]()

```
Qwen3.5-35B-A3B MoE on 16GB Mac:
  FP32 KV → max 32K context
  TurboQuant 1b K + Q2 V → 131K context  (18.6x KV compression)

Gemma 3 4B perplexity (101 tokens, teacher-forced):
  FP16 KV:         PPL = 35.99
  1-bit K + Q4 V:  PPL = 36.00  (+0.03%)
```

---

## Quick Start

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# TQM format (pre-converted)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4

# GGUF format (llama.cpp models directly)
./build/tq_run model.gguf -p "Hello" -k turbo_kv_1b -v q4
```

---

## Supported Models

| Model | Params | Format | Speed | KV Compression |
|-------|--------|--------|-------|----------------|
| **Qwen3.5-35B-A3B** | 35B (3B active) | GGUF | 0.5 tok/s | 18.6x (1b K + Q2 V) |
| **Gemma 3 4B** | 4B | TQM | 20.2 tok/s | 4.9x–7.1x |
| **Qwen3.5-0.8B** | 752M | TQM/GGUF | 80.1 tok/s | 4.9x–7.1x |
| **Gemma 3 270M** | 270M | TQM | 176 tok/s | 4.9x–7.1x |

Architectures: Gemma 3 (sliding window, GeGLU), Qwen3.5 (DeltaNet hybrid), Qwen2-MoE (top-K routing, shared expert).

---

## KV Compression

Keys are compressed via RHT + sign hashing (1-bit) or Lloyd-Max codebook (3/4-bit).
Values are independently quantized to Q4 or Q2.

```bash
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4   # 4.9x total K+V
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q2   # 7.1x total K+V
./build/tq_run model.tqm -p "Hello" -k turbo_kv_3b          # 3-bit keys, FP16 values
./build/tq_run model.tqm -p "Hello" -M                       # show memory stats
```

| Config | K+V/token (Gemma 4B) | Compression | PPL impact |
|--------|---------------------|-------------|------------|
| FP16 K+V | 136.00 KB | 1.0x | reference |
| 1-bit K + FP16 V | 74.38 KB | 1.8x | +0.00% |
| 1-bit K + Q4 V | 27.62 KB | 4.9x | +0.03% |
| 1-bit K + Q2 V | 19.12 KB | 7.1x | +17.3% |

> K-only quantization (V as FP16) is perplexity-lossless.
> Q4 V adds +0.03% PPL — effectively zero. Q2 V degrades noticeably.

---

## The Algorithm

```
Keys:   key → L2 normalize → RHT → Lloyd-Max codebook (b-1 bits) → QJL signs (1 bit)
        1-bit: signs only → attention via XOR + popcount

Values: value → per-block Q4 or Q2 quantization → fused accumulation from packed nibbles
```

The [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026) proves that standard quantizers introduce systematic bias in inner product estimation. RHT + QJL correction makes the estimator provably unbiased.

---

## Analysis Tools

```bash
./build/tq_run model --ppl input.txt -k turbo_kv_1b -v q4   # perplexity
./build/tq_run model --profile-kv -k turbo_kv_1b -p "text"  # activation distribution
./build/tq_run model --recommend -k turbo_kv_1b -p "text"   # per-layer bit allocation
./build/tq_run model --calibrate -k turbo_kv_1b -p "text"   # codebook calibration
./build/tq_run model --attn-entropy -k turbo_kv_1b -p "text" # attention entropy
bash bench/auto_profile.sh model                              # full pipeline
```

---

## Verification

| What | Result | How to reproduce |
|------|--------|------------------|
| Perplexity (1b K + Q4 V) | PPL +0.03% vs FP16 | `--ppl` on Gemma 4B |
| Unbiasedness | < 0.2% relative bias, 100K samples | `test_unbiased` |
| Attention cosine (1-bit) | 0.634 = 2/pi theoretical limit | `test_attention_distribution` |
| Lloyd-Max codebook | MSE within 1.18x of info-theoretic optimal | `test_codebook_theory` |
| Codebook calibration | 49.7% MSE improvement on real activations | `--calibrate` |
| Cumulative error (16 layers) | cosine 0.998 (Q4), sub-linear growth | `test_cumulative_error` |
| NEON/scalar consistency | 14 paths verified | `test_neon_scalar` |
| Edge cases | 29 tests (NaN, Inf, n=1, dim=0) | `test_edge_cases` |
| ASan + UBSan | 31/31 clean | `scripts/sanitize.sh` |
| Rate-distortion gap | Q4: 2.41x vs lower bound | `test_rate_distortion` |

Benchmark scripts: `bench/ablation_test.sh`, `bench/kv_quality_bench.sh`, `bench/long_quality_test.sh`, `bench/sampling_test.sh`

---

## FAQ

**Q: "Is 1-bit attention cosine of 0.634 too low?"**
No. 2/pi = 0.637 is the information-theoretic maximum for sign-only quantization. Our 0.634 matches this limit. For higher cosine, use 3-bit (0.918).

**Q: "How is this different from llama.cpp's KV quantization?"**
llama.cpp uses uniform min-max. TurboQuant uses RHT + Lloyd-Max codebook with QJL residual correction, providing provably unbiased inner product estimation. Codebook centroids verified against theory (`test_codebook_theory`).

**Q: "What about perplexity?"**
Measured. Gemma 4B with 1-bit K + Q4 V: PPL = 36.00 vs 35.99 baseline (+0.03%). K-only quantization is exactly lossless (PPL identical). See `--ppl` flag.

**Q: "Is the NEON code correct?"**
Every NEON path verified against scalar reference (`test_neon_scalar`). A Q4 dequant nibble-interleaving bug was found and fixed during validation. ASan + UBSan clean on all 31 test suites.

**Q: "RHT overhead?"**
147 ns per 128-dim vector (NEON-vectorized). 1-bit attention: 1.2 ns/key. Compared to matmul (~1ms/layer), negligible. See `bench/bench_kv_overhead.cpp`.

**Q: "Only small models?"**
Qwen3.5-35B-A3B MoE runs on a 16GB Mac Air (RSS 4.7GB). GGUF direct loading supports Q2_K through Q6_K and IQ2 formats.

---

## Under the Hood

- **15,000+ lines of C** — zero external dependencies
- **GGUF v3 direct loading** — use llama.cpp models without conversion
- **MoE support** — top-K expert routing, shared expert, SwiGLU
- **12 KV quantization types** — Uniform, PolarQuant, QJL, TurboQuant, TurboQuant KV (1/3/4-bit)
- **Fused Q4 attention** — weighted sum directly from packed nibbles
- **Adaptive compression** — per-layer bit recommendation, codebook calibration
- **NEON vectorized** — matmul, attention, RHT, Hamming distance, Q4 dequant
- **31 test suites** — perplexity, unbiasedness, attention distribution, codebook theory, NEON consistency, edge cases, rate-distortion, cumulative error

---

## References

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — Online Vector Quantization with Near-optimal Distortion Rate
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1-bit Quantized JL Transform
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — Polar Coordinate Quantization

Full changelog: [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
