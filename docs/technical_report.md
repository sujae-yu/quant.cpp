# quant.cpp: Practical 1-bit KV Cache Compression for LLM Inference

## Abstract

quant.cpp is a self-contained C implementation of quant.cpp (ICLR 2026) for KV cache compression in large language model inference. The library implements randomized Hadamard transforms, Lloyd-Max codebooks, and quantized Johnson-Lindenstrauss sign hashing to compress KV cache keys to as few as 1 bit per element. On models ranging from Gemma 270M to Qwen 35B MoE, 1-bit key compression produces byte-identical greedy output compared to uncompressed baselines. Combined with Q4 value quantization, the system achieves 4.9x total K+V compression with +0.03% perplexity degradation on Gemma 4B. The implementation is 28K lines of C with zero external dependencies, loads GGUF models directly, and includes 31 test suites verified under AddressSanitizer.

## 1. Introduction

KV cache memory grows linearly with sequence length, batch size, and model depth, making it the primary memory bottleneck for long-context LLM inference. At 32K context length, a 4B-parameter model requires over 4 GB for FP16 KV storage per layer stack.

Standard quantization methods (uniform, NF4) minimize reconstruction MSE. However, the KV cache is consumed exclusively through inner products during attention. The quant.cpp paper (Zandieh et al., ICLR 2026) demonstrates that MSE-optimal quantizers introduce systematic bias in inner product estimation, and proposes an alternative two-stage approach that is provably unbiased at any bit-width.

This work implements and verifies quant.cpp on real models up to 35B parameters, providing a practical inference engine with no Python or framework dependencies.

## 2. Method

**Randomized Hadamard Transform (RHT).** Before quantization, key vectors are multiplied by a Walsh-Hadamard matrix with random sign flips. This decorrelates channels and Gaussianizes the activation distribution. Measured effect: kurtosis drops from 10-99 (raw) to 3.9-7.9 (post-RHT). The transform is its own inverse.

**Lloyd-Max Codebook.** For 2-4 bit quantization, optimal scalar codebooks are computed for the post-RHT (approximately Gaussian) distribution. Measured MSE is within 1.18x of the information-theoretic optimum. Online calibration on model-specific activations yields 49.7% MSE improvement over the default N(0,1) codebook.

**QJL Sign Hash.** Quantized Johnson-Lindenstrauss projection stores only the sign of random projections (1 bit each). The resulting inner product estimator is unbiased. At 1 bit, attention reduces to XOR + population count -- two instructions per 128-dimensional key.

**1-bit Extreme.** When using only signs after RHT, the expected attention cosine similarity is 2/pi = 0.637. Despite this distortion, the relative ranking of attention scores is preserved sufficiently that greedy decoding produces identical output in practice.

## 3. Implementation

The implementation comprises 28K lines of C/C++/Metal with zero external dependencies (libc/libm only for the core library).

**Model loading.** Safetensors (multi-shard, mmap) and GGUF (Q8_0, Q4_K_M, IQ2_XXS verified). A TQM binary format supports pre-quantized instant loading.

**Architectures.** Three model families: Gemma 3 (sliding window, GeGLU, dual RoPE), Qwen3.5 (DeltaNet + self-attention hybrid, GQA), and Qwen2-MoE (256 experts, top-8 routing, shared expert).

**Quantization types.** 11 KV cache types including uniform 4b/2b, PolarQuant 3b/4b, QJL 1b, quant.cpp 1b/3b/4b, and mixed configurations. Value quantization supports Q4 (per-block scale + packed nibbles) and Q2 (Lloyd-Max codebook).

**Compute backends.** CPU generic (C reference), CPU NEON (vectorized Hadamard butterfly, Hamming distance via `vcntq_u8`, Q4 dequant via `vzip_u8`), Metal (verified on Apple Silicon), and CUDA/Vulkan/ROCm (compile-tested, not hardware-verified).

**Testing.** 31 test files covering roundtrip accuracy, attention distribution, codebook theory, edge cases, NEON/scalar consistency, unbiasedness, and cumulative error. All pass under ASan + UBSan.

## 4. Results

All numbers below are from actual measurements on the systems described. No estimated or projected values are included.

### Table 1: KV Compression Quality

| Model | Config | K+V Compression | PPL Impact |
|-------|--------|-----------------|------------|
| Gemma 4B | 1-bit K + FP16 V | 1.8x (K only) | +0.00% (byte-identical greedy) |
| Gemma 4B | 1-bit K + Q4 V | 4.9x | +0.03% (PPL 36.00 vs 35.99) |
| Gemma 4B | 1-bit K + Q2 V | 7.1x | +17.3% (PPL 42.23 vs 35.99) |
| Qwen 0.8B | 1-bit K (GGUF Q8) | 1.8x (K only) | byte-identical (100 tokens) |
| Qwen 35B MoE | 1-bit K (GGUF IQ2) | 1.8x (K only) | byte-identical (greedy) |

Byte-identical means the greedy-decoded token sequence matches the uncompressed baseline exactly. Verified on 30 diverse prompts for Gemma 4B.

### Table 2: Theoretical vs Measured

| Metric | Theory | Measured | Source |
|--------|--------|----------|--------|
| 1-bit attention cosine | 2/pi = 0.637 | 0.634 | test_attention_distribution |
| Unbiasedness | bias -> 0 | < 0.2% relative | test_unbiased (100K pairs) |
| Lloyd-Max MSE | 1.175x optimal | < 1.18x | test_codebook_theory |
| Codebook calibration | -- | 49.7% MSE gain | --calibrate |
| Rate-distortion gap (Q4) | -- | 2.41x | test_rate_distortion |
| 16-layer cumulative cosine (Q4) | -- | 0.998 | test_cumulative_error |
| Random K cosine (control) | ~0 | 0.089 | test_attention_distribution |

### Table 3: Weight Quantization

| Method | Compression | Quality |
|--------|-------------|---------|
| 1-bit sign hash (weights) | 8.4x vs Q8 | byte-identical greedy (Gemma 4B) |
| Q4+Q2 progressive residual | 6-bit effective | cosine 0.999 |
| Q4 per-block | 4x vs FP16 | standard |
| Q2 Lloyd-Max codebook | 8x vs FP16 | integer Q2xQ8 matmul |

### Table 4: Inference Speed (Apple M3, 6 threads)

| Model | Params | Format | Speed |
|-------|--------|--------|-------|
| Gemma 270M | 270M | TQM Q4 | 176 tok/s |
| Qwen3.5-0.8B | 752M | TQM Q4 | 80 tok/s |
| Qwen3.5-0.8B | 752M | GGUF Q8 | 35 tok/s |
| Gemma 3 4B | 4B | TQM Q4 | 20 tok/s |
| Qwen3.5-35B MoE | 35B (3B active) | GGUF IQ2 | 1-4 tok/s |

### Table 5: Real-Model KV Quality (Qwen3.5-0.8B)

| Type | BPE | MSE | Attention Cosine |
|------|-----|-----|-----------------|
| mixed_4b8 | 5.0 | 0.016 | 0.994 |
| uniform_4b | 4.2 | 0.038 | 0.994 |
| uniform_2b | 2.2 | 0.601 | 0.953 |
| turbo_3b | 7.0 | 0.345 | 0.934 |
| qjl_1b | 1.2 | 1.753 | 0.744 |

## 5. llama.cpp Integration

A self-contained patch adds quant.cpp KV cache support to llama.cpp:

- **4 files, 874 lines**: `ggml-turbo-quant.h`, `ggml-turbo-quant.c`, `tq_kv_cache.cpp`, `test_turbo_quant_kv.cpp`
- **Usage**: `--cache-type-k tq_kv_1b` (drop-in flag)
- **No modifications** to existing llama.cpp source files required
- Includes 7 standalone integration tests
- Results match the standalone engine: 1-bit cosine = 0.634 (= 2/pi)

The patch registers a custom GGML type and intercepts KV cache allocation to use quant.cpp's quantize/attention kernels.

## 6. Limitations and Future Work

**Speed.** Inference is CPU-bound. Metal GPU dispatch works for dense models but MoE expert routing on Metal is work-in-progress. The 35B MoE model runs at 1-4 tok/s, limited by memory bandwidth on 16GB unified memory.

**Model coverage.** Verified on 3 architectures (Qwen3.5, Gemma 3, Qwen2-MoE). Other architectures (Llama, Mistral, Phi) are untested.

**GPU backends.** CUDA, Vulkan, and ROCm backends compile but have not been tested on actual hardware. Metal is verified on Apple Silicon.

**Perplexity evaluation.** PPL measurements use 101-token sequences. Longer evaluation sequences (512-2048 tokens) on standard benchmarks would provide stronger quality evidence.

**Value compression.** Q2 value quantization degrades PPL by 17.3%. The Q4 value path (+0.03%) is the practical choice. Adaptive per-layer bit allocation (measured average 2.0 bits via kurtosis profiling) is implemented but not yet integrated into the inference loop.

**Post-RHT distribution.** The RHT reduces kurtosis substantially (to 3.9-7.9) but the result is not perfectly Gaussian. Lloyd-Max codebooks calibrated to the actual post-RHT distribution outperform the N(0,1) default by 49.7%.

## 7. Conclusion

quant.cpp demonstrates that 1-bit KV cache key compression can produce output identical to uncompressed baselines on models from 270M to 35B parameters. Combined with Q4 value quantization, total K+V compression reaches 4.9x with negligible quality loss (+0.03% PPL). The theoretical 2/pi cosine limit and unbiasedness properties predicted by the quant.cpp paper are confirmed empirically.

The implementation is practical: it loads standard GGUF models, requires no calibration data for the default configuration, and integrates with llama.cpp via a 874-line patch. All code is open source under Apache 2.0.

**Repository:** https://github.com/quantumaikr/quant.cpp

**References:**

- Zandieh et al., "quant.cpp: Online KV Cache Quantization via Unbiased Inner Product Preserving Transform," ICLR 2026
- Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead," AAAI 2025
- Xu et al., "PolarQuant: Polar Coordinate Quantization for KV Cache Compression," AISTATS 2026
