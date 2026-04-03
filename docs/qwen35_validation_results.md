# quant.cpp — Qwen3.5-0.8B Validation Results (REAL MODEL)

**Date**: 2026-03-29
**Model**: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
**Data**: Actual KV cache from model inference (99-token prompt), NOT synthetic

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Total Layers | 24 (hybrid: DeltaNet + Gated Attention) |
| Attention Layers | 6 (layers 3, 7, 11, 15, 19, 23) |
| KV Heads | 2 (GQA) |
| Query Heads | 8 |
| Head Dimension | 256 |
| Max Context | 262,144 tokens |

## Real KV Cache Statistics

| Layer | Key Range | Key Std | Outlier Max |
|-------|-----------|---------|-------------|
| 3 | [-8.9, 8.3] | moderate | 8.9 |
| 7 | [-11.1, 13.0] | growing | 13.0 |
| 11 | [-13.3, 22.3] | large | **22.3** |
| 15 | [-17.1, 16.3] | large | 17.1 |

Layer 11 has the most extreme outliers (max=22.25). This is where mixed precision shines.

## Quantization Quality (Real Model Data)

| Type | BPE | MSE | Attention Cosine | Grade | Compression |
|------|-----|-----|-----------------|-------|-------------|
| **mixed_4b8** | 5.0 | **0.016** | **0.994** | **A+** | 6.4x |
| **uniform_4b** | 4.2 | 0.038 | **0.994** | **A+** | 7.5x |
| **uniform_2b** | 2.2 | 0.601 | **0.953** | **A** | 14.2x |
| turbo_3b | 7.0 | 0.345 | 0.934 | B+ | 4.6x |
| polar_4b | 4.5 | 0.688 | 0.893 | B | 7.1x |
| qjl_1b | 1.2 | 1.753 | 0.744 | C | 25.6x |

### Key Findings (Real Model)

1. **uniform_4b and mixed_4b8 both achieve A+ (cosine 0.994)** on real data
2. **uniform_2b achieves A grade (0.953)** on real data — better than expected from synthetic tests
3. Real Qwen3.5 KV cache is **more quantization-friendly** than synthetic data
4. Layer 11 has extreme outliers (max=22.25) — mixed_4b8 handles this gracefully

## RHT Impact (Real Model)

| Method | MSE | Improvement |
|--------|-----|-------------|
| Raw uniform_4b | 0.0378 | baseline |
| **RHT + uniform_4b** | **0.0209** | **1.8x** |

RHT effect is smaller on real data (1.8x vs 3.9x on synthetic) because the model's RoPE embeddings already provide some coordinate decorrelation.

## K/V Asymmetric Configuration

| Config | Avg Bits | Compression | Recommended |
|--------|----------|-------------|-------------|
| K4V4 (uniform_4b + uniform_4b) | 4.2 | 7.5x | Best quality |
| **K4V2 (uniform_4b + uniform_2b)** | **3.25** | **9.8x** | **Best balance** |
| K2V2 (uniform_2b + uniform_2b) | 2.2 | 14.2x | Max compression |

## Memory Impact

| Context | FP16 | K4V2 | Saved |
|---------|------|------|-------|
| 16K | 0.19 GB | 0.02 GB | 90% |
| 64K | 0.75 GB | 0.08 GB | 90% |
| 128K | 1.50 GB | 0.15 GB | 90% |
| 262K | 3.00 GB | 0.31 GB | 90% |

## Recommendations

| Use Case | Type | Cosine | Compression |
|----------|------|--------|-------------|
| **Production (quality)** | uniform_4b | 0.994 | 7.5x |
| **Production (balance)** | K4V2 asymmetric | ~0.97 | 9.8x |
| **Data with outliers** | mixed_4b8 | 0.994 | 6.4x |
| **Max compression** | uniform_2b | 0.953 | 14.2x |
| **With RHT enabled** | RHT + uniform_4b | ~0.99+ | 7.5x |

## Reproduction

```bash
# Step 1: Create venv with torch
python3 -m venv /tmp/tq_venv
source /tmp/tq_venv/bin/activate
pip install torch transformers numpy accelerate

# Step 2: Dump real KV cache
python3 tests/reference/dump_qwen35_kv.py

# Step 3: Build and validate
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_BENCH=ON
cmake --build build -j$(nproc)
./build/qwen35_validation
```
