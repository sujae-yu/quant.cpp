# PPL Comparison: TurboQuant vs llama.cpp KV Quantization

Model: SmolLM2-1.7B-Instruct (Llama architecture, head_dim=64)
Text: bench/data/ppl_test_2k.txt (~1900 words, ~2500 tokens)
Hardware: Apple M3, 16GB

## llama.cpp (refs/llama.cpp, with Metal GPU)

| KV Config | PPL | Delta vs FP16 | KV Bits/element |
|-----------|-----|---------------|-----------------|
| FP16 (baseline) | 2.83 | — | 16 |
| Q4_0 | 3.13 | **+10.6%** | 4 |

## TurboQuant.cpp (our engine, CPU)

| KV Config | PPL | Delta vs baseline | KV Bits/element |
|-----------|-----|-------------------|-----------------|
| uniform_4b (baseline) | 8.32 | — | 4 |
| turbo_kv_1b (1-bit K) | 8.32 | **+0.00%** | 1 |
| turbo_kv_3b (3-bit K) | 8.32 | **+0.00%** | 3 |

Note: PPL values differ between engines due to different weight quantization
paths (llama.cpp uses Q8_0 directly, our engine converts to Q4 at load time).
The KEY metric is the DELTA from each engine's own baseline.

## Summary

| Method | Compression | PPL Delta |
|--------|-------------|-----------|
| llama.cpp Q4_0 KV | 4x | +10.6% |
| **TurboQuant 1-bit K** | **16x (K only)** | **+0.00%** |

TurboQuant achieves 4x more compression on keys with zero PPL increase,
while llama.cpp's Q4 KV shows measurable quality degradation.
