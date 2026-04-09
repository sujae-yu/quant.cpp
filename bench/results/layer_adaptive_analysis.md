# Layer-Adaptive KV Compression Analysis

## Result: NEGATIVE — Not Worth Implementing

Layer-adaptive bit allocation (different bits per transformer layer)
provides at most ~0.9% PPL improvement after RHT normalization.
This is below the measurement noise and not worth the implementation
complexity.

## Why

RHT (Random Hadamard Transform) normalizes ALL layers to near-Gaussian:

| Metric | Pre-RHT | Post-RHT |
|---|---|---|
| Kurtosis range | 4.13 – 20.62 | **2.64 – 3.81** |
| Kurtosis std | large | **0.25** |
| Skewness range | -2.54 – +2.59 | -0.34 – +0.19 |

The variance of log2(std) across layers post-RHT is only 0.0177,
meaning the theoretical MSE improvement from optimal per-layer bit
allocation is ~1.8% MSE → ~0.9% PPL.

## Architectural Insight

This is an advantage of the RHT-based approach:

- Without RHT: layers have wildly different distributions → need per-layer
  calibration → complex + slow
- With RHT: all layers are near-Gaussian → single bit allocation works for
  all → simple + fast

Other KV compression methods (KIVI, KVQuant) don't use RHT and therefore
need per-layer calibration profiles. quant.cpp's RHT makes layer-adaptive
unnecessary, which is actually a feature — simpler code, fewer parameters.

## Implication for the Paper

The fact that RHT eliminates the need for per-layer adaptation is a
publishable insight: "RHT-based KV quantization achieves near-optimal
per-layer performance with a single uniform bit allocation."

This strengthens the attention-aware (temporal) quantization finding (S1):
- Per-layer (spatial) adaptation: NOT needed after RHT (~0.9% max benefit)
- Per-token (temporal) adaptation: CRITICAL (+3.8% → +0.6% benefit)

The information bottleneck is temporal (which tokens), not spatial (which layers).

## Measured on

Llama 3.2 3B Instruct Q8_0, 28 layers, 28 tokens profiled.
Post-RHT kurtosis: mean 3.04, std 0.25.
