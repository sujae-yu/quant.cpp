#!/usr/bin/env python3
"""TurboQuant.cpp -- Python Quick Start

Demonstrates KV cache compression with the TurboQuant Python bindings.
Requires the shared library to be built first:

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../bindings/python"))
from turboquant import TurboQuant

tq = TurboQuant("cpu")

# Simulate KV cache from LLM (128-dim, 512 tokens)
np.random.seed(42)
keys = np.random.randn(512, 128).astype(np.float32) * 0.15
query = np.random.randn(128).astype(np.float32) * 0.15

# Quantize (compression with high accuracy)
quantized = tq.quantize_keys(keys, TurboQuant.UNIFORM_4B)
print(f"Original:   {keys.nbytes:,} bytes")
print(f"Compressed: {len(quantized):,} bytes ({keys.nbytes / len(quantized):.1f}x)")

# Compute attention on compressed cache
scores = tq.attention(query, quantized, 512, 128, TurboQuant.UNIFORM_4B)
fp32_scores = keys @ query
cosine = np.dot(scores, fp32_scores) / (
    np.linalg.norm(scores) * np.linalg.norm(fp32_scores) + 1e-10
)
print(f"Attention accuracy: {cosine:.4f} (1.0 = perfect)")

# Compare all available types
print("\nType comparison:")
for qtype in [TurboQuant.UNIFORM_4B, TurboQuant.UNIFORM_2B,
              TurboQuant.POLAR_4B, TurboQuant.POLAR_3B,
              TurboQuant.QJL_1B, TurboQuant.TURBO_3B, TurboQuant.TURBO_4B]:
    name = tq.type_name(qtype)
    bpe = tq.type_bpe(qtype)
    q = tq.quantize_keys(keys, qtype)
    deq = tq.dequantize_keys(q, 512, 128, qtype)
    mse = np.mean((keys - deq) ** 2)
    print(f"  {name:12s}  bpe={bpe:.1f}  compress={32/bpe:.1f}x  mse={mse:.6f}")
