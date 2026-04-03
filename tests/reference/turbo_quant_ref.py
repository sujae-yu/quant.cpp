#!/usr/bin/env python3
"""
quant.cpp NumPy reference implementation.

Combines PolarQuant and QJL into a two-stage quantization:
  Stage 1: PolarQuant quantizes keys at 2-bit or 3-bit (theta + rho)
  Stage 2: QJL 1-bit sign hash captures the residual error

Attention score = PolarQuant_score + QJL_residual_correction_score

Reference: quant.cpp concept from PolarQuant + QJL combination.
Block format matches include/turboquant/tq_types.h block_tq_turbo.
"""

import numpy as np
from typing import Optional

from polar_quant_ref import (
    polar_quantize,
    polar_dequantize,
    polar_attention_score,
    polar_pack_block,
    TQ_BK,
)
from qjl_ref import (
    generate_projection_matrix,
    qjl_quantize,
    qjl_attention_score,
    qjl_pack_block,
    TQ_SKETCH_DIM,
    TQ_OUTLIERS,
)


def turbo_quantize(
    keys: np.ndarray,
    rbits: int = 1,
    tbits: int = 2,
    d_sketch: int = TQ_SKETCH_DIM,
    n_outliers: int = TQ_OUTLIERS,
    projection_seed: int = 42,
    block_size: int = TQ_BK,
) -> dict:
    """
    quant.cpp two-stage quantization.

    Stage 1: PolarQuant with (rbits, tbits) for the primary signal.
    Stage 2: QJL 1-bit sign hash on the residual (original - dequantized).

    Total effective bits = rbits + tbits + 1 (QJL).

    Args:
        keys: Input key vectors, shape (n, head_dim), float32.
        rbits: Radius bits for PolarQuant stage.
        tbits: Theta bits for PolarQuant stage.
        d_sketch: QJL sketch dimension.
        n_outliers: QJL outlier count.
        projection_seed: Seed for QJL projection matrix.
        block_size: PolarQuant block size.

    Returns:
        Dictionary with:
          - 'polar': PolarQuant quantized output
          - 'residual_qjl': QJL quantized residual output
          - 'projection': QJL projection matrix
          - 'residual_keys': residual key vectors (for outlier correction)
    """
    n, head_dim = keys.shape

    # Stage 1: PolarQuant
    polar_q = polar_quantize(keys, rbits=rbits, tbits=tbits, block_size=block_size)

    # Compute residual: original - dequantized
    reconstructed = polar_dequantize(polar_q, head_dim=head_dim, block_size=block_size)
    residual = keys - reconstructed

    # Stage 2: QJL on residual
    projection = generate_projection_matrix(head_dim, d_sketch, seed=projection_seed)
    residual_qjl = qjl_quantize(residual, projection, n_outliers=n_outliers)

    return {
        'polar': polar_q,
        'residual_qjl': residual_qjl,
        'projection': projection,
        'residual_keys': residual,
        'rbits': rbits,
        'tbits': tbits,
    }


def turbo_attention_score(
    query: np.ndarray,
    quantized: dict,
    head_dim: int,
    block_size: int = TQ_BK,
) -> np.ndarray:
    """
    Compute attention scores from quant.cpp-compressed keys.

    score = PolarQuant_attention(query, polar_data)
           + QJL_attention(query, residual_data)

    Args:
        query: Query vector, shape (d_key,) or (nq, d_key).
        quantized: Output from turbo_quantize().
        head_dim: Key head dimension.
        block_size: Block size.

    Returns:
        Attention scores.
    """
    # PolarQuant contribution
    polar_scores = polar_attention_score(
        query, quantized['polar'], head_dim=head_dim, block_size=block_size
    )

    # QJL residual correction
    qjl_scores = qjl_attention_score(
        query,
        quantized['residual_qjl'],
        quantized['projection'],
        keys_orig=quantized['residual_keys'],
    )

    return polar_scores + qjl_scores


def turbo_pack_block(
    quantized: dict,
    vec_idx: int,
    block_idx: int = 0,
    block_size: int = TQ_BK,
) -> bytes:
    """
    Pack a quant.cpp block: PolarQuant block + QJL residual block.

    Layout matches block_tq_turbo = block_tq_polar + block_tq_qjl.
    """
    polar_bytes = polar_pack_block(quantized['polar'], vec_idx, block_idx, block_size)
    qjl_bytes = qjl_pack_block(quantized['residual_qjl'], vec_idx)
    return polar_bytes + qjl_bytes


def compute_attention_cosine_similarity(
    query: np.ndarray,
    keys: np.ndarray,
    rbits: int = 1,
    tbits: int = 2,
    block_size: int = TQ_BK,
) -> float:
    """
    Compute cosine similarity between FP32 and quant.cpp attention scores.
    """
    fp32_scores = query @ keys.T
    quantized = turbo_quantize(keys, rbits=rbits, tbits=tbits, block_size=block_size)
    turbo_scores = turbo_attention_score(query, quantized, head_dim=keys.shape[1], block_size=block_size)

    dot = np.sum(fp32_scores * turbo_scores)
    norm_a = np.sqrt(np.sum(fp32_scores ** 2))
    norm_b = np.sqrt(np.sum(turbo_scores ** 2))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return float(dot / (norm_a * norm_b))


# ---- Self-test ----
if __name__ == '__main__':
    np.random.seed(42)
    head_dim = 128
    n_keys = 16
    keys = np.random.randn(n_keys, head_dim).astype(np.float32)
    query = np.random.randn(head_dim).astype(np.float32)

    print("=== quant.cpp 3-bit (Polar 2b + QJL 1b) ===")
    quantized_3b = turbo_quantize(keys, rbits=1, tbits=2)
    turbo_scores_3b = turbo_attention_score(query, quantized_3b, head_dim=head_dim)
    cos_sim_3b = compute_attention_cosine_similarity(query, keys, rbits=1, tbits=2)
    print(f"Attention cosine similarity: {cos_sim_3b:.6f}")

    # Compare with PolarQuant-only at same bit budget
    from polar_quant_ref import compute_attention_cosine_similarity as polar_cos
    polar_only_cos = polar_cos(query, keys, rbits=1, tbits=2)
    print(f"PolarQuant-only (3b) cosine: {polar_only_cos:.6f}")
    improvement_3b = cos_sim_3b - polar_only_cos
    print(f"quant.cpp improvement: {improvement_3b:+.6f}")

    print("\n=== quant.cpp 4-bit (Polar 3b + QJL 1b) ===")
    quantized_4b = turbo_quantize(keys, rbits=1, tbits=3)
    cos_sim_4b = compute_attention_cosine_similarity(query, keys, rbits=1, tbits=3)
    print(f"Attention cosine similarity: {cos_sim_4b:.6f}")

    polar_only_cos_4b = polar_cos(query, keys, rbits=1, tbits=3)
    print(f"PolarQuant-only (4b) cosine: {polar_only_cos_4b:.6f}")
    improvement_4b = cos_sim_4b - polar_only_cos_4b
    print(f"quant.cpp improvement: {improvement_4b:+.6f}")

    # Pack block test
    block_bytes = turbo_pack_block(quantized_3b, vec_idx=0, block_idx=0)
    from polar_quant_ref import TQ_BK
    expected_polar = 8 + TQ_BK // 2
    expected_qjl = 4 + TQ_SKETCH_DIM // 8 + TQ_OUTLIERS
    expected_total = expected_polar + expected_qjl
    print(f"\nBlock size: {len(block_bytes)} bytes (expected {expected_total})")

    print("\nAll quant.cpp reference tests passed.")
