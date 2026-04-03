#!/usr/bin/env python3
"""
Generate binary test vectors for quant.cpp cross-platform verification.

For each quantization type, quantizes a deterministic input and saves:
  - Input float32 array
  - Quantized output bytes (matching C struct layout)

Output files are written to spec/test_vectors/:
  - polar_4b_d128.bin    PolarQuant 4-bit, head_dim=128
  - qjl_1b_d128.bin      QJL 1-bit, head_dim=128
  - uniform_4b_d128.bin   Uniform min-max 4-bit, head_dim=128

Binary format for each file:
  [header]
    uint32: magic = 0x54515456 ("TQTV")
    uint32: version = 1
    uint32: type_id (matches tq_type enum)
    uint32: head_dim
    uint32: n_vectors
    uint32: input_bytes (byte count of input float32 data)
    uint32: output_bytes (byte count of quantized output)
    uint32: reserved = 0
  [input]
    float32[n_vectors * head_dim] -- raw input vectors
  [output]
    uint8[output_bytes] -- quantized blocks, packed per C struct layout
"""

import os
import sys
import struct
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polar_quant_ref import polar_quantize, polar_pack_block, TQ_BK
from qjl_ref import (
    generate_projection_matrix, qjl_quantize, qjl_pack_block,
    TQ_SKETCH_DIM, TQ_OUTLIERS,
)

# Constants
MAGIC = 0x54515456  # "TQTV"
VERSION = 1

# Type IDs matching tq_type enum in tq_types.h
TYPE_POLAR_3B = 0
TYPE_POLAR_4B = 1
TYPE_QJL_1B = 2
TYPE_TURBO_3B = 3
TYPE_TURBO_4B = 4
TYPE_UNIFORM_4B = 5
TYPE_UNIFORM_2B = 6


def generate_deterministic_input(n_vectors: int, head_dim: int, seed: int = 12345) -> np.ndarray:
    """Generate reproducible input vectors."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_vectors, head_dim).astype(np.float32)


def write_header(f, type_id: int, head_dim: int, n_vectors: int,
                 input_bytes: int, output_bytes: int):
    """Write test vector file header."""
    f.write(struct.pack('<IIIIIIII',
                        MAGIC, VERSION, type_id, head_dim, n_vectors,
                        input_bytes, output_bytes, 0))


def generate_polar_4b(output_dir: str, head_dim: int = 128, n_vectors: int = 4):
    """Generate PolarQuant 4-bit test vector."""
    keys = generate_deterministic_input(n_vectors, head_dim)

    # PolarQuant 4-bit: rbits=2, tbits=2
    rbits, tbits = 2, 2
    quantized = polar_quantize(keys, rbits=rbits, tbits=tbits, block_size=TQ_BK)

    # Pack all blocks for all vectors
    D = head_dim // 2
    pairs_per_block = TQ_BK // 2
    num_blocks = (D + pairs_per_block - 1) // pairs_per_block

    output_blocks = b''
    for vi in range(n_vectors):
        for bi in range(num_blocks):
            output_blocks += polar_pack_block(quantized, vi, bi, TQ_BK)

    input_bytes = keys.nbytes
    output_bytes = len(output_blocks)

    filepath = os.path.join(output_dir, 'polar_4b_d128.bin')
    with open(filepath, 'wb') as f:
        write_header(f, TYPE_POLAR_4B, head_dim, n_vectors, input_bytes, output_bytes)
        f.write(keys.tobytes())
        f.write(output_blocks)

    print(f"  Written: {filepath} ({32 + input_bytes + output_bytes} bytes)")
    print(f"    Input: {n_vectors} x {head_dim} float32 = {input_bytes} bytes")
    print(f"    Output: {n_vectors} x {num_blocks} blocks = {output_bytes} bytes")

    return filepath


def generate_qjl_1b(output_dir: str, head_dim: int = 128, n_vectors: int = 4):
    """Generate QJL 1-bit test vector."""
    keys = generate_deterministic_input(n_vectors, head_dim)

    projection = generate_projection_matrix(head_dim, TQ_SKETCH_DIM, seed=42)
    quantized = qjl_quantize(keys, projection, n_outliers=TQ_OUTLIERS)

    output_blocks = b''
    for vi in range(n_vectors):
        output_blocks += qjl_pack_block(quantized, vi)

    input_bytes = keys.nbytes
    output_bytes = len(output_blocks)

    filepath = os.path.join(output_dir, 'qjl_1b_d128.bin')
    with open(filepath, 'wb') as f:
        write_header(f, TYPE_QJL_1B, head_dim, n_vectors, input_bytes, output_bytes)
        f.write(keys.tobytes())
        f.write(output_blocks)

    print(f"  Written: {filepath} ({32 + input_bytes + output_bytes} bytes)")
    print(f"    Input: {n_vectors} x {head_dim} float32 = {input_bytes} bytes")
    print(f"    Output: {n_vectors} blocks = {output_bytes} bytes")

    # Also save the projection matrix for C-side verification
    proj_filepath = os.path.join(output_dir, 'qjl_projection_d128_s256.bin')
    with open(proj_filepath, 'wb') as f:
        f.write(struct.pack('<II', head_dim, TQ_SKETCH_DIM))
        f.write(projection.tobytes())
    print(f"  Written: {proj_filepath} ({8 + projection.nbytes} bytes)")

    return filepath


def uniform_quantize_4b(keys: np.ndarray, block_size: int = TQ_BK) -> bytes:
    """
    Uniform 4-bit min-max quantization, matching block_tq_uniform_4b layout.

    Layout per block:
      - scale:      fp16 (2 bytes)
      - zero_point: fp16 (2 bytes)
      - qs:         BK/2 bytes (2 values per byte, LSB-first)
    """
    n, head_dim = keys.shape
    num_blocks = (head_dim + block_size - 1) // block_size

    output = b''
    for vi in range(n):
        for bi in range(num_blocks):
            start = bi * block_size
            end = min(start + block_size, head_dim)
            block_data = keys[vi, start:end]

            # Pad if needed
            if len(block_data) < block_size:
                block_data = np.pad(block_data, (0, block_size - len(block_data)),
                                    constant_values=0)

            mn = float(block_data.min())
            mx = float(block_data.max())
            scale = (mx - mn) / 15.0  # 4-bit: 2^4 - 1 = 15
            if scale == 0:
                scale = 1.0

            # Quantize
            q_vals = np.clip(np.round((block_data - mn) / scale), 0, 15).astype(np.uint8)

            # Pack 2 values per byte, LSB-first
            n_packed = block_size // 2
            packed = np.zeros(n_packed, dtype=np.uint8)
            for j in range(n_packed):
                lo = q_vals[2 * j]
                hi = q_vals[2 * j + 1]
                packed[j] = (hi << 4) | lo  # LSB-first: low nibble first

            output += struct.pack('<e', np.float16(scale))
            output += struct.pack('<e', np.float16(mn))
            output += packed.tobytes()

    return output


def generate_uniform_4b(output_dir: str, head_dim: int = 128, n_vectors: int = 4):
    """Generate Uniform 4-bit test vector."""
    keys = generate_deterministic_input(n_vectors, head_dim)

    output_blocks = uniform_quantize_4b(keys, block_size=TQ_BK)

    input_bytes = keys.nbytes
    output_bytes = len(output_blocks)

    filepath = os.path.join(output_dir, 'uniform_4b_d128.bin')
    with open(filepath, 'wb') as f:
        write_header(f, TYPE_UNIFORM_4B, head_dim, n_vectors, input_bytes, output_bytes)
        f.write(keys.tobytes())
        f.write(output_blocks)

    print(f"  Written: {filepath} ({32 + input_bytes + output_bytes} bytes)")
    print(f"    Input: {n_vectors} x {head_dim} float32 = {input_bytes} bytes")
    print(f"    Output: {n_vectors} blocks = {output_bytes} bytes")

    return filepath


def verify_test_vector(filepath: str):
    """Read back and verify a test vector file."""
    with open(filepath, 'rb') as f:
        magic, version, type_id, head_dim, n_vectors, input_bytes, output_bytes, reserved = \
            struct.unpack('<IIIIIIII', f.read(32))

        assert magic == MAGIC, f"Bad magic: 0x{magic:08X}"
        assert version == VERSION, f"Bad version: {version}"
        assert reserved == 0, f"Bad reserved: {reserved}"

        input_data = f.read(input_bytes)
        output_data = f.read(output_bytes)

        assert len(input_data) == input_bytes, "Input data truncated"
        assert len(output_data) == output_bytes, "Output data truncated"

        keys = np.frombuffer(input_data, dtype=np.float32).reshape(n_vectors, head_dim)

        print(f"  Verified: {filepath}")
        print(f"    type={type_id}, head_dim={head_dim}, n={n_vectors}")
        print(f"    input_bytes={input_bytes}, output_bytes={output_bytes}")
        print(f"    keys range: [{keys.min():.4f}, {keys.max():.4f}]")


def main():
    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    output_dir = os.path.join(project_root, 'spec', 'test_vectors')
    os.makedirs(output_dir, exist_ok=True)

    print("=== Generating quant.cpp Test Vectors ===\n")

    head_dim = 128
    n_vectors = 4

    print("1. PolarQuant 4-bit (rbits=2, tbits=2):")
    f1 = generate_polar_4b(output_dir, head_dim, n_vectors)

    print("\n2. QJL 1-bit:")
    f2 = generate_qjl_1b(output_dir, head_dim, n_vectors)

    print("\n3. Uniform 4-bit (baseline):")
    f3 = generate_uniform_4b(output_dir, head_dim, n_vectors)

    print("\n=== Verification ===\n")
    for f in [f1, f2, f3]:
        verify_test_vector(f)

    print("\nAll test vectors generated and verified successfully.")


if __name__ == '__main__':
    main()
