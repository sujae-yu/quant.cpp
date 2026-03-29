#!/usr/bin/env python3
"""Dump real KV cache from a small LLM for TurboQuant validation.

Generates binary files with format:
  Header: magic(4B) + layer_idx(4B) + num_heads(4B) + seq_len(4B) + head_dim(4B) = 20 bytes
  Data:   num_heads * seq_len * head_dim * sizeof(float32)

If transformers is available, uses Qwen/Qwen2.5-0.5B-Instruct.
Otherwise, generates realistic synthetic data that mimics real LLM KV cache statistics.
"""

import numpy as np
import struct
import os

# Try to use transformers, fall back to synthetic-but-realistic if not available
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("transformers not available, generating realistic synthetic data")

MAGIC = 0x544B5651  # "QVKT" in little-endian
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../../spec/test_vectors/real_kv")


def write_kv_binary(fname, layer_idx, data):
    """Write a KV tensor to binary file.

    Args:
        fname: output file path
        layer_idx: layer index for the header
        data: numpy array of shape [num_heads, seq_len, head_dim], float32
    """
    num_heads, seq_len, head_dim = data.shape
    with open(fname, "wb") as f:
        f.write(struct.pack("<5I", MAGIC, layer_idx, num_heads, seq_len, head_dim))
        for h in range(num_heads):
            for s in range(seq_len):
                f.write(data[h, s, :].astype(np.float32).tobytes())


def dump_real_kv():
    """Dump KV cache from Qwen2.5-0.5B or generate synthetic equivalent."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if HAS_TRANSFORMERS:
        dump_from_model()
    else:
        generate_realistic_synthetic()


def dump_from_model():
    """Load a real model and capture KV cache tensors."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    prompt = (
        "The key to efficient LLM inference is quantizing the key-value cache, "
        "which reduces memory usage while preserving attention accuracy. "
        "Recent research has shown that"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values

    num_layers = len(past_kv)
    layers_to_dump = min(4, num_layers)
    print(f"Model has {num_layers} layers, dumping first {layers_to_dump}")

    for layer_idx in range(layers_to_dump):
        keys = past_kv[layer_idx][0].squeeze(0).numpy()    # [num_heads, seq_len, head_dim]
        values = past_kv[layer_idx][1].squeeze(0).numpy()

        num_heads, seq_len, head_dim = keys.shape
        print(f"Layer {layer_idx}: heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
        print(f"  Key  stats: mean={keys.mean():.4f}, std={keys.std():.4f}, "
              f"min={keys.min():.4f}, max={keys.max():.4f}")
        print(f"  Value stats: mean={values.mean():.4f}, std={values.std():.4f}, "
              f"min={values.min():.4f}, max={values.max():.4f}")

        fname_k = os.path.join(OUTPUT_DIR, f"layer{layer_idx}_keys.bin")
        write_kv_binary(fname_k, layer_idx, keys)

        fname_v = os.path.join(OUTPUT_DIR, f"layer{layer_idx}_values.bin")
        write_kv_binary(fname_v, layer_idx, values)

        print(f"  Saved: {fname_k}")
        print(f"  Saved: {fname_v}")


def generate_realistic_synthetic():
    """Generate data that mimics real LLM KV cache statistics.

    Real LLM KV caches have these properties:
    - Per-channel variance varies significantly (some dims are "active", some quiet)
    - Distribution is near-Gaussian but with heavier tails
    - RoPE creates periodic structure and occasional outliers
    - Later layers tend to have more concentrated, less uniform distributions
    - Value tensors have smaller magnitude than key tensors
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)

    num_heads = 14   # Qwen2.5-0.5B has 14 KV heads (GQA)
    seq_len = 64
    head_dim = 64    # Qwen2.5-0.5B head_dim

    for layer_idx in range(4):
        # Per-channel standard deviation (log-normal mimics real variance patterns)
        channel_std = np.random.lognormal(
            mean=-1.5, sigma=0.8, size=head_dim
        ).astype(np.float32)
        # Later layers have more varied channel distributions
        channel_std *= (0.5 + 0.5 * layer_idx / 4.0)

        keys = np.zeros((num_heads, seq_len, head_dim), dtype=np.float32)
        for h in range(num_heads):
            for d in range(head_dim):
                keys[h, :, d] = np.random.normal(
                    0, channel_std[d], seq_len
                ).astype(np.float32)

            # Add sparse outliers (RoPE creates these in real models)
            n_outliers = max(1, seq_len // 16)
            outlier_pos = np.random.choice(seq_len, size=n_outliers, replace=False)
            outlier_dim = np.random.choice(head_dim, size=2, replace=False)
            for p in outlier_pos:
                for od in outlier_dim:
                    keys[h, p, od] *= 5.0

        print(f"Layer {layer_idx}: {num_heads}h x {seq_len}seq x {head_dim}d")
        print(f"  Key  stats: mean={keys.mean():.4f}, std={keys.std():.4f}, "
              f"min={keys.min():.4f}, max={keys.max():.4f}")

        fname_k = os.path.join(OUTPUT_DIR, f"layer{layer_idx}_keys.bin")
        write_kv_binary(fname_k, layer_idx, keys)

        # Value tensors: smaller magnitude, more uniform distribution
        values = np.random.normal(
            0, 0.1, (num_heads, seq_len, head_dim)
        ).astype(np.float32)

        print(f"  Value stats: mean={values.mean():.4f}, std={values.std():.4f}")

        fname_v = os.path.join(OUTPUT_DIR, f"layer{layer_idx}_values.bin")
        write_kv_binary(fname_v, layer_idx, values)

        print(f"  Saved: {fname_k}")
        print(f"  Saved: {fname_v}")


if __name__ == "__main__":
    dump_real_kv()
    print(f"\nDone! KV cache saved to {OUTPUT_DIR}")
