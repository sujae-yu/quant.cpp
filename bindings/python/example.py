#!/usr/bin/env python3
"""
quant.cpp -- CLI Wrapper Example

Demonstrates the subprocess-based Python bindings that call the quant binary.
No C FFI, no NumPy, no shared library -- just a model file and the quant binary.

Prerequisites:
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j$(nproc)

Usage:
    python3 bindings/python/example.py models/qwen3.5-0.8b.tqm
    TURBOQUANT_BIN=./build/quant python3 bindings/python/example.py model.gguf
"""

import sys
import os

# Allow running from project root or bindings/python/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from turboquant_cli import quant.cpp


def main():
    if len(sys.argv) < 2:
        print("Usage: python example.py <model_path> [kv_type]")
        print()
        print("  model_path  Path to .tqm, .safetensors, or .gguf model file")
        print("  kv_type     KV cache type (default: turbo_kv_1b)")
        print()
        print("Examples:")
        print("  python example.py models/qwen3.5-0.8b.tqm")
        print("  python example.py model.gguf turbo_kv_3b")
        sys.exit(1)

    model_path = sys.argv[1]
    kv_type = sys.argv[2] if len(sys.argv) > 2 else "turbo_kv_1b"

    # Initialize
    print(f"Loading model: {model_path}")
    print(f"KV cache type: {kv_type}")
    tq = quant.cpp(model_path, kv_type=kv_type)
    print(tq)
    print()

    # Generate text
    print("--- Generation ---")
    text = tq.generate("The capital of France is", max_tokens=64, temperature=0.7)
    print(text)
    print()

    # Memory stats
    print("--- Memory Stats ---")
    try:
        stats = tq.memory_stats()
        print(f"  Tokens in cache:    {stats['tokens']}")
        print(f"  Compressed size:    {stats['compressed_mb']:.2f} MB")
        print(f"  FP16 baseline:      {stats['fp16_mb']:.2f} MB")
        print(f"  Compression ratio:  {stats['ratio']:.2f}x")
        print(f"  Memory saved:       {stats['saved_mb']:.2f} MB")
    except Exception as e:
        print(f"  (Could not get memory stats: {e})")
    print()

    # Perplexity (if a test file exists)
    test_file = os.path.join(os.path.dirname(model_path), "test.txt")
    if os.path.isfile(test_file):
        print("--- Perplexity ---")
        try:
            ppl = tq.perplexity(test_file)
            print(f"  PPL: {ppl:.4f}")
        except Exception as e:
            print(f"  (Could not compute PPL: {e})")
    else:
        print(f"--- Perplexity (skipped: no {test_file}) ---")

    # Model info
    print()
    print("--- Model Info ---")
    try:
        info = tq.info()
        for line in info.split("\n")[:10]:
            print(f"  {line}")
    except Exception as e:
        print(f"  (Could not get model info: {e})")


if __name__ == "__main__":
    main()
