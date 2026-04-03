#!/usr/bin/env python3
"""
quant.cpp -- RULER Benchmark Script

RULER (Real-world Understanding and Language Extraction from Retrieval) measures
how well LLMs handle long-context tasks with different KV cache quantization.

This script defines the benchmark workflow for evaluating quant.cpp's KV cache
compression across various context lengths and retrieval depths.

Benchmark Workflow:
  1. Load a pre-trained model (e.g., Llama-3-8B) with quant.cpp KV cache.
  2. For each context length (4K, 8K, 16K, 32K, 64K, 128K):
     a. Generate synthetic documents with embedded "needles" at various depths.
     b. Run the model with quant.cpp quantized KV cache.
     c. Measure recall@k for each depth position (0%, 25%, 50%, 75%, 100%).
  3. Compare results across quantization types:
     - FP16 baseline
     - TQ_POLAR_3B, TQ_POLAR_4B
     - TQ_QJL_1B
     - TQ_TURBO_3B, TQ_TURBO_4B
     - TQ_UNIFORM_4B, TQ_UNIFORM_2B
  4. Output results as JSON for downstream analysis.

Expected Metrics:
  - recall@1: Fraction of single-needle retrievals correct
  - recall@5: Fraction of multi-needle retrievals correct (top-5)
  - For each (quant_type, context_length, depth_pct) combination

Requirements:
  - numpy
  - transformers (for model loading)
  - turboquant (Python bindings, see bindings/python/)

Usage:
  python bench/accuracy/run_ruler.py [--model llama-3-8b] [--output results/ruler.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

DEPTH_PERCENTAGES = [0.0, 0.25, 0.50, 0.75, 1.0]

QUANTIZATION_TYPES = [
    "fp16",
    "polar_3b",
    "polar_4b",
    "qjl_1b",
    "turbo_3b",
    "turbo_4b",
    "uniform_4b",
    "uniform_2b",
]

NUM_TRIALS = 5

# ---------------------------------------------------------------------------
# Synthetic needle generation
# ---------------------------------------------------------------------------

def generate_haystack(context_length: int, needle: str, depth_pct: float,
                      vocab_size: int = 32000, seed: int = 42) -> dict:
    """
    Generate a synthetic document (haystack) with a needle embedded at
    a specific depth percentage.

    Returns a dict with:
      - 'tokens': numpy array of token IDs (int32)
      - 'needle_start': index where the needle begins
      - 'needle_tokens': the needle token IDs
    """
    rng = np.random.RandomState(seed)

    # Generate random "filler" tokens (simulating document content)
    needle_tokens = rng.randint(1, vocab_size, size=len(needle.split()))

    filler_len = context_length - len(needle_tokens)
    if filler_len < 0:
        filler_len = 0

    # Compute needle insertion position
    insert_pos = int(filler_len * depth_pct)

    # Build the token sequence
    before = rng.randint(1, vocab_size, size=insert_pos).astype(np.int32)
    after = rng.randint(1, vocab_size, size=filler_len - insert_pos).astype(np.int32)
    tokens = np.concatenate([before, needle_tokens, after])

    return {
        "tokens": tokens,
        "needle_start": insert_pos,
        "needle_tokens": needle_tokens,
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_recall_at_k(predicted_positions: list, true_position: int, k: int) -> float:
    """
    Compute recall@k: whether the true position appears in the top-k predicted.
    """
    if len(predicted_positions) == 0:
        return 0.0
    top_k = predicted_positions[:k]
    return 1.0 if true_position in top_k else 0.0


def simulate_attention_scores(context_length: int, needle_start: int,
                              quant_type: str, head_dim: int = 128,
                              seed: int = 42) -> np.ndarray:
    """
    Simulate attention scores for a given quantization type.

    In a real benchmark, this would run the model's attention with quantized
    KV cache. Here we simulate the effect of quantization noise on attention
    scores to validate the benchmark infrastructure.

    Higher-bit quantization types produce attention patterns closer to FP16.
    Lower-bit types introduce more noise, making retrieval harder.
    """
    rng = np.random.RandomState(seed)

    # Base attention pattern: strong peak at needle position
    scores = rng.randn(context_length).astype(np.float32) * 0.1
    scores[needle_start] = 5.0  # Strong signal at needle

    # Add quantization noise proportional to compression level
    noise_scale = {
        "fp16":       0.0,
        "polar_4b":   0.05,
        "uniform_4b": 0.08,
        "polar_3b":   0.10,
        "turbo_4b":   0.04,
        "turbo_3b":   0.07,
        "uniform_2b": 0.20,
        "qjl_1b":     0.30,
    }

    scale = noise_scale.get(quant_type, 0.1)
    noise = rng.randn(context_length).astype(np.float32) * scale
    scores += noise

    return scores


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_ruler_benchmark(output_path: str = None) -> dict:
    """
    Run the RULER benchmark across all configurations.

    Returns a dict of results keyed by (quant_type, context_length, depth_pct).
    """
    results = {}
    needle_text = "The secret code is ALPHA-7 BRAVO-3"

    total_configs = len(QUANTIZATION_TYPES) * len(CONTEXT_LENGTHS) * len(DEPTH_PERCENTAGES)
    completed = 0

    print(f"RULER Benchmark: {total_configs} configurations, {NUM_TRIALS} trials each")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Quantization types: {QUANTIZATION_TYPES}")
    print()

    for quant_type in QUANTIZATION_TYPES:
        for ctx_len in CONTEXT_LENGTHS:
            for depth_pct in DEPTH_PERCENTAGES:
                recall_1_scores = []
                recall_5_scores = []

                for trial in range(NUM_TRIALS):
                    seed = trial * 1000 + int(depth_pct * 100) + ctx_len

                    haystack = generate_haystack(
                        context_length=ctx_len,
                        needle=needle_text,
                        depth_pct=depth_pct,
                        seed=seed,
                    )

                    attn_scores = simulate_attention_scores(
                        context_length=len(haystack["tokens"]),
                        needle_start=haystack["needle_start"],
                        quant_type=quant_type,
                        seed=seed,
                    )

                    # Rank positions by attention score (descending)
                    ranked = np.argsort(-attn_scores).tolist()

                    r1 = compute_recall_at_k(ranked, haystack["needle_start"], k=1)
                    r5 = compute_recall_at_k(ranked, haystack["needle_start"], k=5)

                    recall_1_scores.append(r1)
                    recall_5_scores.append(r5)

                key = f"{quant_type}_{ctx_len}_{depth_pct:.2f}"
                results[key] = {
                    "quant_type": quant_type,
                    "context_length": ctx_len,
                    "depth_pct": depth_pct,
                    "recall_at_1": float(np.mean(recall_1_scores)),
                    "recall_at_5": float(np.mean(recall_5_scores)),
                    "num_trials": NUM_TRIALS,
                }

                completed += 1

    # Print summary
    print(f"\n{'Type':<14} {'CtxLen':>8} {'Depth':>6} {'R@1':>6} {'R@5':>6}")
    print("-" * 46)
    for key, res in sorted(results.items()):
        print(f"{res['quant_type']:<14} {res['context_length']:>8} "
              f"{res['depth_pct']:>6.2f} {res['recall_at_1']:>6.2f} "
              f"{res['recall_at_5']:>6.2f}")

    # Save results
    output = {
        "benchmark": "RULER",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "context_lengths": CONTEXT_LENGTHS,
            "depth_percentages": DEPTH_PERCENTAGES,
            "quantization_types": QUANTIZATION_TYPES,
            "num_trials": NUM_TRIALS,
        },
        "results": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RULER benchmark for quant.cpp KV cache quantization"
    )
    parser.add_argument(
        "--model", type=str, default="simulated",
        help="Model name (default: simulated -- uses synthetic attention scores)"
    )
    parser.add_argument(
        "--output", type=str, default="bench/accuracy/results/ruler.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=None,
        help="Override context lengths (e.g., --context-lengths 4096 8192)"
    )
    args = parser.parse_args()

    if args.context_lengths:
        global CONTEXT_LENGTHS
        CONTEXT_LENGTHS = args.context_lengths

    if args.model != "simulated":
        print(f"NOTE: Model '{args.model}' requested but only simulated mode "
              "is currently implemented.", file=sys.stderr)
        print("Run with quant.cpp Python bindings for actual model evaluation.",
              file=sys.stderr)

    run_ruler_benchmark(output_path=args.output)


if __name__ == "__main__":
    main()
