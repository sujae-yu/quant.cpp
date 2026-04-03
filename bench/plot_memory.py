#!/usr/bin/env python3
"""
plot_memory.py -- KV Cache Memory Usage: quant.cpp vs FP16 (llama.cpp)

Reads the CSV output from long_context_bench.sh and generates:
  - A line chart (PNG) if matplotlib is available
  - An ASCII chart as fallback

Usage:
  python3 bench/plot_memory.py [bench/long_context_results.csv]
"""

import csv
import os
import sys

def read_csv(path):
    """Read benchmark CSV and return lists of context_length, compressed_mb, fp16_mb."""
    ctx_lengths = []
    compressed_mb = []
    fp16_mb = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctx_lengths.append(int(row["context_length"]))
            compressed_mb.append(float(row["compressed_kv_mb"]))
            fp16_mb.append(float(row["fp16_kv_mb"]))

    return ctx_lengths, compressed_mb, fp16_mb


def plot_matplotlib(ctx_lengths, compressed_mb, fp16_mb, output_path):
    """Generate a publication-quality line chart using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot both lines
    ax.plot(ctx_lengths, fp16_mb, "o-", color="#e74c3c", linewidth=2.5,
            markersize=8, label="FP16 KV (llama.cpp)", zorder=3)
    ax.plot(ctx_lengths, compressed_mb, "s-", color="#2ecc71", linewidth=2.5,
            markersize=8, label="Q4 KV (quant.cpp)", zorder=3)

    # Fill the gap between them
    ax.fill_between(ctx_lengths, compressed_mb, fp16_mb,
                     alpha=0.15, color="#3498db", label="Memory saved")

    # Annotate compression ratio at the last point
    if len(ctx_lengths) > 0:
        last_idx = len(ctx_lengths) - 1
        ratio = fp16_mb[last_idx] / compressed_mb[last_idx] if compressed_mb[last_idx] > 0 else 0
        saved = fp16_mb[last_idx] - compressed_mb[last_idx]
        ax.annotate(
            f"  {ratio:.1f}x smaller\n  ({saved:.0f} MB saved)",
            xy=(ctx_lengths[last_idx], (fp16_mb[last_idx] + compressed_mb[last_idx]) / 2),
            fontsize=11, fontweight="bold", color="#2c3e50",
            ha="left", va="center",
        )

    ax.set_xlabel("Context Length (tokens)", fontsize=13)
    ax.set_ylabel("KV Cache Memory (MB)", fontsize=13)
    ax.set_title("KV Cache Memory: quant.cpp Q4 vs FP16 (llama.cpp)\nGemma 3 4B — 34 layers, 4 KV heads, head_dim=256",
                 fontsize=14, fontweight="bold")

    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else f"{x}"))
    ax.set_xlim(left=min(ctx_lengths) * 0.8, right=max(ctx_lengths) * 1.2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.close()


def plot_ascii(ctx_lengths, compressed_mb, fp16_mb):
    """Generate an ASCII chart as fallback when matplotlib is unavailable."""
    WIDTH = 60
    max_val = max(fp16_mb) if fp16_mb else 1.0

    print("\n" + "=" * (WIDTH + 30))
    print("  KV Cache Memory: quant.cpp Q4 vs FP16 (llama.cpp)")
    print("  Gemma 3 4B -- 34 layers, 4 KV heads, head_dim=256")
    print("=" * (WIDTH + 30))
    print()

    for i, ctx in enumerate(ctx_lengths):
        fp16_bar_len = int(fp16_mb[i] / max_val * WIDTH)
        comp_bar_len = int(compressed_mb[i] / max_val * WIDTH)

        ratio = fp16_mb[i] / compressed_mb[i] if compressed_mb[i] > 0 else 0

        print(f"  {ctx:>7,} tokens:")
        print(f"    FP16  |{'#' * fp16_bar_len:{WIDTH}s}| {fp16_mb[i]:>8.1f} MB")
        print(f"    Q4    |{'=' * comp_bar_len:{WIDTH}s}| {compressed_mb[i]:>8.1f} MB  ({ratio:.1f}x)")
        print()

    # Summary
    if len(ctx_lengths) > 0:
        last = len(ctx_lengths) - 1
        saved = fp16_mb[last] - compressed_mb[last]
        ratio = fp16_mb[last] / compressed_mb[last] if compressed_mb[last] > 0 else 0
        print(f"  At {ctx_lengths[last]:,} tokens: {ratio:.1f}x compression, {saved:.0f} MB saved")
    print()


def main():
    # Determine CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default: look relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "long_context_results.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        print("Run bench/long_context_bench.sh first to generate data.", file=sys.stderr)
        sys.exit(1)

    ctx_lengths, compressed_mb, fp16_mb = read_csv(csv_path)

    if not ctx_lengths:
        print("ERROR: No data in CSV file", file=sys.stderr)
        sys.exit(1)

    # Determine output path for PNG
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    assets_dir = os.path.join(project_dir, "docs", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    output_png = os.path.join(assets_dir, "long_context_memory.png")

    # Try matplotlib first, fall back to ASCII
    try:
        import matplotlib
        plot_matplotlib(ctx_lengths, compressed_mb, fp16_mb, output_png)
    except ImportError:
        print("matplotlib not available, generating ASCII chart instead.")
        print(f"Install matplotlib for PNG output: pip install matplotlib")

    # Always print ASCII chart to terminal
    plot_ascii(ctx_lengths, compressed_mb, fp16_mb)


if __name__ == "__main__":
    main()
