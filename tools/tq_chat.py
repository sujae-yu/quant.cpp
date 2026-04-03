#!/usr/bin/env python3
"""
quant.cpp CLI -- Chat with Qwen3.5-0.8B using native C inference engine.

Calls the quant binary for fast inference (14 tok/s on CPU) with a visual
KV cache compression analysis overlay.  Falls back to PyTorch if quant is
not built.

Usage:
    python3 tools/tq_chat.py                          # Interactive mode
    python3 tools/tq_chat.py "Your question here"     # Single question
    python3 tools/tq_chat.py --engine pytorch          # Force PyTorch engine
    python3 tools/tq_chat.py --benchmark               # Run benchmark suite
"""

import sys
import os
import subprocess
import time
import glob
import argparse
import textwrap
import re

# ================================================================
# Colors (disabled when piped)
# ================================================================
IS_TTY = sys.stdout.isatty()

class C:
    if IS_TTY:
        BOLD = "\033[1m"; DIM = "\033[2m"; NC = "\033[0m"
        CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
        RED = "\033[31m"; MAGENTA = "\033[35m"; BLUE = "\033[34m"
        BAR = "\u2588"; BAR_EMPTY = "\u2591"
    else:
        BOLD = DIM = NC = CYAN = GREEN = YELLOW = RED = MAGENTA = BLUE = ""
        BAR = "#"; BAR_EMPTY = "-"

def bar(value, max_val, width=30, color=C.GREEN):
    filled = int(value / max_val * width) if max_val > 0 else 0
    filled = min(filled, width)
    return f"{color}{C.BAR * filled}{C.DIM}{C.BAR_EMPTY * (width - filled)}{C.NC}"

def size_str(bytes_val):
    if bytes_val >= 1024 * 1024 * 1024:
        return f"{bytes_val / 1024**3:.2f} GB"
    elif bytes_val >= 1024 * 1024:
        return f"{bytes_val / 1024**2:.1f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val} B"


# ================================================================
# Path auto-detection
# ================================================================

def find_model():
    """Auto-detect Qwen3.5-0.8B safetensors model path."""
    patterns = [
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
            "/snapshots/*/model.safetensors"
        ),
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
            "/snapshots/*/model.safetensors-00001-of-*.safetensors"
        ),
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
            "/snapshots/*/model-00001-of-*.safetensors"
        ),
    ]
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            return matches[-1]
    return None


def find_tokenizer():
    """Auto-detect tokenizer.json for Qwen3.5-0.8B."""
    patterns = [
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
            "/snapshots/*/tokenizer.json"
        ),
    ]
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            return matches[-1]
    return None


def find_quant():
    """Find the quant binary, searching common build locations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_root, "build", "quant"),
        os.path.join(project_root, "build", "Release", "quant"),
        os.path.join(project_root, "cmake-build-release", "quant"),
        "quant",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return None


# ================================================================
# Qwen3.5-0.8B model spec (for KV analysis without needing torch)
# ================================================================

MODEL_SPEC = {
    "name":           "Qwen3.5-0.8B",
    "n_layers":       6,
    "n_kv_heads":     2,
    "head_dim":       256,
    "hidden_dim":     1536,
    "vocab_size":     151936,
}


# ================================================================
# Header
# ================================================================

def print_header(engine_name="quant", engine_speed=None):
    speed_part = f"  |  {engine_speed:.1f} tok/s" if engine_speed else ""
    engine_label = (
        "Native C Inference Engine" if engine_name == "quant"
        else "PyTorch Inference Engine"
    )
    print()
    print(f"{C.CYAN}{C.BOLD}"
          f"{'=' * 60}{C.NC}")
    print(f"{C.CYAN}{C.BOLD}"
          f"  quant.cpp CLI -- {engine_label}{C.NC}")
    print(f"{C.CYAN}{C.BOLD}"
          f"  Model: {MODEL_SPEC['name']}"
          f"  |  Engine: {engine_name}{speed_part}{C.NC}")
    print(f"{C.CYAN}{C.BOLD}"
          f"{'=' * 60}{C.NC}")
    print()


# ================================================================
# KV analysis (computed from model spec, no actual cache needed)
# ================================================================

def print_kv_analysis(seq_len, gen_tokens=0, elapsed=0.0, kv_type="uniform_4b",
                      n_threads=4):
    """Compute and display KV cache compression analysis from model spec."""
    spec = MODEL_SPEC
    layers   = spec["n_layers"]
    kv_heads = spec["n_kv_heads"]
    head_dim = spec["head_dim"]

    total_tokens = seq_len + gen_tokens
    # FP16 KV size: layers * kv_heads * head_dim * total_tokens * 2 (K+V) * 2 (fp16 bytes)
    total_fp16 = layers * kv_heads * head_dim * total_tokens * 2 * 2

    tq_4b = int(total_fp16 * 4.2 / 16)
    tq_2b = int(total_fp16 * 2.2 / 16)
    k4v2  = int(total_fp16 * (4.2 + 2.2) / 2 / 16)

    print()
    print(f"  {C.BOLD}KV Cache Analysis{C.NC}")
    print(f"  {C.DIM}{'_' * 56}{C.NC}")

    # Model spec line
    print(f"  {C.BOLD}Model:{C.NC} {spec['name']}  {C.DIM}|{C.NC}  "
          f"{C.BOLD}{layers}{C.NC} attn layers  {C.DIM}|{C.NC}  "
          f"{C.BOLD}{kv_heads}{C.NC} KV heads  {C.DIM}|{C.NC}  "
          f"dim {C.BOLD}{head_dim}{C.NC}")

    # Performance line
    if gen_tokens > 0 and elapsed > 0:
        tps = gen_tokens / elapsed
        print(f"  {C.BOLD}Speed:{C.NC} {gen_tokens} tokens in {elapsed:.1f}s "
              f"({C.CYAN}{C.BOLD}{tps:.1f} tok/s{C.NC})  {C.DIM}|{C.NC}  "
              f"prompt {C.BOLD}{seq_len}{C.NC} tokens  {C.DIM}|{C.NC}  "
              f"{C.BOLD}{n_threads}{C.NC} threads  {C.DIM}|{C.NC}  "
              f"kv={C.BOLD}{kv_type}{C.NC}")
    else:
        print(f"  {C.BOLD}Tokens:{C.NC} {total_tokens} total")

    print()
    print(f"  {C.BOLD}{'Method':<22} {'Size':>10}  {'Compress':>9}  Bar{C.NC}")
    print(f"  {'_' * 22} {'_' * 10}  {'_' * 9}  {'_' * 30}")

    configs = [
        ("FP16 (baseline)", total_fp16, 1.0, C.RED),
        ("TQ uniform_4b",   tq_4b,     total_fp16 / tq_4b if tq_4b else 1, C.GREEN),
        ("TQ K4V2 asymmetric", k4v2,   total_fp16 / k4v2 if k4v2 else 1,  C.GREEN),
        ("TQ uniform_2b",   tq_2b,     total_fp16 / tq_2b if tq_2b else 1, C.YELLOW),
    ]

    for name, size, comp, color in configs:
        print(f"  {name:<22} {size_str(size):>10}  {comp:>7.1f}x  "
              f"{bar(size, total_fp16, 30, color)}")

    saved = total_fp16 - k4v2
    pct = saved * 100 // total_fp16 if total_fp16 > 0 else 0
    print()
    print(f"  {C.GREEN}{C.BOLD}Best balance (K4V2): "
          f"saves {size_str(saved)} ({pct}%){C.NC}")

    # Scale projections
    per_token = total_fp16 / total_tokens if total_tokens > 0 else 1
    print()
    print(f"  {C.BOLD}Projected at longer contexts:{C.NC}")
    for ctx in [4096, 16384, 65536, 131072]:
        fp16 = per_token * ctx
        k4v2_proj = fp16 * (4.2 + 2.2) / 2 / 16
        saved_proj = fp16 - k4v2_proj
        ctx_str = f"{ctx // 1024}K"
        print(f"  {ctx_str:>6}: FP16 {size_str(fp16):>10} -> "
              f"TQ {size_str(k4v2_proj):>10}  "
              f"{bar(k4v2_proj, fp16, 20, C.GREEN)} "
              f"save {size_str(saved_proj)}")
    print()


# ================================================================
# Native C engine (quant subprocess)
# ================================================================

def run_native(quant, model_path, tokenizer_path, question,
               max_tokens=150, threads=4, temp=0.7, kv_type="uniform_4b"):
    """Run inference via quant subprocess with streaming output.

    quant writes generated tokens to stdout and metadata (model info,
    --- markers, timing) to stderr.
    """
    cmd = [quant, model_path]
    if tokenizer_path:
        cmd += ["-t", tokenizer_path]
    cmd += [
        "-p", question,
        "-n", str(max_tokens),
        "-j", str(threads),
        "-T", str(temp),
        "-k", kv_type,
        "-q",  # Q8 weight quantization for speed
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    # Read stdout in real-time for streaming tokens
    generated_chunks = []
    try:
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            generated_chunks.append(chunk)
            # Print token as it arrives
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        proc.kill()
        print()
        return "", "", proc.returncode or 1

    proc.wait()
    stderr_text = proc.stderr.read()

    generated_text = "".join(generated_chunks).strip()
    return generated_text, stderr_text, proc.returncode


def parse_quant_stderr(stderr_text):
    """Extract timing and model info from quant stderr output.

    Expected format on the timing line:
        N tokens in X.Xs (Y.Y tok/s, T threads, kv=TYPE)
    """
    info = {
        "gen_tokens": 0,
        "elapsed": 0.0,
        "tok_per_sec": 0.0,
        "n_threads": 4,
        "kv_type": "uniform_4b",
        "model_info": "",
    }

    for line in stderr_text.splitlines():
        # Timing line
        m = re.match(
            r"(\d+)\s+tokens?\s+in\s+([\d.]+)s\s+"
            r"\(([\d.]+)\s+tok/s,\s+(\d+)\s+threads?,\s+kv=(\S+)\)",
            line.strip(),
        )
        if m:
            info["gen_tokens"]  = int(m.group(1))
            info["elapsed"]     = float(m.group(2))
            info["tok_per_sec"] = float(m.group(3))
            info["n_threads"]   = int(m.group(4))
            info["kv_type"]     = m.group(5)
            continue

        # Model info line (e.g. "Model: 6 layers, dim=1536, ...")
        if line.startswith("Model:"):
            info["model_info"] = line.strip()

    return info


def chat_native(quant, model_path, tokenizer_path, question,
                max_tokens=150, threads=4, temp=0.7, kv_type="uniform_4b"):
    """Run a single Q/A turn through the native C engine with visual output."""

    print(f"  {C.BOLD}{C.BLUE}Q:{C.NC} {question}")
    print()
    print(f"  {C.BOLD}{C.GREEN}A:{C.NC} ", end="", flush=True)

    generated_text, stderr_text, rc = run_native(
        quant, model_path, tokenizer_path, question,
        max_tokens=max_tokens, threads=threads, temp=temp, kv_type=kv_type,
    )

    if rc != 0:
        print()
        print(f"  {C.RED}quant exited with code {rc}{C.NC}")
        if stderr_text.strip():
            for line in stderr_text.strip().splitlines():
                print(f"  {C.DIM}{line}{C.NC}")
        return

    # Newline after streamed text
    print()

    # Parse timing info from stderr
    info = parse_quant_stderr(stderr_text)

    # Estimate prompt token count (rough: ~1.3 tokens per word for English)
    prompt_tokens = max(1, int(len(question.split()) * 1.3))

    print_kv_analysis(
        seq_len=prompt_tokens,
        gen_tokens=info["gen_tokens"],
        elapsed=info["elapsed"],
        kv_type=info["kv_type"],
        n_threads=info["n_threads"],
    )


# ================================================================
# PyTorch fallback engine
# ================================================================

def load_pytorch_model():
    """Load model with PyTorch (slow, but works without quant)."""
    import warnings
    import logging
    import contextlib
    import io

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3.5-0.8B"

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        device_label = "MPS (Apple GPU)"
    else:
        device = "cpu"
        dtype = torch.float32
        device_label = "CPU"

    with contextlib.redirect_stderr(io.StringIO()):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, dtype=dtype
        ).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device_label


def chat_pytorch(model, tokenizer, question, max_tokens=150):
    """Run a single Q/A turn through PyTorch with visual output."""
    import torch
    import contextlib
    import io
    import threading

    print(f"  {C.BOLD}{C.BLUE}Q:{C.NC} {question}")
    print()

    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    is_gpu = str(model.device) != "cpu"
    if not is_gpu:
        max_tokens = min(max_tokens, 80)
    dev_name = "GPU" if is_gpu else "CPU"
    est_time = max_tokens * (0.1 if is_gpu else 1.3)

    print(f"  {C.BOLD}{C.GREEN}A:{C.NC} "
          f"{C.DIM}(generating ~{max_tokens} tokens, "
          f"~{est_time:.0f}s on {dev_name}){C.NC}")

    # Spinner while generating
    stop_spinner = threading.Event()

    def spinner():
        chars = list("/-\\|")
        i = 0
        while not stop_spinner.is_set():
            print(f"\r  {C.CYAN}{chars[i % len(chars)]}{C.NC} generating...",
                  end="", flush=True)
            stop_spinner.wait(0.1)
            i += 1
        print(f"\r  {C.GREEN}done{C.NC}              ")

    t = threading.Thread(target=spinner, daemon=True)
    t.start()

    t0 = time.time()
    with torch.no_grad(), contextlib.redirect_stderr(io.StringIO()):
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    stop_spinner.set()
    t.join()
    elapsed = time.time() - t0

    answer = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    gen_tokens = out.shape[1] - prompt_len

    # Print answer with wrapping
    for line in answer.split("\n"):
        wrapped = textwrap.fill(line, width=72,
                                initial_indent="     ",
                                subsequent_indent="     ")
        print(wrapped)

    # KV cache analysis (via model spec, not actual cache)
    print_kv_analysis(
        seq_len=prompt_len,
        gen_tokens=gen_tokens,
        elapsed=elapsed,
        kv_type="fp16",
        n_threads=1,
    )


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="quant.cpp CLI -- Chat with KV cache analysis",
    )
    parser.add_argument(
        "question", nargs="?",
        help="Question to ask (interactive mode if omitted)",
    )
    parser.add_argument(
        "--engine", choices=["native", "pytorch"], default="native",
        help="Inference engine: native (quant, default) or pytorch",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to model safetensors file",
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--max-tokens", "-n", type=int, default=150,
        help="Maximum tokens to generate (default: 150)",
    )
    parser.add_argument(
        "--threads", "-j", type=int, default=4,
        help="Thread count for native engine (default: 4)",
    )
    parser.add_argument(
        "--temp", "-T", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--kv-type", "-k", default="uniform_4b",
        help="KV cache type for native engine (default: uniform_4b)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark suite with multiple questions",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------
    # Engine selection
    # -----------------------------------------------------------
    quant = find_quant()
    use_native = (args.engine == "native" and quant is not None)

    if args.engine == "native" and quant is None:
        print(f"  {C.YELLOW}quant binary not found, "
              f"falling back to PyTorch engine.{C.NC}")
        print(f"  {C.DIM}Build it with: cmake --build build{C.NC}")
        print()
        use_native = False

    # -----------------------------------------------------------
    # Native engine path
    # -----------------------------------------------------------
    if use_native:
        print_header(engine_name="quant")

        model_path = args.model or find_model()
        tokenizer_path = args.tokenizer or find_tokenizer()

        if not model_path:
            print(f"  {C.RED}Model not found.{C.NC}")
            print(f"  {C.DIM}Download with: "
                  f"huggingface-cli download Qwen/Qwen3.5-0.8B{C.NC}")
            print(f"  {C.DIM}Or specify: --model /path/to/model.safetensors{C.NC}")
            return 1

        if args.benchmark:
            questions = [
                "What is 2+2?",
                "Explain KV cache quantization in one paragraph.",
                "Write a Python function that computes fibonacci numbers.",
            ]
            for q in questions:
                chat_native(
                    quant, model_path, tokenizer_path, q,
                    max_tokens=args.max_tokens, threads=args.threads,
                    temp=args.temp, kv_type=args.kv_type,
                )
                print()
                print(f"  {C.DIM}{'=' * 52}{C.NC}")
                print()
        elif args.question:
            chat_native(
                quant, model_path, tokenizer_path, args.question,
                max_tokens=args.max_tokens, threads=args.threads,
                temp=args.temp, kv_type=args.kv_type,
            )
        else:
            # Interactive mode
            print(f"  {C.YELLOW}Interactive mode. "
                  f"Type your question (or 'quit' to exit).{C.NC}")
            print()
            while True:
                try:
                    q = input(f"  {C.BOLD}You:{C.NC} ").strip()
                    if not q or q.lower() in ("quit", "exit", "q"):
                        print(f"\n  {C.DIM}Goodbye!{C.NC}\n")
                        break
                    print()
                    chat_native(
                        quant, model_path, tokenizer_path, q,
                        max_tokens=args.max_tokens, threads=args.threads,
                        temp=args.temp, kv_type=args.kv_type,
                    )
                    print()
                    print(f"  {C.DIM}{'=' * 52}{C.NC}")
                    print()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n  {C.DIM}Goodbye!{C.NC}\n")
                    break

    # -----------------------------------------------------------
    # PyTorch fallback path
    # -----------------------------------------------------------
    else:
        print_header(engine_name="PyTorch")

        print(f"  {C.DIM}Loading Qwen3.5-0.8B...{C.NC}", end="", flush=True)
        model, tokenizer, device_label = load_pytorch_model()
        print(f" {C.GREEN}done{C.NC} {C.DIM}({device_label}){C.NC}")
        print()

        if args.benchmark:
            questions = [
                "What is 2+2?",
                "Explain KV cache quantization in one paragraph.",
                "Write a Python function that computes fibonacci numbers.",
            ]
            for q in questions:
                chat_pytorch(model, tokenizer, q,
                             max_tokens=args.max_tokens)
                print()
                print(f"  {C.DIM}{'=' * 52}{C.NC}")
                print()
        elif args.question:
            chat_pytorch(model, tokenizer, args.question,
                         max_tokens=args.max_tokens)
        else:
            print(f"  {C.YELLOW}Interactive mode. "
                  f"Type your question (or 'quit' to exit).{C.NC}")
            print()
            while True:
                try:
                    q = input(f"  {C.BOLD}You:{C.NC} ").strip()
                    if not q or q.lower() in ("quit", "exit", "q"):
                        print(f"\n  {C.DIM}Goodbye!{C.NC}\n")
                        break
                    print()
                    chat_pytorch(model, tokenizer, q,
                                 max_tokens=args.max_tokens)
                    print()
                    print(f"  {C.DIM}{'=' * 52}{C.NC}")
                    print()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n  {C.DIM}Goodbye!{C.NC}\n")
                    break

    return 0


if __name__ == "__main__":
    sys.exit(main())
