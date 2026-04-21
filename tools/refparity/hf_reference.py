#!/usr/bin/env python3
"""Dump HF reference model per-layer hidden states + logits.

Generalized from tools/pillar1/hf_dump.py. Runs the given HF model on a
prompt, captures every layer's output hidden state, saves as npz.

Usage:
    python hf_reference.py --model Qwen/Qwen3-0.6B --prompt "Hello" --out ref.npz
    python hf_reference.py --model Qwen/Qwen3-0.6B --prompt-file prompt.txt --out ref.npz

Exit codes:
    0  success
    2  environment / config error
"""
import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}


def run(model_name: str, prompt: str, out_path: str,
        dtype: str = "float32") -> int:
    torch_dtype = DTYPE_MAP.get(dtype)
    if torch_dtype is None:
        print(f"error: unknown dtype {dtype!r}; valid: {list(DTYPE_MAP)}",
              file=sys.stderr)
        return 2
    print(f"[refparity/hf] model={model_name} dtype={dtype}", file=sys.stderr)
    print(f"[refparity/hf] prompt: {prompt[:80]!r}{'...' if len(prompt) > 80 else ''}",
          file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, device_map="cpu")
    model.eval()

    ids = tok.encode(prompt, return_tensors="pt")
    seq_len = ids.shape[1]
    print(f"[refparity/hf] tokens: {seq_len}, ids[:8]={ids.tolist()[0][:8]}",
          file=sys.stderr)

    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)

    # transformers Qwen3/Llama append hidden_states PRE-layer plus one
    # final entry AFTER the post-final-norm. So for N layers we get N+1
    # entries: (emb, layer0_out, layer1_out, ..., layer_{N-2}_out, post_norm).
    # The last-layer PRE-NORM output is NOT exposed — use post_norm to
    # cross-check the engine's post_norm slot instead.
    arrays = {
        "tokens": np.array(ids.tolist()[0], dtype=np.int64),
        "logits": out.logits[0].cpu().float().numpy(),  # [seq_len, vocab]
    }
    n_hs = len(out.hidden_states)
    for i, h in enumerate(out.hidden_states):
        arr = h[0].cpu().float().numpy()  # [seq_len, dim]
        if i == 0:
            arrays["emb"] = arr
        elif i == n_hs - 1:
            arrays["post_norm"] = arr
        else:
            arrays[f"h{i-1}"] = arr

    np.savez(out_path, **arrays)

    # Summary line to stderr
    last_logits = arrays["logits"][-1]
    top1 = int(last_logits.argmax())
    top1_tok = tok.decode([top1])
    print(f"[refparity/hf] layers={len(out.hidden_states)-1} "
          f"dim={arrays['h0'].shape[-1]} "
          f"top1_last={top1}({top1_tok!r}) "
          f"→ {out_path}", file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="literal prompt text")
    group.add_argument("--prompt-file", help="read prompt from file")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--dtype", default="float32",
                    choices=list(DTYPE_MAP.keys()),
                    help="HF model dtype (default: float32). Use bfloat16 "
                         "for 4B+ models on 16 GB machines.")
    args = ap.parse_args()

    if args.prompt:
        prompt = args.prompt
    else:
        with open(args.prompt_file) as f:
            prompt = f.read().rstrip("\n")

    return run(args.model, prompt, args.out, dtype=args.dtype)


if __name__ == "__main__":
    sys.exit(main())
