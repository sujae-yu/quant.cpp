#!/usr/bin/env python3
"""HF Qwen3-0.6B per-layer hidden state dump.

Runs a fixed prompt through the reference model, captures the hidden
state after the embedding and after each of the 28 transformer layers,
and saves to an npz file for diffing against our engine's dump.

Usage:
    python hf_dump.py "Hello"
    -> writes hf_dump.npz with arrays emb, h0, h1, ..., h27, final_norm, logits
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "Hello"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tools/pillar1/hf_dump.npz"

print(f"prompt: {PROMPT!r}")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu")
model.eval()

ids = tok.encode(PROMPT, return_tensors="pt")
print(f"tokens: {ids.tolist()[0]}")

with torch.no_grad():
    out = model(ids, output_hidden_states=True, use_cache=False)

# out.hidden_states is a tuple of (n_layers+1) tensors, each [1, seq_len, dim]
# [0] = after embedding (before any layer)
# [i] = after layer (i-1)
# final logits after final norm + lm_head: out.logits [1, seq_len, vocab]
arrays = {}
for i, h in enumerate(out.hidden_states):
    key = "emb" if i == 0 else f"h{i-1}"
    arrays[key] = h[0].cpu().float().numpy()  # [seq_len, dim]
    print(f"  {key}: shape={arrays[key].shape}, first8={arrays[key][-1, :8]}")

arrays["logits"] = out.logits[0].cpu().float().numpy()  # [seq_len, vocab]
print(f"  logits: shape={arrays['logits'].shape}, top5_last=", end="")
last = arrays["logits"][-1]
top5 = np.argsort(-last)[:5]
print({int(t): f'{last[t]:.3f} ({tok.decode([int(t)])!r})' for t in top5})

# Also capture tokens for comparison
arrays["tokens"] = np.array(ids.tolist()[0], dtype=np.int64)

np.savez(OUT, **arrays)
print(f"saved {OUT}")
