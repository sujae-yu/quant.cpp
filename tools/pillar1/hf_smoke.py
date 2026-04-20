#!/usr/bin/env python3
"""HF Qwen3-0.6B smoke test — verify model runs coherent on same prompts
where our engine produces garbage. Establishes the reference baseline."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    ("Hello", 15),
    ("The quick brown fox", 20),
    ("What is 2 plus 2?", 25),
]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
model.eval()

for prompt, n in PROMPTS:
    ids = tok.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False, temperature=1.0)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    print(f"prompt: {prompt!r}")
    print(f"  tokens: {ids.tolist()[0]}")
    print(f"  output: {gen!r}")
    print()
