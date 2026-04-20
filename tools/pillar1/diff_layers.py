#!/usr/bin/env python3
"""Layer-by-layer diff between HF reference and our engine's dumps.

Input:
  tools/pillar1/hf_dump_long.npz  (emb, h0..h27, logits per-position)
  /tmp/qdump/*.bin                (our engine's pos=143 dumps, raw float32)

Output: per-layer table of cosine, max_abs_diff, L2_relative."""
import numpy as np, os, sys, glob

HF_NPZ = sys.argv[1] if len(sys.argv) > 1 else "tools/pillar1/hf_dump_long.npz"
US_DIR = sys.argv[2] if len(sys.argv) > 2 else "/tmp/qdump"
POS    = int(sys.argv[3]) if len(sys.argv) > 3 else 143

hf = np.load(HF_NPZ)
print(f"HF npz keys: {list(hf.keys())[:5]}... shape_h0={hf['h0'].shape}")
print(f"Reading our dumps from {US_DIR} at position {POS}")
print()
print(f"{'slot':<12} {'dim':>6} {'our_norm':>10} {'hf_norm':>10} {'max_abs':>10} {'L2_rel':>10} {'cosine':>8}")
print("-" * 70)

def read_bin(path):
    return np.fromfile(path, dtype=np.float32)

slots = ["emb"] + [f"h{i}" for i in range(28)] + ["post_norm"]
for slot in slots:
    bin_path = os.path.join(US_DIR, f"{slot}.bin")
    if not os.path.exists(bin_path):
        continue
    ours = read_bin(bin_path)
    if slot not in hf.files:
        continue
    hf_arr = hf[slot]
    if hf_arr.ndim == 2:
        ref = hf_arr[POS]  # last-position vector for this layer
    else:
        ref = hf_arr
    if ours.shape != ref.shape:
        print(f"{slot}: shape mismatch us={ours.shape} hf={ref.shape}")
        continue
    diff = ours - ref
    max_abs = np.max(np.abs(diff))
    l2 = np.linalg.norm(diff)
    hf_norm = np.linalg.norm(ref)
    us_norm = np.linalg.norm(ours)
    l2_rel = l2 / max(hf_norm, 1e-9)
    cos = np.dot(ours, ref) / max(us_norm * hf_norm, 1e-9)
    print(f"{slot:<12} {len(ours):>6} {us_norm:>10.3f} {hf_norm:>10.3f} {max_abs:>10.4f} {l2_rel:>10.4%} {cos:>8.4f}")

# Compare top-5 logits (our dump logits.bin is FP32 full-vocab)
print()
logits_path = os.path.join(US_DIR, "logits.bin")
if os.path.exists(logits_path):
    ours_l = read_bin(logits_path)
    hf_l = hf["logits"][POS] if hf["logits"].ndim == 2 else hf["logits"]
    top5_us = np.argsort(-ours_l)[:5]
    top5_hf = np.argsort(-hf_l)[:5]
    print(f"HF  top-5 logits: {[(int(t), f'{hf_l[t]:.2f}') for t in top5_hf]}")
    print(f"Us  top-5 logits: {[(int(t), f'{ours_l[t]:.2f}') for t in top5_us]}")
