#!/usr/bin/env python3
"""Layer-by-layer diff: HF reference npz vs our engine's raw bin dumps.

Generalized from tools/pillar1/diff_layers.py. Produces a tabular report
and exits 0 (PASS) / 1 (FAIL) based on thresholds.

Usage:
    python diff_layers.py ref.npz engine_dump/ \
        --threshold-l2-rel 0.05 \
        --threshold-cosine 0.90

Output (stdout):
    slot   dim   us_norm   hf_norm   max_abs   L2_rel   cosine  [PASS|FAIL]
    emb    ...
    h0     ...
    ...
    → PASS / FAIL — first divergence at layer X

Exit codes:
    0 — all layers within threshold
    1 — divergence detected; diff report identifies layer
    2 — environment / config error
"""
import argparse
import os
import sys

import numpy as np


def read_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def compare(hf_vec: np.ndarray, us_vec: np.ndarray):
    diff = us_vec - hf_vec
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    l2 = float(np.linalg.norm(diff))
    hf_norm = float(np.linalg.norm(hf_vec))
    us_norm = float(np.linalg.norm(us_vec))
    l2_rel = l2 / max(hf_norm, 1e-9)
    denom = max(us_norm * hf_norm, 1e-9)
    cosine = float(np.dot(us_vec, hf_vec) / denom)
    return us_norm, hf_norm, max_abs, l2_rel, cosine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_npz", help="HF reference .npz")
    ap.add_argument("engine_dir", help="engine dump directory")
    ap.add_argument("--pos", type=int, default=None,
                    help="position to compare in HF (default: 0 — matches "
                         "engine's TQ_DUMP_POS=0 default)")
    ap.add_argument("--threshold-l2-rel", type=float, default=0.05,
                    help="max L2_rel per hidden layer (default 0.05 = 5%%)")
    ap.add_argument("--threshold-cosine", type=float, default=0.90,
                    help="min cosine similarity at logits (default 0.90)")
    args = ap.parse_args()

    try:
        hf = np.load(args.ref_npz)
    except Exception as e:
        print(f"error: cannot load {args.ref_npz}: {e}", file=sys.stderr)
        return 2

    # Engine's TQ_DUMP_HIDDEN default is pos=0 (first token); align by default
    seq_len = hf["h0"].shape[0] if hf["h0"].ndim == 2 else 1
    pos = 0 if args.pos is None else args.pos
    if pos >= seq_len:
        print(f"error: pos {pos} >= seq_len {seq_len}", file=sys.stderr)
        return 2

    # Determine layer count from engine dumps
    engine_files = os.listdir(args.engine_dir)
    max_h = -1
    for f in engine_files:
        if f.startswith("h") and f.endswith(".bin"):
            try:
                max_h = max(max_h, int(f[1:-4]))
            except ValueError:
                pass
    slots = ["emb"] + [f"h{i}" for i in range(max_h + 1)]
    has_post_norm = os.path.exists(os.path.join(args.engine_dir, "post_norm.bin"))
    if has_post_norm:
        slots.append("post_norm")

    print(f"{'slot':<12} {'dim':>6} {'us_norm':>10} {'hf_norm':>10} "
          f"{'max_abs':>10} {'L2_rel':>10} {'cosine':>8}  status")
    print("-" * 85)

    first_fail = None
    all_rows = []
    for slot in slots:
        bin_path = os.path.join(args.engine_dir, f"{slot}.bin")
        if not os.path.exists(bin_path):
            continue
        us = read_bin(bin_path)
        if slot == "post_norm":
            # HF npz doesn't usually have post_norm; skip if absent
            if "post_norm" not in hf.files:
                continue
            hf_arr = hf["post_norm"]
            hf_vec = hf_arr[pos] if hf_arr.ndim == 2 else hf_arr
        else:
            if slot not in hf.files:
                continue
            hf_arr = hf[slot]
            hf_vec = hf_arr[pos] if hf_arr.ndim == 2 else hf_arr

        if us.shape != hf_vec.shape:
            print(f"{slot:<12} shape mismatch us={us.shape} hf={hf_vec.shape}")
            continue

        us_norm, hf_norm, max_abs, l2_rel, cosine = compare(hf_vec, us)
        status = "PASS"
        if l2_rel > args.threshold_l2_rel:
            status = "FAIL"
            if first_fail is None:
                first_fail = slot

        print(f"{slot:<12} {len(us):>6} {us_norm:>10.3f} {hf_norm:>10.3f} "
              f"{max_abs:>10.4f} {l2_rel:>10.4%} {cosine:>8.4f}  {status}")
        all_rows.append((slot, status, l2_rel, cosine))

    # Compare top-5 logits
    logits_path = os.path.join(args.engine_dir, "logits.bin")
    logits_pass = True
    if os.path.exists(logits_path) and "logits" in hf.files:
        us_l = read_bin(logits_path)
        hf_l = hf["logits"][pos] if hf["logits"].ndim == 2 else hf["logits"]
        if us_l.shape == hf_l.shape:
            top1_us = int(us_l.argmax())
            top1_hf = int(hf_l.argmax())
            cos_l = float(np.dot(us_l, hf_l) /
                          max(np.linalg.norm(us_l) * np.linalg.norm(hf_l), 1e-9))
            print()
            print(f"logits cosine={cos_l:.4f}  top1 hf={top1_hf} us={top1_us}  "
                  f"{'PASS' if (cos_l >= args.threshold_cosine and top1_us == top1_hf) else 'FAIL'}")
            if cos_l < args.threshold_cosine or top1_us != top1_hf:
                logits_pass = False
                if first_fail is None:
                    first_fail = "logits"

    print()
    if first_fail is None and logits_pass:
        print("→ PASS — all layers within threshold")
        return 0
    else:
        print(f"→ FAIL — first divergence at {first_fail}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
