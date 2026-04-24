#!/usr/bin/env python3
import argparse
import array
import math
import os
import re
import struct
import sys


def read_f32_bin(path):
    data = array.array("f")
    with open(path, "rb") as f:
        data.frombytes(f.read())
    return data


def read_i32_bin(path):
    out = []
    with open(path, "rb") as f:
        raw = f.read()
    for i in range(0, len(raw), 4):
        out.append(struct.unpack_from("<i", raw, i)[0])
    return out


def topk(values, k=5):
    pairs = sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:k]
    return pairs


def cosine(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def rel_l2(a, b):
    num = 0.0
    den = 0.0
    for x, y in zip(a, b):
        d = x - y
        num += d * d
        den += y * y
    if den <= 0.0:
        return 0.0
    return math.sqrt(num / den)


def rank_of(values, token_id):
    target = values[token_id]
    rank = 1
    for v in values:
        if v > target:
            rank += 1
    return rank


def parse_logit_probe(path):
    rows = []
    if not os.path.exists(path):
        return rows
    pat = re.compile(
        r"pos=(?P<pos>\d+).*top5_ids=\[(?P<ids>[0-9,]+)\].*margin=(?P<margin>[-0-9.]+).*entropy=(?P<entropy>[-0-9.]+)"
    )
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line.replace(" ", ""))
            if not m:
                continue
            rows.append(
                {
                    "pos": int(m.group("pos")),
                    "top5_ids": [int(x) for x in m.group("ids").split(",") if x],
                    "margin": float(m.group("margin")),
                    "entropy": float(m.group("entropy")),
                }
            )
    return rows


def first_logits_dump(case_dir):
    dumps_dir = os.path.join(case_dir, "dumps")
    if not os.path.isdir(dumps_dir):
        return None, None
    candidates = []
    for name in os.listdir(dumps_dir):
        m = re.fullmatch(r"logits_p(\d+)\.bin", name)
        if m:
            candidates.append((int(m.group(1)), os.path.join(dumps_dir, name)))
    if not candidates:
        return None, None
    candidates.sort()
    return candidates[0]


def find_llama_logits(llama_dir):
    for name in os.listdir(llama_dir):
        if name.startswith("llamacpp-") and name.endswith(".bin"):
            return os.path.join(llama_dir, name)
    return None


def find_llama_tokens(llama_dir):
    for name in os.listdir(llama_dir):
        if name.endswith("-tokens.bin"):
            return os.path.join(llama_dir, name)
    return None


def summarize_case(case_name, case_dir, ref_logits, ref_top5, ref_top1):
    probe_rows = parse_logit_probe(os.path.join(case_dir, "logit_probe.txt"))
    first_probe = probe_rows[0] if probe_rows else None
    dump_pos, dump_path = first_logits_dump(case_dir)

    print(f"\n[{case_name}]")
    if first_probe:
        overlap = len(set(first_probe["top5_ids"]) & {idx for idx, _ in ref_top5})
        print(
            f"  first probe pos={first_probe['pos']} top5={first_probe['top5_ids']} "
            f"margin={first_probe['margin']:.3f} entropy={first_probe['entropy']:.3f} "
            f"top5_overlap_vs_llama={overlap}/5"
        )
    else:
        print("  no logit probe rows found")

    if dump_path:
        logits = read_f32_bin(dump_path)
        case_top5 = topk(logits, 5)
        case_top1 = case_top5[0][0]
        print(
            f"  dump={os.path.basename(dump_path)} cosine={cosine(logits, ref_logits):.6f} "
            f"rel_l2={rel_l2(logits, ref_logits):.6f} "
            f"top1={case_top1} ref_top1_rank={rank_of(logits, ref_top1)} "
            f"top1_match={'yes' if case_top1 == ref_top1 else 'no'}"
        )
        print(
            "  case top5 ids="
            + str([idx for idx, _ in case_top5])
            + " llama top5 ids="
            + str([idx for idx, _ in ref_top5])
        )
    else:
        print("  no quant logits dump found (re-run qwen36_parity.sh with --dump-hidden)")


def summarize_hidden_delta(run_dir):
    base_dir = os.path.join(run_dir, "baseline", "dumps")
    lk_dir = os.path.join(run_dir, "llama_kernels", "dumps")
    if not os.path.isdir(base_dir) or not os.path.isdir(lk_dir):
        return

    deltas = []
    for name in os.listdir(base_dir):
        if not name.endswith(".bin"):
            continue
        a_path = os.path.join(base_dir, name)
        b_path = os.path.join(lk_dir, name)
        if not os.path.exists(b_path):
            continue
        if not re.search(r"^(h\d+|post_norm|logits|emb)", name):
            continue
        a = read_f32_bin(a_path)
        b = read_f32_bin(b_path)
        if len(a) != len(b) or len(a) == 0:
            continue
        deltas.append((name, rel_l2(a, b), cosine(a, b)))

    if not deltas:
        return

    deltas.sort(key=lambda x: x[1], reverse=True)
    print("\n[baseline vs llama_kernels hidden delta]")
    for name, l2v, cosv in deltas[:8]:
        print(f"  {name}: rel_l2={l2v:.6f} cosine={cosv:.6f}")


def main():
    ap = argparse.ArgumentParser(description="Analyze qwen36 parity run output")
    ap.add_argument("run_dir", help="output directory from tools/qwen36_parity.sh")
    args = ap.parse_args()

    llama_dir = os.path.join(args.run_dir, "llama_debug")
    ref_path = find_llama_logits(llama_dir)
    if not ref_path:
        print("llama-debug logits not found", file=sys.stderr)
        return 1

    ref_logits = read_f32_bin(ref_path)
    ref_top5 = topk(ref_logits, 5)
    ref_top1 = ref_top5[0][0]

    tok_path = find_llama_tokens(llama_dir)
    prompt_tokens = len(read_i32_bin(tok_path)) if tok_path else None

    print(f"run_dir: {args.run_dir}")
    print(f"llama logits: {ref_path}")
    print(f"vocab: {len(ref_logits)}")
    if prompt_tokens is not None:
        print(f"prompt_tokens: {prompt_tokens}")
    print(f"llama top5 ids: {[idx for idx, _ in ref_top5]}")
    print(f"llama top5 logits: {[round(val, 4) for _, val in ref_top5]}")

    summarize_case("baseline", os.path.join(args.run_dir, "baseline"), ref_logits, ref_top5, ref_top1)
    summarize_case("llama_kernels", os.path.join(args.run_dir, "llama_kernels"), ref_logits, ref_top5, ref_top1)
    summarize_hidden_delta(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
