#!/usr/bin/env python3
import argparse
import array
import csv
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_PROMPTS = [
    "Explain quantum mechanics in simple terms with examples.",
    "Once upon a time in a faraway land",
    "What is the capital of France?",
]

DEFAULT_CASES = [
    ("baseline", {}),
    ("llama_kernels", {"TQ_USE_LLAMA_KERNELS": "1"}),
    ("route_temp_2", {"TQ_MOE_ROUTE_TEMP": "2.0"}),
    ("route_temp_2_llama_kernels", {"TQ_MOE_ROUTE_TEMP": "2.0", "TQ_USE_LLAMA_KERNELS": "1"}),
    ("force_qk_norm", {"TQ_FORCE_QK_NORM": "1"}),
    ("force_qk_norm_llama_kernels", {"TQ_FORCE_QK_NORM": "1", "TQ_USE_LLAMA_KERNELS": "1"}),
]


def parse_case_spec(spec: str):
    if ":" not in spec:
        return spec, {}
    name, rest = spec.split(":", 1)
    env = {}
    rest = rest.strip()
    if rest:
        for item in rest.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"invalid case item '{item}' in '{spec}'")
            k, v = item.split("=", 1)
            env[k.strip()] = v.strip()
    return name.strip(), env


def read_f32_bin(path: Path):
    data = array.array("f")
    with path.open("rb") as f:
        data.frombytes(f.read())
    return data


def topk(values, k=5):
    return sorted(enumerate(values), key=lambda x: x[1], reverse=True)[:k]


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


def run_cmd(cmd, env, stdout_path: Path, stderr_path: Path):
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, env=env)
    return proc.returncode


def find_first_logits_dump(dumps_dir: Path):
    candidates = []
    for p in dumps_dir.glob("logits_p*.bin"):
        try:
            pos = int(p.stem.split("_p")[-1])
        except ValueError:
            continue
        candidates.append((pos, p))
    if not candidates:
        return None, None
    candidates.sort()
    return candidates[0]


def prompt_slug(idx: int):
    return f"p{idx:02d}"


def case_score(metrics):
    return metrics["cosine"] - metrics["rel_l2"]


def main():
    ap = argparse.ArgumentParser(description="Run repeated qwen36 parity loop")
    ap.add_argument("--model", default="models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")
    ap.add_argument("--quant-bin", default="build/quant")
    ap.add_argument("--llama-debug-bin", default="refs/llama.cpp/build/bin/llama-debug")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--n", type=int, default=1, help="number of generated tokens")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="custom case as name:ENV=VALUE,ENV2=VALUE2 . repeatable; overrides defaults if provided",
    )
    ap.add_argument("--keep", type=int, default=0, help="keep output directories (default: 0 => temp)")
    args = ap.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    cases = [parse_case_spec(s) for s in args.case] if args.case else DEFAULT_CASES
    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="qwen36_loop_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    quant_bin = Path(args.quant_bin)
    llama_debug_bin = Path(args.llama_debug_bin)
    model = Path(args.model)

    if not quant_bin.exists():
        print(f"quant binary not found: {quant_bin}", file=sys.stderr)
        return 2
    if not llama_debug_bin.exists():
        print(f"llama-debug binary not found: {llama_debug_bin}", file=sys.stderr)
        return 2
    if not model.exists():
        print(f"model not found: {model}", file=sys.stderr)
        return 2

    base_env = os.environ.copy()
    base_env.update(
        {
            "TQ_NO_METAL": "1",
            "TQ_NO_MLOCK": "1",
            "TQ_QWEN35MOE_NO_PRESET": "1",
            "TQ_NO_MOE_TEMP_AUTO": "1",
            "TQ_DUMP_POS": "all",
        }
    )

    all_rows = []

    for pidx, prompt in enumerate(prompts):
        pslug = prompt_slug(pidx)
        prompt_dir = out_dir / pslug
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        ref_dir = prompt_dir / "llama_debug"
        ref_dir.mkdir(exist_ok=True)
        ref_cmd = [
            str(llama_debug_bin),
            "-m",
            str(model),
            "-p",
            prompt,
            "-n",
            str(args.n),
            "-c",
            str(args.ctx),
            "--temp",
            "0",
            "--threads",
            str(args.threads),
            "--device",
            "none",
            "--fit",
            "off",
            "--no-op-offload",
            "--save-logits",
            "--logits-output-dir",
            str(ref_dir),
        ]
        rc = run_cmd(ref_cmd, base_env, ref_dir / "stdout.txt", ref_dir / "stderr.txt")
        if rc != 0:
            print(f"[{pslug}] llama-debug failed with code {rc}", file=sys.stderr)
            continue

        ref_bin = next(ref_dir.glob("llamacpp-*.bin"))
        ref_logits = read_f32_bin(ref_bin)
        ref_top5 = topk(ref_logits, 5)
        ref_top1 = ref_top5[0][0]

        prompt_rows = []
        print(f"\n[{pslug}] {prompt}")
        print(f"  llama top5 ids: {[idx for idx, _ in ref_top5]}")

        for case_name, case_env in cases:
            case_dir = prompt_dir / case_name
            dumps_dir = case_dir / "dumps"
            dumps_dir.mkdir(parents=True, exist_ok=True)

            env = base_env.copy()
            env.update(case_env)
            env["TQ_DUMP_HIDDEN"] = str(dumps_dir)

            cmd = [
                str(quant_bin),
                str(model),
                "-p",
                prompt,
                "-n",
                str(args.n),
                "-T",
                "0",
                "-j",
                str(args.threads),
                "--ctx",
                str(args.ctx),
            ]
            rc = run_cmd(cmd, env, case_dir / "stdout.txt", case_dir / "stderr.txt")
            if rc != 0:
                row = {"prompt": pslug, "case": case_name, "rc": rc}
                all_rows.append(row)
                prompt_rows.append(row)
                print(f"  {case_name:26s} rc={rc}")
                continue

            dump_pos, dump_path = find_first_logits_dump(dumps_dir)
            if dump_path is None:
                row = {"prompt": pslug, "case": case_name, "rc": 99}
                all_rows.append(row)
                prompt_rows.append(row)
                print(f"  {case_name:26s} no logits dump")
                continue

            logits = read_f32_bin(dump_path)
            case_top5 = topk(logits, 5)
            row = {
                "prompt": pslug,
                "case": case_name,
                "rc": 0,
                "dump_pos": dump_pos,
                "cosine": cosine(logits, ref_logits),
                "rel_l2": rel_l2(logits, ref_logits),
                "score": case_score({"cosine": cosine(logits, ref_logits), "rel_l2": rel_l2(logits, ref_logits)}),
                "top1": case_top5[0][0],
                "top1_match": int(case_top5[0][0] == ref_top1),
                "ref_top1_rank": rank_of(logits, ref_top1),
                "top5_overlap": len(set(idx for idx, _ in case_top5) & set(idx for idx, _ in ref_top5)),
            }
            all_rows.append(row)
            prompt_rows.append(row)
            print(
                f"  {case_name:26s} cos={row['cosine']:.6f} rel_l2={row['rel_l2']:.6f} "
                f"ref_rank={row['ref_top1_rank']:4d} overlap={row['top5_overlap']}/5"
            )

        ranked = [r for r in prompt_rows if r.get("rc") == 0]
        ranked.sort(key=lambda r: r["score"], reverse=True)
        if ranked:
            best = ranked[0]
            print(
                f"  best: {best['case']} score={best['score']:.6f} "
                f"(cos={best['cosine']:.6f}, rel_l2={best['rel_l2']:.6f})"
            )

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "prompt",
        "case",
        "rc",
        "dump_pos",
        "cosine",
        "rel_l2",
        "score",
        "top1",
        "top1_match",
        "ref_top1_rank",
        "top5_overlap",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    agg = {}
    for row in all_rows:
        if row.get("rc") != 0:
            continue
        case = row["case"]
        if case not in agg:
            agg[case] = {"n": 0, "cosine": 0.0, "rel_l2": 0.0, "score": 0.0, "rank": 0.0, "overlap": 0.0}
        agg[case]["n"] += 1
        agg[case]["cosine"] += row["cosine"]
        agg[case]["rel_l2"] += row["rel_l2"]
        agg[case]["score"] += row["score"]
        agg[case]["rank"] += row["ref_top1_rank"]
        agg[case]["overlap"] += row["top5_overlap"]

    if agg:
        print("\n[aggregate]")
        ranked = []
        for case, vals in agg.items():
            n = vals["n"]
            ranked.append(
                (
                    vals["score"] / n,
                    case,
                    vals["cosine"] / n,
                    vals["rel_l2"] / n,
                    vals["rank"] / n,
                    vals["overlap"] / n,
                    n,
                )
            )
        for avg_score, case, avg_cos, avg_l2, avg_rank, avg_overlap, n in sorted(ranked, reverse=True):
            print(
                f"  {case:30s} avg_score={avg_score:.6f} avg_cos={avg_cos:.6f} "
                f"avg_rel_l2={avg_l2:.6f} avg_ref_rank={avg_rank:.1f} avg_top5_overlap={avg_overlap:.2f} n={n}"
            )

    print(f"\nsummary: {csv_path}")
    if args.keep == 0 and not args.out_dir:
        print(f"workspace: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
