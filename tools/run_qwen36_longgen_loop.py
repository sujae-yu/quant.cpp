#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path


DEFAULT_CASES = [
    ("baseline", {}),
    ("llama_kernels", {"TQ_USE_LLAMA_KERNELS": "1"}),
    ("rt15", {"TQ_MOE_ROUTE_TEMP": "1.5"}),
    ("rt20", {"TQ_MOE_ROUTE_TEMP": "2.0"}),
    ("rt25", {"TQ_MOE_ROUTE_TEMP": "2.5"}),
    ("qkn", {"TQ_FORCE_QK_NORM": "1"}),
    ("qkn_lk", {"TQ_FORCE_QK_NORM": "1", "TQ_USE_LLAMA_KERNELS": "1"}),
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


def extract_generated_text(stdout: str):
    marker = "---"
    idx = stdout.find(marker)
    if idx < 0:
        return stdout.strip()
    return stdout[idx + len(marker):].strip()


def word_count(text: str):
    return len([w for w in text.split() if w])


def unique_word_ratio(text: str):
    words = [w.strip(".,!?;:\"'()[]{}").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def repeated_tail_score(text: str):
    words = [w for w in text.split() if w]
    if len(words) < 8:
        return 0
    tail = words[-24:]
    best = 1
    for span in range(1, min(6, len(tail) // 2 + 1)):
        pat = tail[-span:]
        reps = 1
        pos = len(tail) - span * 2
        while pos >= 0 and tail[pos:pos + span] == pat:
            reps += 1
            pos -= span
        if reps > best:
            best = reps
    return best


def score_case(row):
    penalty = 0.0
    if row["repetition_detected"]:
        penalty += 50.0
    penalty += max(0, row["tail_repeat"] - 1) * 5.0
    return row["words"] * row["unique_ratio"] - penalty


def main():
    ap = argparse.ArgumentParser(description="Run repeated qwen36 long-generation loop")
    ap.add_argument("--model", default="models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf")
    ap.add_argument("--quant-bin", default="build/quant")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--case", action="append", default=[])
    args = ap.parse_args()

    cases = [parse_case_spec(s) for s in args.case] if args.case else DEFAULT_CASES
    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="qwen36_longgen_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base.update(
        {
            "TQ_NO_METAL": "1",
            "TQ_NO_MLOCK": "1",
            "TQ_QWEN35MOE_NO_PRESET": "1",
            "TQ_NO_MOE_TEMP_AUTO": "1",
        }
    )

    rows = []
    print(f"prompt: {args.prompt}")
    for case_name, case_env in cases:
        env = env_base.copy()
        env.update(case_env)

        case_dir = out_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = case_dir / "stdout.txt"
        stderr_path = case_dir / "stderr.txt"
        cmd = [
            args.quant_bin,
            args.model,
            "-p",
            args.prompt,
            "-n",
            str(args.n),
            "-T",
            "0",
            "-j",
            str(args.threads),
            "--ctx",
            str(args.ctx),
        ]
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            rc = subprocess.run(cmd, stdout=out, stderr=err, env=env).returncode

        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        text = extract_generated_text(stdout)
        rep = bool(re.search(r"repetition loop detected after (\d+) tokens", stderr))
        m = re.search(r"repetition loop detected after (\d+) tokens", stderr)
        rep_after = int(m.group(1)) if m else -1
        row = {
            "case": case_name,
            "rc": rc,
            "words": word_count(text),
            "unique_ratio": unique_word_ratio(text),
            "tail_repeat": repeated_tail_score(text),
            "repetition_detected": int(rep),
            "repetition_after": rep_after,
            "score": 0.0,
            "preview": text[:160].replace("\n", " "),
        }
        row["score"] = score_case(row)
        rows.append(row)
        rep_msg = f" rep@{rep_after}" if rep else ""
        print(
            f"  {case_name:18s} words={row['words']:3d} uniq={row['unique_ratio']:.3f} "
            f"tailrep={row['tail_repeat']} score={row['score']:.2f}{rep_msg}"
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    print("\n[ranked]")
    for row in rows:
        rep_msg = f" rep@{row['repetition_after']}" if row["repetition_detected"] else ""
        print(f"  {row['case']:18s} score={row['score']:.2f} words={row['words']} uniq={row['unique_ratio']:.3f}{rep_msg}")

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "rc",
                "words",
                "unique_ratio",
                "tail_repeat",
                "repetition_detected",
                "repetition_after",
                "score",
                "preview",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nsummary: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
