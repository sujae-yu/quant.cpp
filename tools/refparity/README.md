# Reference-Parity Framework

## Why

Across 8 releases (v0.19.0 → v0.26.0) we repeatedly found the same class of bug:
**paraphrased reference implementation**. Each fix was correct individually but
cumulatively revealed that our engine drifted from the reference (llama.cpp /
HF transformers) in several subtle ways — different eps formulation, missing
NEOX ordering in batched path, over-broad QK-norm disable, silent prompt
truncation, ...

**The meta-bug: we read reference code and wrote "something similar" rather
than copying it exactly.** Over 30+ rounds of Mission C this cost us real time.

This framework prevents that class of bug by automating the comparison.

## What it does

For each (model, prompt) pair in the test matrix:
1. Run the prompt through HF transformers (FP32 ground truth)
2. Run the same prompt through our engine with `TQ_DUMP_HIDDEN` enabled
3. Compare per-layer hidden states: cosine similarity + L2 relative error
4. Report first layer / position where divergence exceeds threshold
5. CI PASS if no layer exceeds 5% L2_rel; FAIL otherwise with clear diagnostic

## Scope (tier 1 coverage first)

| Model | HF name | Purpose |
|---|---|---|
| Qwen3-0.6B | Qwen/Qwen3-0.6B | Small fast repro; catches Qwen3 family bugs |
| Qwen3.5-4B | Qwen/Qwen3.5-4B | Hybrid (DeltaNet + self-attn) reference |
| Llama-3.2-1B | meta-llama/Llama-3.2-1B | Standard transformer baseline |

Tier 2 architectures (Qwen3.6-35B MoE, DeepSeek) are NOT in the matrix — too
large for 16 GB Mac FP32 HF run. These get llama.cpp diff instead (future).

## Files

- `hf_reference.py` — runs HF model, dumps per-layer hidden states + logits
- `engine_reference.sh` — runs our engine with matching instrumentation
- `diff_layers.py` — layer-by-layer comparison with thresholds
- `run_matrix.sh` — executes full (model × prompt) matrix, reports PASS/FAIL
- `matrix.json` — test matrix definition

## Usage

```bash
# First-time setup (reuses tools/pillar1/venv by default — override with VENV_DIR=...)
python3.12 -m venv tools/pillar1/venv
source tools/pillar1/venv/bin/activate
pip install torch transformers accelerate

# Run full matrix — invoke from project root
bash tools/refparity/run_matrix.sh
FILTER=qwen3 bash tools/refparity/run_matrix.sh     # only entries whose name contains "qwen3"

# Focused single comparison (from project root)
source tools/pillar1/venv/bin/activate
python tools/refparity/hf_reference.py --model Qwen/Qwen3-0.6B --prompt "Hello" --out /tmp/ref.npz
bash tools/refparity/engine_reference.sh models/Qwen3-0.6B-Q4_K_M.gguf "Hello" /tmp/eng
python tools/refparity/diff_layers.py /tmp/ref.npz /tmp/eng
```

Reports land in `tools/refparity/reports/` as `<name>__p<idx>.diff` (one per
prompt) on failure.

## Exit codes

- `0` — all layers within threshold for all (model, prompt) pairs
- `1` — divergence detected; diff report identifies the offending layer
- `2` — environment or configuration error

## Methodology notes

- **Quantization noise baseline**: expect ~1-3% L2_rel per layer due to Q4/Q5
  quantization vs FP32 reference. Threshold set at 5% accordingly.
- **Accumulation compounding**: quantization errors compound across 28-40 layers.
  By layer N-1, total divergence can be 20-30% even with correct engine.
  Per-layer threshold + cosine > 0.9 at logits is the PASS condition.
- **First-divergence bisect**: if failing, the FIRST layer where L2_rel spikes
  above prior-layer baseline is the bug localization point.
- **Position alignment**: the engine dumps `TQ_DUMP_POS=0` (first token) so
  `diff_layers.py` defaults to pos=0 too. Override with `--pos N` to inspect
  later positions (engine-side, set `TQ_DUMP_POS=N` in engine_reference.sh).
- **HF hidden_states layout** (transformers ≥5.x, Qwen3/Llama): 29 entries for
  28-layer model — `(emb, layer0_out, layer1_out, …, layer_{N-2}_out, post_norm)`.
  The LAST element is already post-final-RMSNorm (hf_reference.py maps it to
  `post_norm`). The final layer's pre-norm output is not exposed by HF.

## Known baseline findings (Qwen3-0.6B Q4_K_M)

First end-to-end run on `Qwen3-0.6B-Q4_K_M.gguf`, "Hello" prompt:

| slot | L2_rel | cosine | notes |
|---|---:|---:|---|
| emb | 1.8% | 0.9998 | clean |
| h0–h1 | 15-20% | 0.98 | marginal — Q4 noise amplified in early layers on a 0.6B model |
| h2–h26 | ~3.9% | 0.9997 | steady Q4 quantization baseline |
| post_norm | ~100% | 0.24 | **real divergence — needs investigation** |
| logits | — | 0.51 | top-1 mismatch (HF 21806 vs engine 11) |

Framework correctly identifies the post_norm + logits divergence as a genuine
engine bug (cannot be explained by Q4 quantization alone — mid-layer stays
at 3.9%). This is tracked as a separate investigation; Phase 1's goal is only
to ship the detection infrastructure.
