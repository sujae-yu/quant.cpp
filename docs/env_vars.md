# Environment Variables

Reference for `TQ_*` runtime env vars. Grouped by purpose. Everything
here is opt-in; defaults are the tested production path.

## Performance / resource controls

| Var | Default | Purpose |
|---|---|---|
| `TQ_NO_METAL` | off | Skip Metal (Apple GPU) path; force CPU-only |
| `TQ_NO_MLOCK` | off | Don't `mlock` the mmap'd weights; lets OS page out cold experts on small machines |
| `TQ_NO_Q4` | off | Skip load-time FP32→internal-Q4 recompression; use on-the-fly GGUF dequant. Quality tradeoff — see `state.md` R5 |
| `TQ_NO_BATCH_PREFILL` | off | Force per-token prefill (disables batched matrix prefill path) |
| `TQ_NO_MOE_BATCH` | off | Opt-out of batched MoE dispatch (default-on). Restores per-token MoE forward |
| `TQ_NO_MOE_BATCH_DYNAMIC` | off | Opt-out of FCFS dynamic dispatch (default-on). Wave-mode expert dispatch instead |
| `TQ_MOE_BATCH_CHUNK` | 8 | Tokens per batched MoE call (1-20 sensible range); larger = more speedup, worse numerical stability above ~20 |
| `TQ_MOE_BATCH_SELFTEST` | off | Route N=1 MoE through batch(N=1) kernel — proves equivalence vs per-token path |
| `TQ_PHI3_SPLIT` | 0 | Phi-3 fused QKV/FFN split to separate Q4 weights. **Off by default** — degrades chat quality per feedback/perf_commits_need_chat_test |
| `TQ_MOE_FAST_EXP` | off | Use Schraudolph fast-exp in MoE SwiGLU (vs exact expf default). ~2% per-call error; may re-introduce long-gen drift |
| `TQ_MOE_ROUTE_TEMP` | `1.0` (auto-flipped to `2.0` on qwen35moe arch at load time) | Softmax temperature on top-K expert routing. **`2.0` extends Qwen3.6-35B coherence from 117 → 200+ tokens** on the "Once upon a time" drift-trigger prompt (measured R26). Auto-detected for qwen35moe at model load (see also `TQ_NO_MOE_TEMP_AUTO`). Other arch default stays `1.0` (identity). Trade: slightly less decisive routing = slightly broader expert mix, but top-K set unchanged. `"Paris"` factual probe still correct at T=2.0 |
| `TQ_NO_MOE_TEMP_AUTO` | off | Disable the qwen35moe auto-default flip. Use if you want the prior baseline T=1.0 behavior on Qwen3.6-35B |
| `TQ_KV_PROBE` | off | Dump per-layer K quantization roundtrip stats (rms, MSE, cosine) at positions 0/25/50/100/200. Useful to verify KV compression is behaving uniformly across layers and not drifting over position. See the R32 finding — turbo_kv_4b holds cosine ≥0.994 across all layers and positions on Llama-3.2-1B |

## Quality / correctness

| Var | Default | Purpose |
|---|---|---|
| `TQ_NO_AUTO_SERIAL` | off | Opt-out of Qwen3.6 auto single-thread mode. Multi-thread is non-deterministic at T=0 — default forces `-j 1` on qwen35moe+DeltaNet hybrid. Cost: ~2-3× slower decode |
| `TQ_FORCE_QK_NORM` | off | Force QK-norm on Qwen hybrid (normally disabled for that arch) |
| `TQ_ROPE_PAIRS` | off | Force LLaMA-style interleaved RoPE pairs (overrides NEOX auto-detect) |
| `TQ_NO_PLE` | off | Disable Gemma-4 per-layer-embedding path |

## Debugging — general

| Var | Default | Purpose |
|---|---|---|
| `TQ_DEBUG` | off | Prints per-layer output norms, attention range, tokenized prompt, etc. |
| `TQ_DEBUG_PREFILL` | off | Per-layer `final x sum` / `sumabs` during prefill (layers 0-3) |
| `TQ_DEBUG_WQ` | off | L0 pre-norm RMS at first token |

## Debugging — refparity framework

The `tools/refparity/` framework uses these to produce comparable dumps
against HF FP32 reference. Do not enable in production — each dump
is a fsync'd file.

| Var | Value | Purpose |
|---|---|---|
| `TQ_DUMP_HIDDEN` | `/path/to/dir` | Dump `emb.bin`, `h0.bin`…`hN.bin`, `post_norm.bin`, `logits.bin` (one raw FP32 file per slot) |
| `TQ_DUMP_POS` | `0` (default) or `N` or `all` | Which token position to dump. `all` is expensive (28 × seq_len files) |
| `TQ_DUMP_INTERMEDIATE` | off | Also dump per-layer sub-stage: `h{l}_in/postattn/preffn/ffnout` — bisects attention vs FFN divergences |

## Debugging — DeltaNet (Qwen3.5/3.6)

Added in the 2026-04-21 DeltaNet investigation. Probe or ablate the
recurrent state to localize drift.

| Var | Value | Purpose |
|---|---|---|
| `TQ_DELTA_PROBE` | `call1,call2,...` | Print per-layer `delta_state` L2 norm at listed layer-0 call counts. E.g. `TQ_DELTA_PROBE=50,100,115,120` |
| `TQ_DELTA_RESET_EVERY` | `N` | Zero `delta_state` + `conv_state` every N-th layer-0 call. Diagnostic only (destroys useful context) |
| `TQ_DELTA_RESET_LAYER` | `N` or unset | Combined with `RESET_EVERY`, clears only that layer's slice. `-1` or unset = all layers |

## Examples

**Reproduce BPE UTF-8 regression suite**:
```bash
bash scripts/test_models.sh    # runs test_tokenizer.sh at tail
```

**Reference-parity diff on one model**:
```bash
export PYTHONPATH=tools/pillar1/venv/lib/python3.12/site-packages
python tools/refparity/hf_reference.py --model Qwen/Qwen3-0.6B --prompt "Hello" --out /tmp/ref.npz
TQ_DUMP_HIDDEN=/tmp/eng TQ_NO_METAL=1 TQ_NO_MLOCK=1 TQ_NO_BATCH_PREFILL=1 TQ_NO_AUTO_SERIAL=1 \
    ./build/quant models/Qwen3-0.6B-Q4_K_M.gguf -p "Hello" -n 1 -T 0
python tools/refparity/diff_layers.py /tmp/ref.npz /tmp/eng
```

**Probe Qwen3.6 DeltaNet state at drift boundary**:
```bash
TQ_DELTA_PROBE=50,100,115,118,120 \
    ./build/quant models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
    -p "Once upon a time in a faraway land" -n 125 -T 0 2>&1 | grep delta-probe
```

**35B best-quality user config**:
```bash
./build/quant models/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf \
    -p "<your prompt>" -n 200 -T 0 --rep-penalty 1.3
```

## Notes

- Most `TQ_NO_*` envs exist because the default path has a correctness
  or quality tradeoff someone wanted to A/B. Flipping them usually
  trades speed for determinism or vice versa. Read `state.md` and
  `bench/results/` for the measured impact before relying on any.
- New envs land with `state.md` entries documenting *why* they exist.
  Don't add undocumented envs.
