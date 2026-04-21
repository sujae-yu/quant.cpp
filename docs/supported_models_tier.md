# Supported Models — Honest Tier Matrix

*Last updated: 2026-04-21 (post v0.27.0)*

This page groups supported GGUF models into tiers based on **measured**
quality and stability on a 16 GB M1 Pro Mac, not marketing claims. If a
model isn't listed, we haven't validated it end-to-end and you're on the
edge of supported.

## Tier 1 — Production (coherent + stable long-form)

Works out of the box for chat, code, and multi-hundred-token generation.
Regression suite covers all of these.

| Model | Quant | Decode | TTFT | Notes |
|---|---|---:|---:|---|
| Llama-3.2-1B-Instruct | Q8_0 | 53-57 t/s | 0.12s | Best speed, small vocab |
| Llama-3.2-3B-Instruct | Q8→Q4 | 26-29 t/s | 0.97s | Balanced |
| Phi-3.5-mini-instruct | Q4_K_M | 14-16 t/s | 0.95-2.3s | SentencePiece; best chat quality at ~3.8B |
| Qwen3-0.6B | Q4_K_M | 50-60 t/s | 0.17s | Smallest Qwen3 |
| Gemma-4-e2b | Q8 | 24-25 t/s | 0.46s | Dual-FFN + PLE |

**Recommended for**: user-facing chat, code completion, embedding server,
any single-call or short-multi-turn workload.

## Tier 2 — Stable at short-to-medium generation

Coherent output up to ~200-400 tokens; long-form quality degrades
gradually but remains usable.

| Model | Quant | Decode | Notes |
|---|---|---:|---|
| Qwen3.5-4B | Q4_K_M | 18-23 t/s | Dense DeltaNet hybrid; 561-word prompt coherent |
| Gemma-4-e4b | Q8 | ~3.5 t/s | Slow but stable; research use |

**Recommended for**: one-shot Q&A, short essays. Avoid long narrative
without explicit length guardrails.

## Tier 3 — Experimental / short-generation only

Runs and produces English, but hits repetition loops on long generation.
Use with user-facing guards (`--rep-penalty`, shorter `-n`).

| Model | Quant | Decode | Practical config | Drift boundary |
|---|---|---:|---|---:|
| Qwen3.6-35B-A3B | UD-IQ4_XS | 12-16 t/s warm | `--rep-penalty 1.3` | ~117 tok default; ~200 tok with rep-penalty |
| Qwen3.6-35B-A3B | UD-Q5_K_M | 10-13 t/s warm | `--rep-penalty 1.3` | 200+ tok (hits -n budget, graceful tail degrade) |
| Qwen3.6-35B-A3B | UD-Q3_K_S | 14 t/s warm | shorter `-n` | ~100 tok |

**Status**: The 117-token repetition cliff on Qwen3.6-35B is a
distributed multi-layer DeltaNet-state accumulation phenomenon (see
`.claude/state.md` R16-R19). No single-line fix applies. The
`--rep-penalty 1.3` mitigation is the best user-facing option today.

**Recommended for**: short narrative continuations, summarization of
moderate documents, technical Q&A. Not for >200-token open-ended
generation without guards.

## Not Yet Tested / Not Supported

- Qwen2.5 family — likely works (BPE fix applies) but not in the
  regression suite
- Mistral / Mixtral — no GGUF loader path exercised
- Gemma-4 26B variants — known to fail Metal path (auto-CPU works but
  slow on 16 GB)

## BPE UTF-8 (v0.27.0, all tiers)

Until v0.27.0, international text (accents, CJK, Cyrillic, byte-fallback
emoji) was silently double-encoded on both encode and decode for every
Llama-3 and Qwen3-family model. If you were running a pre-v0.27.0 build
on non-English prompts, please upgrade — prior outputs were in a
different token distribution than the model was trained on.

See `bench/results/2026-04-21_bpe_utf8_fix_proof.md` for the end-to-end
proof and 11/11 HF parity measurements.

## Verification command

Reproducible regression:

```bash
bash scripts/test_models.sh     # 15 coherence tier + 11 tokenizer UTF-8
# → PASS: 15 / 11, FAIL: 0 / 0
```

All numbers above are from 16 GB M1 Pro CPU-only, warm cache. Cold-start
TTFT can be 3-10× higher on first call.
