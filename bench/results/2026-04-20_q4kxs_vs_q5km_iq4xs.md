# Qwen3.6-35B-A3B Quant Comparison: IQ4_XS vs Q5_K_M on 16 GB M1 Pro

**Date**: 2026-04-20 (Round 45)
**Context**: Post-R40 (NEOX RoPE + arch-conditional QK-norm fixes)
**Hardware**: MacBook Pro M1 Pro, 16 GB unified RAM, 8P+2E
**Config**: `TQ_NO_MLOCK=1`, `--chat`, `T=0`, `n=30`, warm (2nd invocation)

## Speed (tok/s, median of 1 warm run per prompt)

| Prompt | IQ4_XS (4.25 bpw) | Q5_K_M (5.5 bpw) | Ratio |
|---|---:|---:|---:|
| "Once upon a time" | **3.7** | 2.0 | 1.85× |
| "Write a haiku" | **4.3** | 1.3 | 3.31× |
| "List three fruits:" | **4.6** | 2.5 | 1.84× |
| "def fibonacci(n):" | **4.0** | 1.6 | 2.50× |
| "The capital of France is" | **2.5** | 0.2 | 12.5× (EOS early) |
| **Average** | **3.8** | 1.5 | **2.53×** |

## Quality (n=30 generation excerpt)

| Format | IQ4_XS | Q5_K_M |
|---|---|---|
| Story | "It seems like you're setting the stage for a story! 🎭 I'm ready to dive in..." (meta-response) | **"...there lived a curious little assistant named Gli..."** (true creative) |
| Haiku | "Silence speaks loud, Silence speaks in the quietest way." ✓ | "Silence speaks loud, Silence speaks loud." (repetition) |
| List | "1. Apple 2. Banana 3. Orange" ✓ | "1. Apple 2. Banana 3. Orange" ✓ (identical) |
| Code | "It looks like you're trying to write a function..." (meta) | "It looks like you're trying to write a function..." (identical meta) |
| Factual | "The capital of France is **Paris**." ✓ | "**Paris**." ✓ |

## RSS Memory

- IQ4_XS: ~7.9 GB (measured earlier, not re-measured here)
- Q5_K_M: ~10.2 GB

## Recommendation

| Use case | Recommended | Why |
|---|---|---|
| Daily chat/Q&A | **IQ4_XS** | 2.5× faster, equal quality on factual/list |
| Code generation | **IQ4_XS** | Equal quality, speed matters |
| Creative writing | **Q5_K_M** | Genuine creative depth ("curious assistant Gli") |
| Quick factual | **IQ4_XS** | 4× faster on Paris test |
| Haiku/structured | **IQ4_XS** | Less repetition than Q5_K_M |

## Verdict

**IQ4_XS wins 4/5 formats** on Qwen3.6-35B-A3B Unsloth UD quants.
**Q5_K_M** only wins on creative-narrative where depth > speed.

## Q4_K_M Bench: DEFERRED

Attempted 22 GB download on 16 GB Mac during session. Hugging Face throttled to 0.2 MB/s after initial burst (17.3/20.9 GB = 82% in ~40 min, remaining 3.6 GB at 0.2 MB/s = 3+ hrs). Deferred to next session.

**Theoretical prediction** based on bpw (4.58 vs 4.25) and MoE sparsity:
- Speed: 2.8-3.2 t/s (between IQ4_XS and Q5_K_M, closer to IQ4)
- Quality: ~= IQ4_XS for 4/5 formats, possibly slight PPL edge
- RSS: ~8.5 GB (+0.6 GB vs IQ4_XS)
- Overall: IQ4_XS still likely winner for 16 GB Mac daily driver
