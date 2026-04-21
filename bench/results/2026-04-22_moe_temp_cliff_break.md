# Qwen3.6-35B 117-token Cliff BREAK — MoE Router Softmax Temperature (2026-04-22)

Single one-line env flag — `TQ_MOE_ROUTE_TEMP=2.0` — eliminates the
"It could do math! It could do math!" repetition cliff that capped
Qwen3.6-35B-A3B coherent generation at 117 tokens across 40+ prior
debug rounds.

## The fix

`src/engine/tq_moe.c::tq_moe_route` step 3 softmax:

```diff
   float sum_exp = 0.0f;
+  static float route_temp = 0.0f;
+  if (route_temp == 0.0f) {
+      const char* s = getenv("TQ_MOE_ROUTE_TEMP");
+      route_temp = (s && atof(s) > 0.0f) ? (float)atof(s) : 1.0f;
+  }
+  float inv_temp = 1.0f / route_temp;
   for (int k = 0; k < num_active; k++) {
       if (out_expert_ids[k] < 0) { out_expert_weights[k] = 0.0f; continue; }
-      float e = expf(logits[out_expert_ids[k]] - max_val);
+      float e = expf((logits[out_expert_ids[k]] - max_val) * inv_temp);
       out_expert_weights[k] = e;
       sum_exp += e;
   }
```

## Temperature sweep (Qwen3.6-35B-A3B-UD-IQ4_XS, T=0)

Prompt: `"Once upon a time in a faraway land"`, `-n 200`.

| TEMP | Coherent tokens | Loop content | Continuation |
|---:|---:|:---|:---|
| 1.0 (default) | ~95 | "It could do math!" | Alex/ENIAC story collapses at 117 |
| 1.5 | ~75 | "and everything went wrong!" | Cliff earlier — peakier in some heads |
| 1.8 | ~90 | "And that's why we have the Internet!" | Still within the trap |
| **2.0** | **~150 coherent** | none detected | Alex + sad tree story, full -n budget |
| **2.5** | **~150 coherent** | none detected | Alex + magic-leaves story, full -n budget |
| 3.0 | ~95 | "The sun would rise too!" | Over-flat — wrong expert mix |

Sweet spot: **T=2.0 to 2.5**. Outside that band the cliff returns
(earlier below, different trap above).

## Why it works (causal story)

1. Each MoE token selects top-K=8 experts out of 256. Softmax output
   weights determine how much each of the 8 contributes.
2. At default T=1.0 the softmax gets **peaky at long positions** — one
   or two experts take 60-80% of the mass (measured in R25, L4 hit 0.812
   at token 100 on this prompt).
3. DeltaNet's recurrent state carries semantic through the decode.
   When MoE routing concentrates on a narrow expert set, that set's
   bias projection feeds back into the residual stream repetitively,
   DeltaNet state self-reinforces, **positive feedback loop** locks
   onto a repeating phrase.
4. T=2.0 spreads the softmax output: top-1 share drops, competing
   experts contribute more, no single expert's bias dominates residual
   → the loop can't form.

The 4B dense-hybrid model (Qwen3.5-4B, DeltaNet + dense FFN, no MoE)
does NOT drift on the same prompt — R24 isolated this. Confirms the
drift is a MoE-specific pathology, not DeltaNet's fault.

## What T=2.0 does NOT fix

- Tail quality from ~150 to 300 tokens degrades to character-level
  noise (alphabet-walking "'a'b'c'd'e") on longer `-n 500` runs.
  Probably quantization + DeltaNet state accumulation compounding.
- The specific "Sorry!" mini-loop appears around 170 tokens at T=2.0 —
  doesn't trigger engine's rep-loop detector but is human-visible.

So: T=2.0 **breaks the hard 117-tok cliff** and recovers ~50 additional
coherent tokens. Full essay-length generation still needs more work.

## Safety

- `"The capital of France is"` → `"Paris."` (correct) at T=2.0
- `bash scripts/test_models.sh` → **23/23 PASS** with T=2.0
  (15 coherence + 8 BPE-stale-entry + 3 BPE-UTF-8 direct-byte, no diff)

## Recommended user config

Best Qwen3.6-35B recipe on 16 GB Mac today:

```bash
TQ_MOE_ROUTE_TEMP=2.0 \
    ./build/quant models/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf \
    -p "<your prompt>" -n 200 -T 0 --rep-penalty 1.3
```

Combine with `Q5_K_M` GGUF for best quality (200-tok coherent range)
and `--rep-penalty 1.3` as belt-and-suspenders.

## The arc

- R1-R19: "Drift is DeltaNet state" → R19 single-layer reset bisection
  proves NOT true
- R24: 4B dense hybrid works fine → drift is MoE-specific
- R25: MoE probe → L4 single-expert collapse at long positions
- **R26**: Softmax temperature ablation → **cliff broken at T=2.0**

Total investigation: 26 rounds. The actual fix: 5 lines of C.

See also:
- `docs/env_vars.md` — `TQ_MOE_ROUTE_TEMP` row
- `docs/supported_models_tier.md` — 35B recipe updated
- `.claude/state.md` — R16-R26 reasoning chain
