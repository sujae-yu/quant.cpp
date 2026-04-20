# Long-Sequence Transformer Bug — Discovered 2026-04-20

**After R3 BPE fix**, tokenization is now correct. But a **SEPARATE
transformer-forward bug** manifests on inputs of ~100+ tokens. This
document records the reproducer for follow-on debugging.

## Reproducer (Qwen3-0.6B Q4_K_M, deterministic)

```bash
# Generate 50 synthetic words = ~144 tokens after tokenization
WORDS=$(python -c "print(' '.join(f'word{i}' for i in range(1,51)))")
./build/quant models/Qwen3-0.6B-Q4_K_M.gguf -p "$WORDS. Continue:" -n 12 -T 0
```

## Observed behavior (binary search by prompt length)

| Prompt words | Token count | Our output | HF output (reference) |
|---|---:|---|---|
| 20 | 54   | `"word., check ( (-lang PL"`  | (coherent) |
| 30 | 84   | `"nahme innocence thisds..."` | (coherent) |
| 40 | 114  | **UTF-8 garbage** (`è�� ç�° å�¨egan...`) | (coherent) |
| 50 | 144  | UTF-8 garbage                | `" word51 word52 word53 word54"` ✓ |
| 70 | 204  | UTF-8 garbage                | (coherent) |
| 90 | 264  | UTF-8 garbage                | (coherent) |

**Break threshold: ~100-120 prompt tokens** where Qwen3-0.6B Q4_K_M
transitions from semi-coherent to UTF-8 byte garbage.

## Why this is NOT the R3 BPE bug

The R3 fix (tq_tokenizer.c:1442) made our tokens match HF. Token IDs
are correct across all prompt lengths. So any remaining divergence is
in the **transformer forward** or **KV cache** pipeline.

Confirmed: our Qwen3-0.6B produces token 9707 for "Hello" (matching HF).

## Why this is NOT just a small-model artifact

- Qwen3-0.6B HF at FP32 produces coherent output on 144-token input.
- Qwen3.6-35B on 235-word clean English also garbles (via repetition
  loop detection).
- Both models fail at similar relative thresholds when run through
  our engine.

## Candidate root causes (for follow-on debugging)

1. **KV cache quantization degradation past N tokens**
   - Default `turbo_kv_4b` KV compression. Test: `TQ_KV_TYPE=fp32` to
     isolate.
2. **Batched prefill path** (`tq_forward_batch`)
   - Check per-token-baseline: `TQ_NO_BATCH_PREFILL=1` to force single-
     token forward, compare if threshold changes.
3. **Partial rotary RoPE with growing position** at pos≥ threshold
   - Qwen3 has `rope_theta=1000000`; large `pos` values may hit a
     numerical issue in sin/cos table.
4. **Attention dispatch branch at seq_len > 128 / > 256**
   - Some kernels have special-case paths for long sequences.

## Next steps (methodology)

Apply Pillar 1 methodology to transformer forward:
- Run HF Qwen3-0.6B on the 144-token reproducer, capture all 28
  post-layer hidden states + logits (at LAST position).
- Add `TQ_DUMP_POS=last` to our engine so dump fires at the LAST
  prefill position instead of pos=0.
- Diff layer-by-layer. First layer with L2 diff > 1% of the HF norm
  is the divergence point. Bisect further into attn/FFN/norm sub-steps.

## Impact

R3 BPE fix still delivered real value: short-prompt coherence on all
Qwen3 models + Phi-3 math + improved overall engine quality.

Long-doc use cases (document Q&A, code review, long narrative
continuation) remain blocked by this separate bug until fixed.

Pillar 2 (long prefill speed) is also affected — even if we speed it
up 10×, the output would still be garbage, so speed work should wait
until this is fixed.
