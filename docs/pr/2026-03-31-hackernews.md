# Hacker News Post — 2026-03-31

## Title

Show HN: Pure C LLM engine with 3.8x KV cache compression (9K lines, zero deps)

## URL

https://github.com/quantumaikr/quant.cpp

## Comment (post immediately after submission)

Hi HN, I built a LLM inference engine in pure C that compresses the KV cache during inference.

The problem: at long contexts (32K+ tokens), the KV cache — not the weights — becomes the memory bottleneck. A 4B model at 32K context needs 4.4 GB just for KV in FP16.

quant.cpp quantizes the KV cache to Q4 on-the-fly, reducing that to 1.2 GB (3.8x compression) with 0.999 cosine similarity to FP16 output. Based on three recent papers: quant.cpp (ICLR '26), QJL (AAAI '25), PolarQuant (AISTATS '26).

Technical details:
- 9,000 lines of C11, libc only, no external dependencies
- Q4 weight quantization with ARM NEON 2-row batching
- Thread pool, integer Q4×Q8 attention (vdotq_s32)
- Multi-architecture: Qwen3.5 (DeltaNet hybrid) + Gemma 3 (sliding window)
- Dual tokenizer: GPT2 byte-level BPE + SentencePiece
- TQM format: pre-quantized mmap binary for instant loading

Speed matches llama.cpp single-thread (51 vs 50.7 tok/s on Qwen3.5-0.8B Q4). The value is memory efficiency at long contexts, not raw speed.

Built in 2 days with Claude Code. v0.1.0 just released.

---

## Posting Notes

- **Best time**: US weekday morning (UTC 14:00-16:00)
- **HN audience cares about**: technical depth, honesty, zero-dep C code, paper implementations
- **Avoid**: marketing language, speed claims without context, "revolutionary" etc.
- **Expected questions**:
  - "How does KV quantization affect perplexity?" → 0.999 cosine, per-layer verified
  - "Why not contribute to llama.cpp?" → Different approach (KV compression is orthogonal)
  - "Built in 2 days?" → With AI assistance (Claude Code), honest about it
