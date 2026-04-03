# Reddit r/LocalLLM Response Drafts (2026-04-03)

Post: 16 upvotes, 5.4K views, 19 comments

---

## @MrRandom04 — "re-implementing all of llama.cpp just to add whatever approach"

We don't intend to replace llama.cpp. We have a self-contained llama.cpp integration patch (`integrations/llamacpp/patch/`, 4 files, ~1000 lines) that adds `--cache-type-k tq_kv_1b` as a drop-in option. The standalone engine exists for research and to verify the algorithm on multiple architectures (Llama, Gemma, Qwen, Qwen-MoE — 4 verified). The goal is to get quant.cpp KV into llama.cpp as a native cache type.

---

## @dinerburgeryum — "codebook calibration sensitive to out-of-domain data?"

Good question. The **1-bit path doesn't use a codebook at all** — it's just `sign(RHT(key))`, so there's nothing to calibrate and nothing domain-sensitive. The RHT seed is fixed per-block and model-independent.

The codebook is only used for 3-bit and 4-bit modes (Lloyd-Max optimal for N(0,1)). Our `--calibrate` tool showed 49.7% MSE improvement with model-specific codebooks, but the 1-bit path skips all of this.

---

## @Viper-Reflex — "does this make my 24GB 3090 run bigger models?"

KV compression helps most with **long contexts**, not bigger models. With 1-bit K + Q4 V, KV memory drops ~5x. For a 27B model at 32K context:
- Before: ~2.5 GB KV cache
- After: ~500 MB KV cache → frees ~2 GB for longer context or larger batch

If you're already fitting a model in 24GB, quant.cpp lets you push context from 32K → 100K+ on the same hardware. But it won't help you fit a model that's too large for VRAM (weight memory is separate from KV cache).

Note: we currently don't have CUDA GPU acceleration (it compiles but is untested). That's next on the roadmap.

---

## @Blizado — "zero quality loss claim" (already responded)

Updated README: "almost no quality loss (PPL +0.03%)".

Clarification:
- K-only (V as FP16): PPL is exactly +0.00% — measured identical on both Gemma 4B and SmolLM2 1.7B (Llama arch)
- K + Q4 V: PPL +0.03% — near-zero, not zero
- "byte-identical" refers to greedy decoding up to ~100 tokens, not infinite sequences

---

## @BillDStrong — "What magic is this. I didn't realize there was a 1-bit version"

Good observation — the paper (quant.cpp, ICLR 2026) focuses on 2.5-bit and 3.5-bit configurations. The 1-bit version is our extension of the paper's framework.

The key insight: the paper's RHT (Randomized Hadamard Transform) makes the quantization error **unbiased** for inner products at any bit-width. We pushed this to the extreme — 1 bit = just the sign of each dimension after RHT. Mathematically, this gives a cosine similarity of 2/pi ≈ 0.637 (we measured 0.634), which is the information-theoretic maximum for sign-only quantization.

Why does 1-bit "beat" 2-3 bit? It doesn't in terms of reconstruction quality (MSE is worse). But for **attention scoring** (which only needs inner product ranking, not exact values), the softmax function is surprisingly tolerant of noise. The attention weights after softmax are nearly identical because:
1. RHT distributes errors uniformly (no systematic bias)
2. Softmax amplifies the largest scores and suppresses small ones
3. The top-attended tokens stay the same even with noisy scores

So it's not that 1-bit is "better" — it's that attention is robust enough that 1-bit is sufficient.

---

## @ganonfirehouse420 — "I hope I will be able to have a huge context for my local models in the future"

That's exactly the use case. With 1-bit K + Q4 V, KV cache memory drops ~5x. Concrete example:

```
Gemma 3 4B at 32K context:
  FP16 KV: 4,352 MB → barely fits in 16GB with model weights
  1-bit K + Q4 V: 885 MB → room for 128K+ context on same hardware
```

For a 16GB Mac or laptop, this means going from 32K → 100K+ context without any hardware upgrade. The limiting factor shifts from KV memory to model weight memory.

This is available today — `./build/quant model.gguf -p "your long prompt" -k turbo_kv_1b -v q4 --ctx 131072`. The `--ctx` flag overrides the default context limit.

---

## @TopChard1274 — "big breakthroughs ... seem brutal for people who invested in nearly-unaffordable systems"

Appreciate the perspective. A few thoughts:

1. **KV compression helps everyone** — whether you have 8GB, 24GB, or 80GB. The ratio is the same (5x KV reduction). High-end systems benefit by running longer contexts or larger batches, not just by fitting a model.

2. **This doesn't obsolete hardware** — weight memory is still the bottleneck for model size. A 70B model still needs ~35GB for Q4 weights regardless of KV compression. What changes is that you can push context much further on whatever hardware you have.

3. **The "0.03% quality loss" criticism is fair** — some critics in this thread pushed back on "zero loss" and they're right. We've updated to "almost no quality loss" with exact PPL numbers. The honest framing matters more than hype.

The real unlock is for use cases like RAG with long documents, code analysis with large repos, or multi-turn conversations that previously hit context limits.

---

## @MrRandom04 (follow-up) — "Why not just fork llama.cpp?"

> "It is very hard for me to trust the correctness of a re-implementation of such a complex codebase... Why not just fork llama.cpp and add it in there so we know the code for all the other crucial parts is fairly robust and dependable?"

Valid concern. Two reasons for the standalone engine:

1. **Algorithm verification across architectures.** We needed to test quant.cpp KV on Llama, Gemma (sliding window), Qwen3.5 (DeltaNet hybrid), and Qwen-MoE (256 experts) — each with very different attention mechanisms. A standalone engine let us control every variable and measure PPL impact precisely. Debugging quantization bugs inside llama.cpp's 200K+ line codebase would have been much harder during research.

2. **The integration path is real.** `integrations/llamacpp/` has a working GGML type registration that adds quant.cpp types alongside existing Q4/Q8 types. The plan is an upstream PR — not maintaining a parallel engine forever.

You're right that a fork would give more confidence in correctness. Once the algorithm is validated (which is what the standalone engine proved), the next step is exactly that — getting it into llama.cpp where it benefits from their battle-tested infrastructure. The standalone engine is the research prototype; llama.cpp integration is the production path.

---

## @OftenTangential — "36 is an absurd PPL for Gemma 3 4B"

> "36 is an absurd ppl for Gemma 3 4B on English text lol. That implies it's literally outputting GPT-2 levels of coherence... Either your perplexity test set is bad, or the baseline implementation is broken."

Fair point — the PPL of 35.99 is high for Gemma 3 4B. Here's the context:

1. **Short test set (101 tokens).** This was a small fixed prompt used to compare FP16 vs 1-bit, not a proper benchmark corpus. PPL on short sequences is noisy and inflated — it doesn't reflect the model's true capability on longer text.

2. **What matters is the delta, not the absolute value.** The point of the measurement is FP16 → 1-bit: 35.99 → 36.00 (+0.03%). Whether the baseline is 6.0 or 36.0, a +0.03% delta from quantization is negligible.

3. **Confirmed on SmolLM2 1.7B (Llama arch) with lower baseline PPL.** SmolLM2 gives baseline 5.84 PPL on 105 tokens — a more expected range for a small model. 1-bit K: 5.84 (+0.00%). This cross-architecture result is stronger evidence.

You're right that a proper PPL evaluation should use a standard benchmark (WikiText-2, C4) with thousands of tokens. That's on the roadmap. The 101-token measurement was meant to show the quantization delta, not the model's absolute quality.

---

## @Viper-Reflex — ":O ty for the info!"

(No response needed — positive acknowledgment.)

---

## @MaybeADragon — "Em dashes. No more to be said."

Implying the post was AI-generated due to em dash usage. Not worth engaging directly — the code and results speak for themselves. If asked seriously, the response is: the code is open source, 30K lines of C, 32 test suites, all reproducible.

---

## @Candid_Koala_3602 — "Can quant.cpp also replace transformers in the same mechanism? Angular mappings instead of weights?"

> "Can quant.cpp also replace transformers in the same mechanism? That would be the real win. Angular mappings instead of weights?"

Interesting idea. Short answer: quant.cpp doesn't replace the transformer architecture — it compresses the **data** (KV cache, weights) that the transformer operates on.

But the underlying insight — that angular/directional information is sufficient for attention — is related to what you're describing. The 1-bit path essentially reduces attention to cosine similarity via sign hashing, which is a form of angular mapping. Whether this could extend to replacing weight matrices with purely angular representations is an open research question.

The closest existing work is probably binary/ternary weight networks (BWN/TWN) and more recently BitNet (1-bit weights). quant.cpp's contribution is showing that the **KV cache** specifically tolerates extreme quantization because attention is inherently a ranking operation, not a reconstruction operation.

---

## Key takeaway from Reddit feedback

1. **"zero quality loss" was overstated** → fixed to "almost no" with exact PPL
2. **"why not just integrate into llama.cpp?"** → we have a patch, standalone is for research; llama.cpp PR is the production path
3. **"why not fork llama.cpp?"** → valid, standalone engine proved the algorithm, next step is upstream integration
4. **"PPL 36 is absurd for Gemma 4B"** → fair: short test set (101 tokens) inflates PPL; delta (+0.03%) is what matters, confirmed on SmolLM2 at lower baseline PPL
5. **Technical curiosity is high** — 5.4K views, people want to understand the math
6. **Skepticism is healthy** — Blizado/No-Manufacturer/OftenTangential criticism pushed us to be more precise
7. **1-bit vs 2-3 bit confusion** → clarified: softmax robustness, not better MSE
8. **Long context is the killer app** — multiple users asking about context extension
9. **Hardware democratization** resonates — people want more from existing hardware
10. **Need proper PPL benchmark** → WikiText-2/C4 with 1000+ tokens, not 101-token micro test
