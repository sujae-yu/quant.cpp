# r/LocalLLaMA Post — 2026-03-31

## Title

quant.cpp — Pure C inference engine with 3.8x KV cache compression. Runs Gemma 3 4B at 32K context using 1.2 GB KV instead of 4.4 GB.

## Body

We built a C inference engine from scratch focused on one thing llama.cpp doesn't do: **compressing the KV cache**.

At short contexts, KV memory doesn't matter much. But at 32K+ tokens, it becomes the dominant memory cost — often larger than the model weights themselves.

**The numbers (Gemma 3 4B):**

```
Context     llama.cpp KV (FP16)    quant.cpp KV (Q4)    Saved
─────────   ──────────────────     ──────────────────    ──────
4K tokens          544 MB                145 MB           399 MB
32K tokens       4,352 MB              1,156 MB         3,196 MB
128K tokens     17,408 MB              4,624 MB        12,784 MB
```

3.8x compression with verified output quality (per-layer exact match against PyTorch).

**Speed is competitive, not the selling point:**
- Single-thread Q4: 51.1 tok/s (llama.cpp: 50.7 tok/s) on Qwen3.5-0.8B
- Same ballpark. We're not claiming to be faster.

**What's different:**
- 3.8x KV cache compression (quant.cpp/PolarQuant/QJL algorithms from ICLR 2026)
- 3 models: Gemma 3 4B, Qwen3.5-0.8B, Gemma 3 270M
- Pure C, zero dependencies, ~1MB binary
- Multi-architecture: DeltaNet hybrid (Qwen) + sliding window (Gemma)
- Gemma 4 ready (same architecture family)

**Quick start:**
```bash
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

Built in 2 days. 9,000 lines of C. 20 test suites. First release: v0.1.0.

The KV compression matters most for long context on limited RAM — exactly the scenario local LLM users care about.

GitHub: https://github.com/quantumaikr/quant.cpp

---

## Posting Notes

- **Flair**: `New Model` or `Resource`
- **Best time**: UTC Tue-Thu 1-3 PM (US East morning)
- **Expected questions**:
  - "What about quality degradation?" → 0.999 cosine similarity, per-layer PyTorch match
  - "vs llama.cpp?" → Same speed, different value prop (KV compression)
  - "Only 3 models?" → Multi-arch engine, more coming. Gemma 4 ready.
  - "Q4 KV vs FP16 isn't fair" → Both are valid choices. We offer the option llama.cpp doesn't.
