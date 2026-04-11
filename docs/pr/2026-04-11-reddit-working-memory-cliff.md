# Reddit r/LocalLLaMA — The Working Memory Cliff

**Title:** `[Research] We measured the working memory cliff of Llama-3.2-1B/3B-Q4 — both fall to 0% retrieval at <1% of their nominal 128K context window`

**Flair:** `Research`

---

## Body

We've been pushing a "load the whole document instead of chunking" position for a while in this sub. Last month we showed [chunk-RAG hallucinated 7/7 questions while full-document inference got 7/7 correct](https://github.com/quantumaikr/quant.cpp/blob/main/docs/beyond-rag-manifesto.md) on a synthetic 600-token document with Llama-3.2-3B and 6.4× KV compression. Several of you pointed out — correctly — that 600 tokens is not a stress test.

So we ran 204 NIAH trials at context lengths 256–2048 to find where the model actually breaks. The result is sharper than we expected.

### The cliff

**Llama-3.2-1B-Instruct-Q8_0** (graded cliff):

| ctx | fp32 KV | 6.4× compressed |
|---:|:-:|:-:|
| 256 | 89% | 89% |
| **512** | **100%** | **100%** |
| 1024 | 44% | 22% |
| 1536 | 0% | 0% |
| 2048 | 0% | 0% |

**Llama-3.2-3B-Instruct-Q4** (default loader; **step function**):

| ctx | fp32 KV | 6.4× compressed |
|---:|:-:|:-:|
| 512 | 100% | 100% |
| **1024** | **100%** | **100%** |
| **1280** | **0%** | **0%** |
| 1536–2048 | 0% | 0% |

**1024 → 1280 is 256 tokens.** 18/18 → 0/18. There is no degradation interval. The model goes from following the chat-template instruction perfectly to completely ignoring it within a single 25% step in context length.

Both models reach effective working memory at **less than 1% of their nominal 128K context window** (1B Q8 ≈ 0.4%, 3B Q4 ≈ 0.78%).

### KV compression is orthogonal to the cliff

We compared 6.4× `turbo_kv_4b -v q4 --k-window 128` against FP32 KV baseline across the same grid. **18 of 20 (model × ctx × method) cells are bit-for-bit identical between baseline and compressed.** The single disagreement is the 1B cliff cell where both methods sit at the noise floor anyway.

We also re-ran the cliff transition with FP32 *weights* (`TQ_NO_Q4=1`) to rule out a quantization confound. Same cliff, same location: 100% at 1024, 0% at 1280, both with FP32 weights. **The cliff is a model property, not a KV-cache property and not a weight-quantization artifact.**

### The failure mode is *not* "I don't know"

Above the cliff, the model produces one of three things. The first two (wikitext continuation, header echoes) are unsurprising. The third one is the consequential one.

**Synthesised hallucination, 1B fp32 ctx=1024:**

> *"In 2023 Boulter was hired as the chief financial officer..."*

The haystack is a Wikipedia article about Robert Boulter (English actor). The needle is "The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023." The model **fused them** — produced a coherent invented sentence grafting the needle's "2023 hire" onto Boulter's biography.

This is the same silent-hallucination failure mode that vector RAG produces on retrieval miss — happening in the regime that was supposed to *eliminate* it.

### Honest scope

- 2 models (1B, 3B), 3 needles, 1 language (English), 1 content domain (Wikipedia bio), 204 trials total.
- We tried 8B (Llama-3.1-Q4_K_M) but each inference takes ~12 min on Metal — full grid is infeasible until we optimize the Q4_K_M kernel. v2 work.
- The 22 pp gap at the 1B cliff cell between fp32 and compressed is not statistically significant at n=9 — we ran into a CLI bug attempting the seed sweep, fixed it mid-round, and left the proper stochastic robustness check for v2.
- We tried 6 prompt formats and finalized on the most permissive one. Format-sensitivity is a separate ceiling worth measuring.

If you can falsify the cliff at a different model, prompt, or language, we want the data.

### Try it

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp
cmake -B build_metal -DTQ_BUILD_METAL=ON && cmake --build build_metal -j8

# 1B Q8 sweep (~30 min on M-series)
MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  NIAH_CONTEXTS="256 512 1024 1536 2048" \
  bash bench/niah_test.sh
```

### Links

- **Tech report (arXiv-style draft)**: `docs/paper/working-memory-cliff.md` in the repo
- **Master table**: `bench/results/niah/master_table.md`
- **Raw CSVs + per-run CLI logs**: `bench/results/niah/`
- **GitHub**: https://github.com/quantumaikr/quant.cpp

### What this changes about "Beyond RAG"

The honest reframing: **Beyond RAG works for documents that fit in the model's *effective* working memory, which is 2–3 orders of magnitude smaller than the nominal context window** for the configurations we measured. Memory-wise, 128K context fits in 9.5 GB on a 16 GB Mac. Retrieval-wise, the 3B Q4 model stops following instructions at ~1024 tokens regardless of compression. Edge vendors making "long-context replaces RAG" claims should publish effective working memory measurements alongside memory allocation numbers — the gap is enormous.

If you have NIAH data at 1B–3B scale on llama.cpp / MLC / ollama defaults, we'd love to compare. We want to know if this is a quant.cpp loader artifact or universal at this regime.
