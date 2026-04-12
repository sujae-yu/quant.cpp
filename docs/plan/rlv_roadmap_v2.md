# RLV Roadmap v2 — Post-Breakthrough

> **Date**: 2026-04-12
> **Model**: Phi-3.5-mini (Q8_0, fixed)
> **Language**: English (fixed)
> **Baseline**: 19/20 (95%) on 20-question Wikitext stress test
> **Key constraint**: speed (currently ~4min/question, target ~30s)

---

## Current State

| Metric | Value |
|--------|-------|
| Accuracy | 19/20 (95%) |
| Speed | ~4 min/question |
| Model | Phi-3.5-mini Q8_0, 3.8B, CPU |
| Server | quant-server-unified (quant.h) |
| Inference | ~6.5 tok/s |
| Bottleneck | Every question = locator LLM + lookup LLM + verifier LLM = 3+ inference calls |

## Speed Budget Analysis

```
Current per-question breakdown (~240s):
  Locator LLM call:     ~15s (classify chunk)
  Lookup LLM call:      ~20s (extract answer, 64 tokens)
  Verifier:             ~5s  (literal check, fast)
  Research retries:     ~200s (0-3 extra locate+lookup cycles)

Without retries (~40s):
  Locator + Lookup + Verify = ~40s per question

Target: 30s without retries, ~60s with 1 retry
```

---

## Phase 1: Speed (Week 1) — "4min → 30s"

### 1.1 Eliminate locator LLM call
**Impact: -15s/question**

The LLM classification was unreliable (Loop 5 finding: RRF-first beats LLM). Remove the LLM call entirely from the locator — use pure BM25+keyword RRF.

```python
# Before: BM25 + keyword + LLM call (15s)
# After:  BM25 + keyword only (0.01s)
```

### 1.2 KV cache pre-build with save_context
**Impact: -10s on lookup (eliminates prefill)**

Pre-compute KV cache for each chunk during gist stage. At lookup time, `load_context` instead of re-prefilling.

```python
# During gist build (one-time):
for chunk in chunks:
    ctx = quant_new(model, config)
    quant_generate(ctx, chunk.text, null_callback, null)  # prefill
    quant_save_context(ctx, f"cache/{chunk.id}.kv")

# During lookup (per-question):
ctx = quant_new(model, config)
quant_load_context(ctx, f"cache/{chunk_id}.kv")  # instant
quant_generate(ctx, question, on_token, data)     # generate only
```

### 1.3 Reduce max_tokens
**Impact: -5s per LLM call**

Most answers are <20 tokens. Reduce from 64 to 24 for lookup, 8 for locator.

### 1.4 Parallel-safe server (connection reuse)
**Impact: -2s (eliminate server restart overhead)**

Keep server running across questions. Current overhead: 0.5s startup per question × 3 calls = 1.5s waste.

### Phase 1 Target

| Metric | Before | After |
|--------|--------|-------|
| Locator | 15s (LLM) | **0.01s** (BM25 only) |
| Lookup | 20s | **10s** (KV cache + fewer tokens) |
| Verify | 5s | **2s** (literal only, no LLM fallback) |
| Retries | 200s | **30s** (faster per-retry) |
| **Total (no retry)** | **40s** | **~12s** |
| **Total (1 retry)** | **~80s** | **~25s** |

---

## Phase 2: Robustness (Week 2) — "Always correct or honestly uncertain"

### 2.1 Unanswerable question detection
Test with questions that have NO answer in the document. RLV should return "I don't know" with high confidence.

### 2.2 Long document scaling (100K+ tokens)
Test with 100K token documents (50+ chunks). Verify locator BM25 accuracy doesn't degrade.

### 2.3 Adversarial questions
- Misleading questions ("What year did Du Fu visit Paris?" — never happened)
- Ambiguous questions ("Who is the poet?" — multiple poets in document)
- Questions spanning multiple chunks

### 2.4 Accuracy regression suite
Freeze the 20-question Wikitext + 7-question Acme as a CI regression test. Any code change must pass 26/27+.

---

## Phase 3: Depth (Week 3) — "Harder questions"

### 3.1 Comparison questions
"Compare Du Fu's early career with his later years" → requires reading 2+ chunks and synthesizing.

### 3.2 Query decomposition
"What was happening in Du Fu's personal life when the An Lushan Rebellion began?" → decompose into:
1. "When did the An Lushan Rebellion begin?" → 755
2. "What was Du Fu's situation in 755?" → family, famine

### 3.3 Multi-document RLV
Feed 3 separate documents. "Compare the careers of Robert Boulter and Du Fu" → cross-document synthesis.

---

## Phase 4: Product (Week 4) — "People can use it"

### 4.1 CLI integration
```bash
quantcpp rlv --doc document.txt "What is the main argument?"
```

### 4.2 HTTP API
```bash
quantcpp rlv serve --doc document.txt --port 8080
curl localhost:8080/ask -d '{"question": "..."}'
```

### 4.3 Technical report
"RLV: Cliff-Aware Document QA for Small Language Models"
- Cliff measurement methodology
- RLV architecture + Karpathy loop evolution
- 19/20 result + ablation study
- Comparison: RLV vs RAG vs Long-context

---

## Success Criteria

| Phase | Gate | Metric |
|-------|------|--------|
| 1 | Speed | < 30s/question (no retry) |
| 2 | Robustness | 90%+ on new 20 questions + 100% unanswerable detection |
| 3 | Depth | Comparison questions working |
| 4 | Product | End-user can run `quantcpp rlv` |

---

## Fixed Decisions

| Decision | Rationale |
|----------|-----------|
| English only | Focus on methodology, not multilingual tokenization |
| Phi-3.5-mini Q8_0 | Best speed/quality ratio (32K vocab, 6.5 tok/s) |
| CPU only | Democratization — runs on any laptop |
| quant.h unified server | Proven correct (no libturboquant sync bugs) |
