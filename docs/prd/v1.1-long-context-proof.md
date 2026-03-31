# TurboQuant.cpp v1.1 PRD — Long Context Proof

**Version**: 1.1
**Date**: 2026-03-31
**Status**: Draft
**Author**: Product / Engineering

---

## Overview

TurboQuant.cpp v1.1 proves the practical value of KV cache compression by supporting 3B+ parameter models and demonstrating measurable memory savings at long context lengths (8K-32K tokens). The release culminates in a public benchmark showing that TurboQuant continues inference where llama.cpp runs out of memory.

Current state: 47 stars, 9 forks, two toy-sized models (270M, 0.8B), no GitHub Release, no long-context proof. v1.1 fixes all three gaps.

---

## Objectives

1. **Practical model support**: Run at least one 3B+ model with verified output quality (cosine similarity > 0.99 vs reference).
2. **Long context proof**: Produce a reproducible benchmark showing 5-7x KV memory reduction at 32K context, including an OOM crossover chart.
3. **First public release**: Ship GitHub Release v0.1.0 with pre-built binaries and benchmark data.
4. **Community traction**: Post benchmark results to r/LocalLLaMA and Hacker News with concrete data.

---

## Scope

### Phase A: Larger Model Support

**Goal**: Make TurboQuant useful beyond toy models.

**Target model** (pick one, in priority order):
1. Llama 3.2 3B — widest community adoption
2. Qwen3.5-3B — existing Qwen architecture support reduces work
3. Gemma 3 4B — existing Gemma architecture support

**Required changes**:

| Area | Current Limitation | Required Change |
|------|-------------------|-----------------|
| Buffer allocation | Stack-allocated `float[4096]` arrays | Dynamic allocation based on model config (`n_embd`, `n_head`, `n_ff`) |
| Intermediate dimensions | Hardcoded or capped at 4096 | Read from model config, allocate at init time |
| Weight loading | Assumes small weight files | Streaming/mmap loading for multi-GB safetensors |
| Memory budget | No tracking | Add peak memory tracking and reporting |
| KV cache sizing | Sized for small models | Scale with `n_layers * n_heads * head_dim * max_seq_len` |

**Deliverables**:
- [ ] Refactor all stack-allocated per-layer buffers to heap allocation sized from model config
- [ ] Implement or extend architecture dispatch for the chosen 3B model
- [ ] Converter script (`tq_convert`) handles 3B+ safetensors
- [ ] End-to-end inference produces coherent text verified against PyTorch reference
- [ ] Document supported model in README

### Phase B: Long Context Benchmark

**Goal**: Prove KV compression matters with hard numbers.

**Benchmark design**:
- **Context lengths**: 1K, 2K, 4K, 8K, 16K, 32K tokens
- **Measurements**: KV cache memory (bytes), peak RSS, tokens/sec, output quality
- **Comparison**: TurboQuant (PolarQuant 3-bit KV) vs llama.cpp (FP16 KV)
- **Hardware**: 8GB RAM machine (or constrained via `ulimit`)

**Key experiments**:

1. **Memory scaling chart**: X-axis = context length, Y-axis = KV cache memory. Two lines: TurboQuant vs llama.cpp. Should show ~7x gap widening linearly.

2. **OOM crossover**: Find the context length N where llama.cpp exceeds available memory but TurboQuant still runs. For a 3B model on 8GB RAM, this crossover should be around 16K-32K tokens.

3. **Quality preservation**: At each context length, measure output cosine similarity to prove compression does not degrade quality at long contexts.

**Deliverables**:
- [ ] `bench/long_context.sh` — automated benchmark script
- [ ] `bench/plot_memory.py` — generates PNG chart from benchmark data
- [ ] CSV output with raw numbers for reproducibility
- [ ] Chart showing memory crossover point
- [ ] Quality metrics at each context length

### Phase C: Release and Community

**Goal**: Make the proof visible.

**GitHub Release v0.1.0**:
- [ ] Tag `v0.1.0` on main
- [ ] Pre-built binaries: macOS ARM64 (`tq_run`, `tq_convert`)
- [ ] Pre-built binaries: Ubuntu x86-64 (via GitHub Actions or cross-compile)
- [ ] CHANGELOG.md with feature summary
- [ ] Release notes include benchmark chart and key numbers

**README update**:
- [ ] Long context benchmark chart (the memory scaling PNG)
- [ ] Updated model table with 3B model
- [ ] "Why KV compression matters" section with concrete numbers

**Community posts**:
- [ ] r/LocalLLaMA post: lead with the OOM crossover chart, explain what KV cache compression enables
- [ ] Hacker News: "Show HN" with the benchmark as the hook
- [ ] Prepare responses for expected questions (quality loss, model support, vs llama.cpp)

---

## Non-Goals

- GPU or Metal backend support
- New quantization types beyond existing PolarQuant/QJL/Uniform
- Speed improvements or multi-threading optimization
- Supporting more than one new model architecture
- GGUF format compatibility
- Speculative decoding or other advanced inference features
- Windows support

---

## Stakeholders

| Role | Who | Interest |
|------|-----|----------|
| Developer / Maintainer | Core team | Architecture decisions, implementation |
| Early adopters | GitHub stargazers (47) | Want to run useful models, need proof of value |
| r/LocalLLaMA community | ~500K members | Care about practical benchmarks, memory efficiency |
| Potential contributors | Forks (9) | Need clear architecture, build instructions, tests |

---

## User Personas

**Alex — Memory-Constrained Hobbyist**
Runs LLMs on a 8GB laptop. Cannot use llama.cpp for long conversations because KV cache eats all RAM. Wants to chat with a 3B model at 16K+ context without OOM.

**Sam — LLM Framework Evaluator**
Evaluates inference engines for integration. Needs benchmark data comparing memory usage. Will not consider TurboQuant without reproducible numbers on a real model.

---

## Technical Constraints

- **Language**: Pure C11 core, no external dependencies (libc/libm only)
- **Backward compatibility**: Must not break existing Qwen3.5-0.8B and Gemma 3 270M support
- **SIMD**: All NEON code must have scalar fallback for x86 CI
- **Platforms**: macOS ARM64 (primary), Ubuntu x86-64 (CI and release)
- **Testing**: All new code must have unit tests. Existing tests must continue to pass.

---

## Success Criteria

| # | Criterion | Measurement | Target |
|---|-----------|-------------|--------|
| 1 | 3B+ model runs | End-to-end inference, coherent output | Cosine sim > 0.99 vs PyTorch |
| 2 | KV memory reduction | Measured at 32K context | 5-7x less than llama.cpp |
| 3 | OOM crossover | llama.cpp OOMs at N tokens, TurboQuant does not | Demonstrated on 8GB RAM |
| 4 | GitHub Release | v0.1.0 published with binaries | macOS ARM64 + Ubuntu x86-64 |
| 5 | Community reception | Reddit/HN post with benchmark data | Positive reception, not spam-filtered |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 3B model has unsupported ops (e.g., new attention variant) | Medium | High | Start with Qwen3.5-3B which shares architecture with existing 0.8B support |
| Dynamic allocation refactor breaks existing models | Medium | High | Run existing test suite after every refactor step; keep old paths as fallback |
| Quality degrades at long context with 3-bit KV | Low | High | Measure cosine similarity at each context length; fall back to 4-bit if needed |
| llama.cpp does not actually OOM at expected context length | Medium | Medium | Use `ulimit -v` to constrain memory; pick hardware/model combo where crossover is clear |
| Reddit post gets spam-filtered again | Medium | Low | Build karma first; post from established account; follow subreddit rules exactly |

---

## Execution Order

Phases are sequential with clear gates:

```
Phase A (Week 1-2)          Phase B (Week 3)           Phase C (Week 4)
─────────────────           ──────────────             ──────────────
Buffer refactor        -->  Benchmark script      -->  Tag v0.1.0
3B model support       -->  Memory measurement    -->  Build binaries
Verify output quality  -->  OOM crossover test    -->  Update README
                            Generate chart        -->  Community posts
```

**Gate A->B**: 3B model produces coherent output with verified quality.
**Gate B->C**: Benchmark chart shows clear memory advantage and OOM crossover.

---

## Appendix: KV Cache Memory Math

For a 3B model with typical config (32 layers, 32 heads, 128 head_dim):

```
KV cache per token = 2 * n_layers * n_heads * head_dim * dtype_size
                   = 2 * 32 * 32 * 128 * 2 bytes (FP16)
                   = 524,288 bytes per token

At 32K context:
  FP16 KV:     524,288 * 32,768 = 16 GB  (exceeds 8GB RAM)
  3-bit TQ KV: 524,288 * 32,768 / 5.3 = ~3 GB  (fits in 8GB RAM)
```

This is the crossover. At 32K context on 8GB RAM, llama.cpp cannot run. TurboQuant can.
