# TurboQuant.cpp — Work Breakdown Structure v1.1

**Version**: 1.1
**Date**: 2026-03-31
**Focus**: Larger Model Support, Long Context Benchmarks, Release & Community

---

## Phase A: Larger Model Support (Days 1-3)

### A1: Dynamic Buffer Allocation (~6h)

Current code has stack-allocated fixed-size arrays (`float gate_vals[128]`, `float sk[128]`) in `src/engine/tq_transformer.c`. These assume max dim=4096, but larger models (e.g., Gemma 3 4B) have intermediate_dim up to 10240 and head counts that may exceed 128.

- [ ] Audit all stack-allocated fixed-size buffers in `src/engine/tq_transformer.c` (~2h)
  - [ ] Identify every `float arr[N]` pattern with hardcoded sizes
  - [ ] Document each buffer's actual max dimension requirement per model config
  - [ ] List buffers that are safe (small, bounded) vs. those that must be dynamic
- [ ] Define runtime state struct in `include/turboquant/tq_engine.h` (~1h)
  - [ ] Add dynamic buffer pointers to `tq_state_t` (e.g., `float* xb`, `float* xb2`, `float* gate_vals`, `float* sk`)
  - [ ] Add buffer size fields so reallocation can be checked
  - [ ] Add `tq_state_init(model)` that allocates buffers based on model config
  - [ ] Add `tq_state_free(state)` to clean up
- [ ] Replace stack arrays with dynamic buffers in `src/engine/tq_transformer.c` (~2h)
  - [ ] Replace `float gate_vals[128]` with `state->gate_vals` (sized to `n_heads`)
  - [ ] Replace `float sk[128]` with `state->sk` (sized to `max(n_heads, head_dim)`)
  - [ ] Replace any `float xb_q8[4096]` or similar with `state->xb_q8` (sized to `hidden_dim` or `intermediate_dim`)
  - [ ] Ensure all replaced buffers use model config dimensions, not hardcoded constants
- [ ] Add compile-time and runtime guards (~1h)
  - [ ] Add `assert(state->buf_size >= required)` checks at entry points
  - [ ] Add `TQ_MAX_DIM` compile-time constant as a sanity upper bound (e.g., 32768)
  - [ ] Verify zero regressions: existing Gemma 270M / Qwen 0.8B tests still pass

### A2: Gemma 3 4B Support (~8h)

Target model: `google/gemma-3-4b-it` (or `unsloth/gemma-3-4b-it`). Architecture matches Gemma 270M but with larger dimensions: hidden_dim=2560, n_layers=34, n_heads=8, n_kv_heads=4, head_dim=256, intermediate_size=10240.

- [ ] Download and inspect model (~1h)
  - [ ] Download model weights (safetensors) and tokenizer
  - [ ] Inspect config.json: confirm hidden_dim, n_layers, n_heads, n_kv_heads, head_dim, intermediate_size
  - [ ] Verify weight tensor names match existing Gemma name mapping
- [ ] Convert model to TQM format (~2h)
  - [ ] Run `tq_convert` on Gemma 3 4B safetensors
  - [ ] Verify conversion completes without errors
  - [ ] Verify output file size is reasonable (expected ~2-3 GB for Q4 weights)
  - [ ] Spot-check weight statistics (mean, std) for sanity
- [ ] Validate forward pass (~3h)
  - [ ] Run `tq_run` with a simple prompt, verify no crashes or NaNs
  - [ ] Compare logits against PyTorch reference for first 10 tokens of a fixed prompt
  - [ ] Verify KV cache quantization works at head_dim=256
  - [ ] Verify attention scores are numerically stable at 34 layers
- [ ] Performance baseline (~2h)
  - [ ] Measure tok/s on Apple Silicon (M-series)
  - [ ] Measure peak memory usage
  - [ ] Measure KV cache memory at various context lengths (512, 2048, 8192)
  - [ ] Document results in `docs/benchmarks.md`

### A3: Llama 3.2 3B Architecture Support — Stretch Goal (~12h)

Llama 3.2 3B uses standard transformer with GQA, RoPE, SwiGLU, RMSNorm. Very similar to Gemma but no sliding window, no dual RoPE, no post-norms, different tokenizer.

- [ ] Architecture delta analysis (~2h)
  - [ ] Document Llama vs Gemma architectural differences
  - [ ] Identify code paths that need Llama-specific branches
  - [ ] Identify shared code paths (SwiGLU FFN, RMSNorm, GQA attention)
- [ ] Weight name mapping in `src/engine/tq_model.c` (~3h)
  - [ ] Add Llama weight name patterns (e.g., `model.layers.N.self_attn.q_proj.weight`)
  - [ ] Add auto-detection of model family from weight names or config
  - [ ] Handle Llama's different norm placement (pre-norm only, no post-norm)
  - [ ] Handle Llama's RoPE config (single frequency base, no sliding window)
- [ ] Tokenizer support (~4h)
  - [ ] Llama 3 uses SentencePiece BPE with byte-fallback — verify compatibility with existing `tq_tokenizer.c`
  - [ ] Handle Llama's special tokens (`<|begin_of_text|>`, `<|eot_id|>`, etc.)
  - [ ] Add chat template formatting for Llama 3 instruction format
- [ ] Validation (~3h)
  - [ ] Run `tq_convert` on Llama 3.2 3B
  - [ ] Run `tq_run` and verify coherent text generation
  - [ ] Compare logits against PyTorch reference
  - [ ] Verify tok/s and memory are competitive

---

## Phase B: Long Context Benchmark (Days 3-5)

### B1: KV Cache Memory Measurement (~4h)

- [ ] Add memory tracking infrastructure (~2h)
  - [ ] Add `tq_cache_memory_stats_t` struct: `total_bytes`, `key_bytes`, `value_bytes`, `n_tokens`, `n_layers`
  - [ ] Implement `tq_cache_get_stats(cache)` that computes actual memory usage
  - [ ] Track per-layer memory breakdown (useful for mixed-precision schemes)
- [ ] Integrate with CLI (~1h)
  - [ ] Add `--memory` flag to `tq_run` that prints KV cache stats after generation
  - [ ] Print format: `KV Cache: {total_mb} MB ({key_mb} MB keys + {value_mb} MB values) for {n_tokens} tokens`
  - [ ] Print per-layer breakdown when `--verbose` is also set
- [ ] Baseline measurements (~1h)
  - [ ] Measure FP16 KV memory (theoretical: `2 * n_layers * n_kv_heads * head_dim * seq_len * 2 bytes`)
  - [ ] Measure Q4 KV memory (TurboQuant uniform_4b)
  - [ ] Calculate and report compression ratio
  - [ ] Verify compression ratio matches expected ~3.5-4x for Q4

### B2: Long Context Test (~6h)

- [ ] Create test prompt generator (~1h)
  - [ ] Script to generate prompts of increasing length: 1K, 4K, 8K, 16K, 32K tokens
  - [ ] Use natural text (e.g., concatenated paragraphs) not random tokens
  - [ ] Include a "needle" question at the end to test retrieval quality
- [ ] TurboQuant long context runs (~2h)
  - [ ] Run `tq_run` at each context length, record: wall time, peak RSS, tok/s, KV cache MB
  - [ ] Verify no OOM or crash up to model's max context
  - [ ] Record quality metric: perplexity or needle retrieval accuracy
- [ ] llama.cpp baseline comparison (~2h)
  - [ ] Run equivalent llama.cpp binary on same model and prompts
  - [ ] Record same metrics: wall time, peak RSS, tok/s, KV cache MB
  - [ ] Use llama.cpp's default FP16 KV cache (no quantization)
  - [ ] Also test llama.cpp with Q4_0 KV cache for fair comparison
- [ ] Find OOM crossover point (~1h)
  - [ ] On a fixed memory budget (e.g., 8 GB), find max context for each engine
  - [ ] Document the crossover: "TurboQuant supports Nx longer context in same memory"
  - [ ] Record exact numbers for chart generation

### B3: Benchmark Script & Chart (~4h)

- [ ] Create `bench/long_context_bench.sh` (~2h)
  - [ ] Accept model path and engine paths as arguments
  - [ ] Run both engines at each context length (1K, 4K, 8K, 16K, 32K)
  - [ ] Output CSV: `context_length,tq_memory_mb,llamacpp_memory_mb,tq_tok_s,llamacpp_tok_s,tq_quality,llamacpp_quality`
  - [ ] Handle OOM gracefully (record "OOM" instead of crashing)
  - [ ] Add `--quick` mode that only tests 1K, 4K, 8K
- [ ] Create chart generation script (~1.5h)
  - [ ] `bench/plot_long_context.py` using matplotlib
  - [ ] Dual y-axis chart: memory (MB) on left, tok/s on right
  - [ ] X-axis: context length (log scale)
  - [ ] Two series per metric: TurboQuant (solid) vs llama.cpp (dashed)
  - [ ] Highlight OOM point for llama.cpp with annotation
  - [ ] Save as `docs/assets/long_context_chart.png`
- [ ] Generate and validate chart (~0.5h)
  - [ ] Run benchmark on target hardware
  - [ ] Generate chart, verify it looks correct
  - [ ] Verify numbers match raw CSV data

---

## Phase C: Release & Community (Days 5-7)

### C1: GitHub Release v0.1.0 (~3h)

- [ ] Pre-release checklist (~1h)
  - [ ] All tests pass (`ctest --test-dir build --output-on-failure`)
  - [ ] No compiler warnings (`-Wall -Wextra -Werror`)
  - [ ] `tq_run` works end-to-end on at least one model
  - [ ] README is accurate and up-to-date
  - [ ] LICENSE file exists
- [ ] Create release (~1h)
  - [ ] Create annotated git tag: `git tag -a v0.1.0 -m "TurboQuant.cpp v0.1.0"`
  - [ ] Write release notes (update `scripts/release_notes_v0.1.0.md` if needed)
  - [ ] Include: feature summary, supported models, build instructions, known limitations
  - [ ] Publish via `gh release create v0.1.0 --title "v0.1.0" --notes-file scripts/release_notes_v0.1.0.md`
- [ ] Release artifacts (~1h)
  - [ ] Attach pre-built binaries for macOS arm64 (if feasible)
  - [ ] Attach example model conversion script
  - [ ] Verify release page renders correctly on GitHub

### C2: README Update with Benchmark Results (~2h)

- [ ] Update performance section (~1h)
  - [ ] Add long context memory comparison chart (`docs/assets/long_context_chart.png`)
  - [ ] Add "Memory Savings" section with concrete numbers (e.g., "4x less KV cache memory")
  - [ ] Add OOM crossover highlight (e.g., "32K context in 4 GB where llama.cpp needs 16 GB")
- [ ] Add "Supported Models" section (~0.5h)
  - [ ] List Gemma 270M, Qwen 0.8B, Gemma 3 4B (and Llama 3.2 3B if completed)
  - [ ] Include model size, expected tok/s, memory usage for each
- [ ] Polish and review (~0.5h)
  - [ ] Proofread all new content
  - [ ] Verify all links work
  - [ ] Verify all images render correctly
  - [ ] Update badge if applicable (build status, version)

### C3: Community Launch (~3h)

- [ ] r/LocalLLaMA post (~1.5h)
  - [ ] Draft post: "TurboQuant.cpp — KV cache compression lets you run 32K context where llama.cpp OOMs"
  - [ ] Lead with the chart, not speed claims
  - [ ] Include: what it is, how it works (1 paragraph), the benchmark results, how to try it
  - [ ] Link to GitHub repo and release
  - [ ] Prepare for FAQ: "How is this different from llama.cpp Q4 KV?", "Does this work with GGUF?"
- [ ] Hacker News post (~1h)
  - [ ] Title: "Show HN: Pure C inference engine with 7.5x KV cache compression"
  - [ ] Write concise top-comment explaining the technical approach
  - [ ] Highlight: zero external dependencies, single-file C, quantized integer attention
- [ ] Follow-up engagement (~0.5h)
  - [ ] Monitor comments for first 24 hours
  - [ ] Respond to technical questions
  - [ ] File issues for any valid feature requests or bug reports

---

## Effort Summary

| Phase | Sub-task | Estimated Hours |
|-------|----------|-----------------|
| A1 | Dynamic Buffer Allocation | 6h |
| A2 | Gemma 3 4B Support | 8h |
| A3 | Llama 3.2 3B (stretch) | 12h |
| B1 | KV Cache Memory Measurement | 4h |
| B2 | Long Context Test | 6h |
| B3 | Benchmark Script & Chart | 4h |
| C1 | GitHub Release v0.1.0 | 3h |
| C2 | README Update | 2h |
| C3 | Community Launch | 3h |
| | **Total** | **48h** |
| | **Total (excluding stretch A3)** | **36h** |

---

## Dependencies

```
A1 (Dynamic Buffers) ──→ A2 (Gemma 4B) ──→ A3 (Llama 3B, stretch)
                     └──→ B1 (Memory Tracking) ──→ B2 (Long Context Test) ──→ B3 (Chart)
                                                                           └──→ C2 (README)
                                                                                C1 (Release) ──→ C3 (Community)
```

## Completion Criteria

- [ ] `tq_run --model gemma-3-4b-it.tqm --prompt "What is AI?"` generates coherent text
- [ ] KV cache memory usage is reported via `--memory` flag
- [ ] Long context benchmark CSV and chart exist at `bench/` and `docs/assets/`
- [ ] GitHub release v0.1.0 is published
- [ ] README includes benchmark chart and supported models list
