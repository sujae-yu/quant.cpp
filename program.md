# quant.cpp — Agent Program Specification

## Objective

Build a production-grade C/C++ library for KV cache quantization that achieves:
- **5x memory reduction** at 3-bit quantization
- **< 1% attention score error** vs FP16 baseline
- **Cross-platform**: CPU (AVX2 + NEON) + CUDA + Metal
- **Zero external dependencies** in core library

## How You Work

1. Read `docs/wbs_v0.1.md` to understand the full task list
2. Identify the **next unchecked item** in milestone order (M0 → M1 → M2 → ...)
3. Implement it
4. Run `./score.sh` to measure your progress
5. If score improved or stayed same: commit and continue
6. If score dropped: revert and try a different approach
7. Mark completed items in `docs/wbs_v0.1.md` with `[x]`
8. Repeat from step 2

## Constraints

- **DO NOT modify** anything in `refs/` — these are read-only references
- **DO NOT modify** `program.md` or `score.sh`
- **DO modify** `src/`, `include/`, `tests/`, `bench/`, `spec/`, `integrations/`, `CMakeLists.txt`
- **DO update** `docs/wbs_v0.1.md` checkboxes as you complete items
- Every change must pass `./score.sh` without decreasing the score
- Prefer small, incremental changes over large rewrites
- Commit after each completed WBS item with a descriptive message

## Success Metric

`./score.sh` outputs a single score from 0.0 to 1.0:

```
Score breakdown:
  build_success:     0/1    — Does it compile?
  test_pass_rate:    0/1    — What fraction of tests pass?
  type_coverage:     0/1    — How many quantization types implemented?
  bench_exists:      0/1    — Do benchmarks exist and run?
  api_coverage:      0/1    — How many public API functions implemented?
  cross_platform:    0/1    — Does it build on multiple platforms?
  wbs_completion:    0/1    — What fraction of WBS items are checked?

Total: 0.00 / 1.00
```

Your goal: **maximize this score toward 1.0**.

## Algorithm References

When implementing algorithms, read the reference code:

### PolarQuant (tq_polar.c)
- Read `refs/PolarQuant/models/modeling_llama_polar.py` lines 135-157
- Read `refs/PolarQuant/models/kernel4group.py` lines 14-81
- Key: atan2 → angle, norm → radius, group min-max → quantize, pack rho<<tbits|theta

### QJL (tq_qjl.c)
- Read `refs/QJL/models/llama2_utils_qjl.py` lines 7-185
- Read `refs/QJL/qjl_kernel/csrc/qjl_quant_kernel.cu` lines 54-168
- Key: random projection → sign quantization → 8-bit pack, outlier detection via L2 norm top-k

### Block Structures (tq_types.h)
- Read `refs/llama.cpp/ggml/src/ggml-common.h` lines 86-347 for block patterns
- Read `refs/onnx/docs/docsgen/source/technical/int2.md` for bit-packing spec
- Key: self-contained blocks with embedded scale, static_assert size checks

### Type Traits (tq_traits.c)
- Read `refs/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` lines 207-396
- Key: function pointer table indexed by type enum, O(1) dispatch

### CUDA Kernels (src/backend/cuda/)
- Read `refs/QJL/qjl_kernel/csrc/` for QJL kernel patterns
- Read `refs/vllm/csrc/cache_kernels.cu` for fused cache kernel patterns
- Key: warp-level reduction, shared memory, template<scalar_t, cache_t>

### Metal Shaders (src/backend/metal/)
- Read `refs/llama.cpp/ggml/src/ggml-metal/` for Metal shader patterns
- Key: threadgroup memory, SIMD-group reduction

## Current State

Check `./score.sh` output to see where you are.
Check `docs/wbs_v0.1.md` to see what's done and what's next.
