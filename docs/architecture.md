# quant.cpp Architecture

## 4-Layer Stack

```
Layer 3: Integration    — llama.cpp, vLLM, ONNX Runtime plugins
Layer 2: Cache          — PagedQuantCache, Progressive Compression
Layer 1: Compute        — TypeTraits, BlockFormat, SIMD/GPU dispatch
Layer 0: Specification  — FormatSpec, OpSchema, TestVectors
```

## Design Principles

1. **Spec-First** (ONNX) — Format specification before implementation
2. **Zero-Overhead Dispatch** (llama.cpp) — Type traits table with function pointers
3. **Fused Kernels** (vLLM) — Minimize memory bandwidth via operation fusion
4. **Self-Contained Blocks** (llama.cpp) — Each block embeds its own metadata
5. **Progressive Compression** (novel) — Recent tokens at high precision, old tokens compressed

## Type System

The `tq_type_traits_t` table provides O(1) dispatch:

```c
const tq_type_traits_t TQ_TRAITS[TQ_TYPE_COUNT] = {
    [TQ_TYPE_POLAR_3B] = {
        .quantize  = tq_polar_quantize_ref,  // or _neon / _avx2
        .attention = tq_polar_attention_ref,
        ...
    },
};
```

At init time, SIMD-optimized function pointers replace the reference implementations.
