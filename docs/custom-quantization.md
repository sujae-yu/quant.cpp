# Add Your Own KV Quantization Type

This guide walks through adding a custom KV cache quantization type to quant.cpp. By the end, you will have a working 8-bit uniform quantizer registered in the type system with tests.

---

## Table of Contents

- [How KV Quantization Works](#how-kv-quantization-works)
- [The 3 Functions You Need](#the-3-functions-you-need)
- [Step 1: Define the Block Structure](#step-1-define-the-block-structure)
- [Step 2: Implement quantize, dequantize, attention](#step-2-implement-quantize-dequantize-attention)
- [Step 3: Register in the Traits Table](#step-3-register-in-the-traits-table)
- [Step 4: Write Tests](#step-4-write-tests)
- [Step 5: Verify with score.sh](#step-5-verify-with-scoresh)
- [Reference: Existing Types](#reference-existing-types)

---

## How KV Quantization Works

During LLM inference, the KV cache stores key and value vectors from every previous token. For long sequences, this cache dominates memory. quant.cpp compresses these vectors using block-based quantization.

The pipeline for each attention head:

```
1. quantize:   float keys [block_size] --> block_tq_xxx (compact bytes)
2. cache:      store quantized blocks in paged KV cache
3. attention:  query x quantized_keys --> attention scores [seq_len]
4. dequantize: (optional, for debugging) block_tq_xxx --> float [block_size]
```

Key design principles:

- **Block-based**: Data is processed in fixed-size blocks (typically 128 elements). Each block is self-contained with its own scale/offset metadata.
- **ONNX LSB-first bit-packing**: Multi-bit values are packed into bytes with the least significant bits first.
- **O(1) dispatch**: Function pointers in a global traits table enable type-agnostic code paths.
- **Fused attention**: Each type can compute `query * key` directly from the quantized representation, avoiding full dequantization during inference.

## The 3 Functions You Need

Every quantization type must implement exactly three functions matching these signatures:

```c
// Quantize: float array --> packed block
typedef void (*tq_quantize_fn)(const float* src, void* dst, int n);

// Dequantize: packed block --> float array
typedef void (*tq_dequantize_fn)(const void* src, float* dst, int n);

// Attention: compute query @ quantized_keys scores
typedef void (*tq_attention_fn)(const float* query, const void* kv_cache,
                                float* scores, int seq_len, int head_dim);
```

Parameters:
- `src`/`dst`: Input/output data. For quantize, `dst` points to your block struct. For dequantize, `src` points to it.
- `n`: Number of elements (at most `block_size`).
- `query`: The current query vector `[head_dim]`.
- `kv_cache`: Array of quantized blocks, one per cached position.
- `scores`: Output attention logits `[seq_len]` (one score per cached key).
- `seq_len`: Number of cached positions.
- `head_dim`: Dimension of each head.

## Step 1: Define the Block Structure

Add your block struct to `include/turboquant/tq_types.h`. The block must be self-contained: it stores all metadata (scale, zero point) alongside the quantized data.

Here is a complete example for 8-bit uniform quantization:

```c
/* In include/turboquant/tq_types.h, before the #endif */

/* Uniform 8-bit quantization block (high-quality baseline)
 * Simple min-max linear mapping to 256 levels.
 * 8.25 bits per element: (4 bytes meta + 128 bytes data) / 128 elements.
 */
typedef struct {
    uint16_t scale;           /* (max - min) / 256, stored as fp16 (2B)  */
    uint16_t zero_point;      /* minimum value, stored as fp16 (2B)      */
    uint8_t  qs[TQ_BK];      /* 8-bit: 1 byte per value, 256 levels     */
} block_tq_uniform_8b;        /* 132 bytes per 128 elements              */

/* Compile-time size check */
TQ_CHECK_SIZE(block_tq_uniform_8b, 4 + TQ_BK);
```

Important rules for block structs:

1. **Always add a `TQ_CHECK_SIZE` assertion.** The build will fail with a clear error if your struct has unexpected padding.
2. **Use `uint16_t` for FP16 scale/offset fields.** Convert at runtime with helper functions.
3. **Bit-packing uses LSB-first convention** (ONNX compatible). For sub-byte types, lower-indexed values go into lower bits.
4. **Block size is `TQ_BK` (128) for most types**, or `TQ_BK_QJL` (256) for QJL-family types.

## Step 2: Implement quantize, dequantize, attention

Create a new source file `src/core/tq_uniform_8b.c`. Here is the complete implementation:

```c
/* src/core/tq_uniform_8b.c -- Uniform 8-bit KV cache quantization */

#include "turboquant/turboquant.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ---- FP16 conversion helpers ---- */

static uint16_t u8_fp32_to_fp16(float v) {
    union { float f; uint32_t u; } bits;
    bits.f = v;
    uint32_t sign = (bits.u >> 16) & 0x8000;
    int32_t  exp  = ((bits.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits.u >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float u8_fp16_to_fp32(uint16_t h) {
    union { float f; uint32_t u; } bits;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) { bits.u = sign; return bits.f; }
    if (exp == 31) { bits.u = sign | 0x7F800000 | (mant << 13); return bits.f; }
    exp = exp - 15 + 127;
    bits.u = sign | (exp << 23) | (mant << 13);
    return bits.f;
}

/* ---- Quantize: float[n] --> block_tq_uniform_8b ---- */

void tq_uniform_8b_quantize_ref(const float* src, void* dst, int n) {
    block_tq_uniform_8b* block = (block_tq_uniform_8b*)dst;
    int count = (n > TQ_BK) ? TQ_BK : n;

    /* Find min/max */
    float mn = FLT_MAX, mx = -FLT_MAX;
    for (int i = 0; i < count; i++) {
        if (src[i] < mn) mn = src[i];
        if (src[i] > mx) mx = src[i];
    }

    float range = mx - mn;
    if (range < 1e-8f) range = 1e-8f;
    float scale = range / 256.0f;   /* 8-bit: 256 bins */

    block->scale      = u8_fp32_to_fp16(scale);
    block->zero_point = u8_fp16_to_fp16(mn);

    /* Quantize each element to [0, 255] */
    for (int i = 0; i < count; i++) {
        int q = (int)floorf((src[i] - mn) / scale);
        if (q < 0)   q = 0;
        if (q > 255) q = 255;
        block->qs[i] = (uint8_t)q;
    }
    /* Zero-fill remainder if n < TQ_BK */
    for (int i = count; i < TQ_BK; i++) {
        block->qs[i] = 0;
    }
}

/* ---- Dequantize: block_tq_uniform_8b --> float[n] ---- */

void tq_uniform_8b_dequantize_ref(const void* src, float* dst, int n) {
    const block_tq_uniform_8b* block = (const block_tq_uniform_8b*)src;
    int count = (n > TQ_BK) ? TQ_BK : n;

    float scale = u8_fp16_to_fp32(block->scale);
    float mn    = u8_fp16_to_fp32(block->zero_point);

    for (int i = 0; i < count; i++) {
        /* Reconstruct with mid-bin centering (+0.5) for lower MSE */
        dst[i] = mn + ((float)block->qs[i] + 0.5f) * scale;
    }
}

/* ---- Attention: query @ quantized keys --> scores ---- */

void tq_uniform_8b_attention_ref(const float* query, const void* kv_cache,
                                  float* scores, int seq_len, int head_dim) {
    const block_tq_uniform_8b* blocks = (const block_tq_uniform_8b*)kv_cache;
    int blocks_per_head = (head_dim + TQ_BK - 1) / TQ_BK;

    for (int t = 0; t < seq_len; t++) {
        float dot = 0.0f;
        for (int b = 0; b < blocks_per_head; b++) {
            const block_tq_uniform_8b* blk = &blocks[t * blocks_per_head + b];
            float scale = u8_fp16_to_fp32(blk->scale);
            float mn    = u8_fp16_to_fp32(blk->zero_point);
            int offset = b * TQ_BK;
            int count = head_dim - offset;
            if (count > TQ_BK) count = TQ_BK;

            for (int i = 0; i < count; i++) {
                float val = mn + ((float)blk->qs[i] + 0.5f) * scale;
                dot += query[offset + i] * val;
            }
        }
        scores[t] = dot;
    }
}
```

Implementation notes:

- **Mid-bin centering** (`+0.5`): Placing the reconstruction point at the center of each quantization bin reduces MSE by up to 25% compared to bin-edge reconstruction.
- **Attention function**: Dequantizes on the fly while computing the dot product, avoiding a separate buffer allocation.
- **Clamp to valid range**: Always clamp quantized indices to `[0, max_level]` to handle floating-point edge cases.

## Step 3: Register in the Traits Table

Two files need changes:

### 3a. Add the enum value to `tq_types.h`

In `include/turboquant/tq_types.h`, add to the `tq_type` enum:

```c
typedef enum {
    TQ_TYPE_POLAR_3B  = 0,
    TQ_TYPE_POLAR_4B  = 1,
    // ... existing types ...
    TQ_TYPE_UNIFORM_3B= 12,
    TQ_TYPE_UNIFORM_8B= 13,   /* <-- NEW: Min-Max uniform 8-bit */
    TQ_TYPE_COUNT     = 14    /* <-- UPDATE: increment by 1 */
} tq_type;
```

### 3b. Register in the traits table in `tq_traits.c`

In `src/core/tq_traits.c`, add forward declarations and a table entry:

```c
/* Add forward declarations at the top */
extern void tq_uniform_8b_quantize_ref(const float* src, void* dst, int n);
extern void tq_uniform_8b_dequantize_ref(const void* src, float* dst, int n);
extern void tq_uniform_8b_attention_ref(const float* query, const void* kv,
                                         float* scores, int seq_len, int head_dim);

/* Add entry in the TQ_TRAITS array */
tq_type_traits_t TQ_TRAITS[TQ_TYPE_COUNT] = {
    /* ... existing entries ... */

    [TQ_TYPE_UNIFORM_8B] = {
        .name       = "uniform_8b",
        .block_size = TQ_BK,
        .type_size  = sizeof(block_tq_uniform_8b),
        .bpe        = (float)sizeof(block_tq_uniform_8b) * 8.0f / TQ_BK,
        .quantize   = tq_uniform_8b_quantize_ref,
        .dequantize = tq_uniform_8b_dequantize_ref,
        .attention  = tq_uniform_8b_attention_ref,
        .residual_type = TQ_TYPE_COUNT,  /* no residual */
    },
};
```

The `tq_type_traits_t` fields explained:

| Field | Description |
|-------|-------------|
| `name` | String identifier (used by `tq_type_from_name` lookup) |
| `block_size` | Number of float elements per block (128 for most types) |
| `type_size` | `sizeof(your_block_struct)` in bytes |
| `bpe` | Bits per element, computed as `type_size * 8 / block_size` |
| `quantize` | Function pointer: `float* --> block` |
| `dequantize` | Function pointer: `block --> float*` |
| `attention` | Function pointer: fused attention kernel |
| `residual_type` | For composite types (e.g., Turbo = Polar + QJL). Set to `TQ_TYPE_COUNT` for standalone types. |

### 3c. Add to `tq_get_format_spec` in `tq_traits.c`

Add a case to the switch statement in `tq_get_format_spec`:

```c
case TQ_TYPE_UNIFORM_8B:
    spec.algorithm = TQ_ALG_UNIFORM; spec.key_bits = 8; break;
```

### 3d. Add source file to CMakeLists.txt

Since the CMakeLists.txt uses `file(GLOB TQ_CORE_SOURCES src/core/*.c)`, any `.c` file placed in `src/core/` is automatically included. No CMake changes are needed.

## Step 4: Write Tests

Create `tests/test_uniform_8b.cpp`:

```cpp
#include <gtest/gtest.h>
extern "C" {
#include "turboquant/turboquant.h"
void tq_uniform_8b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_8b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_8b_attention_ref(const float* query, const void* kv,
                                  float* scores, int seq_len, int head_dim);
}
#include <cmath>
#include <vector>

/* Test 1: Roundtrip quantize -> dequantize should have low MSE */
TEST(Uniform8B, RoundtripBasic) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++)
        input[i] = sinf(i * 0.1f);

    block_tq_uniform_8b block;
    tq_uniform_8b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_8b_dequantize_ref(&block, output.data(), TQ_BK);

    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    /* 8-bit has 256 levels; MSE should be very small */
    EXPECT_LT(mse, 0.0001);
}

/* Test 2: Extreme value range */
TEST(Uniform8B, ExtremeValues) {
    std::vector<float> input(TQ_BK);
    for (int i = 0; i < TQ_BK; i++)
        input[i] = (float)i / TQ_BK * 100.0f - 50.0f;

    block_tq_uniform_8b block;
    tq_uniform_8b_quantize_ref(input.data(), &block, TQ_BK);

    std::vector<float> output(TQ_BK);
    tq_uniform_8b_dequantize_ref(&block, output.data(), TQ_BK);

    double mse = 0;
    for (int i = 0; i < TQ_BK; i++) {
        double d = input[i] - output[i];
        mse += d * d;
    }
    mse /= TQ_BK;
    /* range=100, step=100/256~0.39, MSE ~ step^2/12 ~ 0.013 */
    EXPECT_LT(mse, 0.05);
}

/* Test 3: Block struct has expected size */
TEST(Uniform8B, BlockSize) {
    /* 4 bytes metadata (scale + zero_point) + 128 data bytes = 132 */
    EXPECT_EQ(sizeof(block_tq_uniform_8b), 4u + TQ_BK);
}

/* Test 4: Type traits are registered correctly */
TEST(Uniform8B, TypeTraits) {
    EXPECT_STREQ(tq_type_name(TQ_TYPE_UNIFORM_8B), "uniform_8b");
    EXPECT_EQ(tq_type_block_size(TQ_TYPE_UNIFORM_8B), (size_t)TQ_BK);
    EXPECT_EQ(tq_type_type_size(TQ_TYPE_UNIFORM_8B), sizeof(block_tq_uniform_8b));
    EXPECT_GT(tq_type_bpe(TQ_TYPE_UNIFORM_8B), 8.0f);  /* 8.25 bpe */
}

/* Test 5: Attention scores match FP32 reference */
TEST(Uniform8B, AttentionCosine) {
    const int head_dim = 128;
    const int seq_len  = 4;

    std::vector<float> query(head_dim);
    for (int i = 0; i < head_dim; i++)
        query[i] = sinf(i * 0.3f);

    /* Create keys and quantize them */
    std::vector<float> keys(seq_len * head_dim);
    for (int i = 0; i < seq_len * head_dim; i++)
        keys[i] = cosf(i * 0.07f);

    int blocks_per_head = (head_dim + TQ_BK - 1) / TQ_BK;
    std::vector<block_tq_uniform_8b> qkeys(seq_len * blocks_per_head);
    for (int t = 0; t < seq_len; t++) {
        for (int b = 0; b < blocks_per_head; b++) {
            int offset = t * head_dim + b * TQ_BK;
            int count = head_dim - b * TQ_BK;
            if (count > TQ_BK) count = TQ_BK;
            tq_uniform_8b_quantize_ref(&keys[offset], &qkeys[t * blocks_per_head + b], count);
        }
    }

    /* Compute quantized attention scores */
    std::vector<float> scores(seq_len);
    tq_uniform_8b_attention_ref(query.data(), qkeys.data(), scores.data(), seq_len, head_dim);

    /* Compute FP32 reference scores */
    std::vector<float> ref_scores(seq_len);
    for (int t = 0; t < seq_len; t++) {
        float dot = 0;
        for (int i = 0; i < head_dim; i++)
            dot += query[i] * keys[t * head_dim + i];
        ref_scores[t] = dot;
    }

    /* Compute cosine similarity between score vectors */
    double dot_ab = 0, dot_aa = 0, dot_bb = 0;
    for (int t = 0; t < seq_len; t++) {
        dot_ab += scores[t] * ref_scores[t];
        dot_aa += scores[t] * scores[t];
        dot_bb += ref_scores[t] * ref_scores[t];
    }
    double cosine = dot_ab / (sqrt(dot_aa) * sqrt(dot_bb));
    EXPECT_GT(cosine, 0.999);  /* 8-bit should be very close to FP32 */
}
```

Register the test in `CMakeLists.txt` (or it may be auto-discovered via GLOB). Check the test target section for the pattern used.

## Step 5: Verify with score.sh

Build and run the tests:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
ctest --test-dir build --output-on-failure -R Uniform8B
```

Run the full scoring harness:

```bash
bash score.sh --quick    # Build + correctness (fast)
bash score.sh            # Full 5-dimension evaluation
```

The scoring harness checks:
- **Structure**: Do sources and tests exist?
- **Correctness**: Does it build with zero warnings? Do all tests pass?
- **Quality**: Is roundtrip MSE below threshold? Is attention cosine above 0.99?

---

## Reference: Existing Types

Study these files for implementation patterns:

| Type Family | Source File | Complexity | Good Starting Point? |
|-------------|------------|-----------|---------------------|
| Uniform 4-bit | `src/core/tq_uniform.c` | Simple | Yes -- closest to the example above |
| Uniform 2-bit | `src/core/tq_uniform.c` | Medium | Yes -- shows sub-block scales |
| Uniform 3-bit | `src/core/tq_uniform.c` | Medium | Shows 3-bit packing (non-power-of-2) |
| Polar | `src/core/tq_polar.c` | Complex | Advanced -- polar coordinate encoding |
| QJL | `src/core/tq_qjl.c` | Complex | Advanced -- random projection hashing |
| Turbo | `src/core/tq_turbo.c` | Complex | Advanced -- composite (Polar + QJL residual) |
| TurboKV | `src/core/tq_turbo.c` | Complex | Advanced -- RHT + codebook + QJL |
| Mixed | `src/core/tq_uniform.c` | Medium | Shows outlier handling |

### File Checklist

When adding a new quantization type, you will touch these files:

| File | Change |
|------|--------|
| `include/turboquant/tq_types.h` | Add block struct, enum value, `TQ_CHECK_SIZE` |
| `src/core/tq_yourtype.c` | Implement quantize, dequantize, attention |
| `src/core/tq_traits.c` | Add forward declarations, traits table entry, format spec case |
| `tests/test_yourtype.cpp` | Roundtrip MSE, block size, traits, attention cosine tests |

No other files require changes. The CMake build system uses `file(GLOB)` to discover sources in `src/core/` and tests are pattern-matched similarly.
