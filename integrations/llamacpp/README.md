# TurboQuant -- llama.cpp Integration Guide

## Quick Start (3 steps)

### 1. Add TurboQuant to your llama.cpp build

```cmake
# In your llama.cpp CMakeLists.txt, add:
add_subdirectory(path/to/TurboQuant.cpp turboquant)
target_link_libraries(llama PRIVATE turboquant)
```

### 2. Register types at startup

```cpp
#include "integrations/llamacpp/tq_kv_cache.cpp"

int main() {
    tq_ggml_register_types();
    // ... rest of llama.cpp init
}
```

### 3. Use the --kv-cache-type flag

```bash
./llama-cli -m model.gguf --kv-cache-type tq-uniform-4b
```

## Available Types

| CLI Name | Compression | Quality | Recommended For |
|----------|-------------|---------|-----------------|
| `tq-uniform-4b` | 7.5x | A+ (0.995) | Default choice |
| `tq-uniform-2b` | 14.2x | B (0.897) | Max compression |
| `tq-polar-4b` | 7.1x | B (0.827) | Research |
| `tq-polar-3b` | 7.1x | B (0.827) | Research |
| `tq-turbo-3b` | 5.6x | B+ (0.917) | Balanced quality/compression |
| `tq-turbo-4b` | 5.6x | A (0.960) | High quality with compression |
| `tq-qjl-1b` | 25.6x | C (0.700) | Extreme compression |

All CLI names also accept short forms: `turbo3`, `polar4`, `uniform4`, `qjl1`, etc.

## How It Works

TurboQuant hooks into llama.cpp's KV cache layer:

```
Normal:  key_states (FP16) -> KV cache (FP16) -> attention
TurboQ:  key_states (FP16) -> quantize -> KV cache (3-4 bit) -> attention
```

The quantization is transparent to the rest of llama.cpp's pipeline. During
prefill and token generation, key/value vectors are quantized before being
stored in the KV cache. During attention computation, the quantized keys are
used directly via the `vec_dot` callback (dequantize + dot product), avoiding
full materialization of FP32 keys.

## Memory Impact

| Model | Context | FP16 Cache | TurboQuant (turbo3) | Saved |
|-------|---------|------------|---------------------|-------|
| Llama-3.2-3B | 64K | 7.00 GB | 1.25 GB | 82% |
| Llama-3-8B | 32K | 2.00 GB | 0.36 GB | 82% |
| Qwen2.5-0.5B | 128K | 10.50 GB | 1.88 GB | 82% |

## Build Instructions

### Option A: Add as subdirectory (recommended)

```cmake
# In llama.cpp's CMakeLists.txt:
add_subdirectory(path/to/TurboQuant.cpp turboquant)
target_link_libraries(llama PRIVATE turboquant)
```

### Option B: Link pre-built library

Build TurboQuant first:

```bash
cd /path/to/TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Then in llama.cpp's CMakeLists.txt:

```cmake
option(LLAMA_TURBOQUANT "Enable TurboQuant KV cache compression" OFF)

if(LLAMA_TURBOQUANT)
    set(TURBOQUANT_LIB_PATH "" CACHE PATH "Path to libturboquant.a")
    set(TURBOQUANT_INCLUDE_PATH "" CACHE PATH "Path to TurboQuant include/")

    add_library(turboquant STATIC IMPORTED)
    set_target_properties(turboquant PROPERTIES
        IMPORTED_LOCATION "${TURBOQUANT_LIB_PATH}/libturboquant.a"
    )

    target_sources(llama PRIVATE tq_kv_cache.cpp)
    target_include_directories(llama PRIVATE ${TURBOQUANT_INCLUDE_PATH})
    target_link_libraries(llama PRIVATE turboquant)
    target_compile_definitions(llama PRIVATE LLAMA_TURBOQUANT=1)
endif()
```

Build with:

```bash
cmake -B build \
  -DLLAMA_TURBOQUANT=ON \
  -DTURBOQUANT_LIB_PATH=/path/to/TurboQuant.cpp/build \
  -DTURBOQUANT_INCLUDE_PATH=/path/to/TurboQuant.cpp/include
cmake --build build -j$(nproc)
```

## Programmatic Integration

```cpp
#include "tq_ggml_type.h"

// Initialize TurboQuant types (call once at startup)
tq_ggml_register_types();

// Parse CLI argument
tq_type kv_type = tq_parse_kv_cache_type("turbo3");

// Create context
tq_context_t* ctx = tq_llamacpp_create_context();

// Quantize keys during prefill
size_t key_buf_size = tq_quantize_keys_size(n_tokens, head_dim, kv_type);
void* key_buf = malloc(key_buf_size);
tq_llamacpp_quantize_keys(ctx, key_data, n_tokens, head_dim,
                           kv_type, key_buf, key_buf_size);

// Compute attention during decode
float* scores = (float*)malloc(seq_len * sizeof(float));
tq_llamacpp_attention(ctx, query, key_buf, seq_len, head_dim,
                       kv_type, scores);

// Print memory savings
tq_llamacpp_print_config(kv_type, 4, n_heads, head_dim, max_seq_len);

// Cleanup
free(scores);
free(key_buf);
tq_free(ctx);
```

## Architecture

```
llama.cpp main loop
    |
    v
KV cache manager (modified)
    |
    +-- tq_ggml_register_types()      [startup: register GGML type IDs]
    +-- tq_parse_kv_cache_type()      [CLI: parse --kv-cache-type arg]
    +-- tq_llamacpp_quantize_keys()   [prefill/decode: compress keys]
    +-- tq_llamacpp_quantize_values() [prefill/decode: compress values]
    +-- tq_llamacpp_attention()       [decode: Q*K^T from quantized cache]
    |
    v
libturboquant.a
    |
    +-- CPU generic (reference C implementation)
    +-- CPU AVX2 (x86 optimized, auto-detected)
    +-- CPU NEON (ARM optimized, auto-detected)
    +-- CUDA (GPU, optional via TQ_BUILD_CUDA)
    +-- Metal (Apple GPU, optional via TQ_BUILD_METAL)
```

## Troubleshooting

**Q: Build fails with "turboquant.h not found"**
A: Ensure `TURBOQUANT_INCLUDE_PATH` points to the directory containing
`turboquant/turboquant.h` (i.e., the `include/` directory).

**Q: Linker errors about undefined tq_* symbols**
A: Make sure `libturboquant.a` is built and `TURBOQUANT_LIB_PATH` is correct.
Also ensure `-lm` is linked (TurboQuant uses libm for math functions).

**Q: Quality degradation with QJL 1-bit**
A: QJL 1-bit provides maximum compression but lower quality. Use `turbo3`
or `turbo4` for near-lossless quality with good compression.

**Q: How do I choose between turbo3 and turbo4?**
A: `turbo3` provides the best compression (5.6x) while maintaining good quality.
`turbo4` uses one more bit for higher quality at slightly less compression.
For most use cases, `turbo3` is the recommended default.

## License

Apache 2.0 -- same as TurboQuant.cpp
