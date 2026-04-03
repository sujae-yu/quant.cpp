# quant.cpp Integration Guide

## llama.cpp Integration

1. Add quant.cpp as a subdirectory in your llama.cpp build
2. Include `integrations/llamacpp/tq_ggml_type.h`
3. Call `tq_ggml_register_types()` during init
4. Use `--kv-cache-type turbo3` CLI option

## vLLM Integration

1. Build Python bindings: `pip install ./bindings/python`
2. Import: `from turboquant import quant.cppContext`
3. Set `kv_cache_dtype="turbo3"` in engine config

## C API Quick Reference

```c
tq_context_t* ctx;
tq_init(&ctx, TQ_BACKEND_CPU);
tq_quantize_keys(ctx, keys, n, dim, TQ_TYPE_POLAR_4B, out, size);
tq_attention(ctx, query, kv, seq_len, dim, TQ_TYPE_POLAR_4B, scores);
tq_free(ctx);
```
