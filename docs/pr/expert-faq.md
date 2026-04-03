# Expert FAQ: Anticipated Hard Questions

Honest answers to the toughest questions expert developers and ML researchers will ask about quant.cpp.

---

## 1. "Your PPL improvement (-3.2%, -12.2%) seems suspicious. Quantization should add noise, not improve quality. Is this just noise in your measurement?"

Fair question. The -12.2% number on SmolLM2 1.7B at 999 tokens is likely within noise range for a single short sequence -- we do not claim statistical significance on that specific number. The mechanism we believe explains it: delta compression forces a low-pass filter on key representations, which may act as implicit regularization on this particular model/sequence. Cross-model results are more honest: Qwen3.5 0.8B shows +0.9%, Qwen3.5 4B shows +0.6%. The real claim is "near-lossless at 3-bit with delta," not "quantization magically improves quality." We should have hedged the README numbers harder.

## 2. "33K LOC sounds small but how much is actually useful vs boilerplate? What's the real core?"

The `src/` and `include/` directories total ~22.5K lines. The core forward pass (transformer, attention, KV cache, quantization) is roughly 6K lines across `tq_transformer.c`, `tq_uniform.c`, `tq_ops.c`, and the quantization kernels. Model loading (GGUF parsing, weight dequant) is another ~5K. The rest is backends (NEON, Metal, CUDA, Vulkan), MoE routing, DeltaNet recurrence, and test infrastructure. There is genuine boilerplate (FP16 converters are duplicated across files), but no generated code or padding. You can read the full attention path in one sitting.

## 3. "You say 'embeddable' but do you have a clean C API? What does integration actually look like?"

Yes. The public API is in `include/turboquant/turboquant.h` and `tq_engine.h`. The core integration surface is: `tq_load_model()` to load a GGUF file, `tq_create_state()` to allocate KV cache and activations, `tq_generate()` to run inference. All functions take and return C types, no C++ exceptions, no global state, no framework. The llama.cpp integration patch is 874 lines across 4 files. That said, the API is not yet stable -- struct layouts and function signatures may change before 1.0.

## 4. "Delta compression sounds like it would break parallel decoding. How does this work with batched inference?"

It does break parallel decoding in the general case. Delta compression is inherently sequential: each P-frame depends on the reconstruction of the previous key. During prefill, we could process all keys then compress sequentially, but we have not implemented that optimization. Currently quant.cpp is a single-sequence, autoregressive-only engine. No batched inference, no continuous batching, no parallel prefill. This is a real limitation -- delta KV is designed for memory-constrained edge deployment, not throughput-optimized serving.

## 5. "Why pure C instead of C++? Isn't this just NIH syndrome?"

The goal is embeddability. Pure C11 compiles on any platform with a C compiler -- iOS, Android NDK, WASM via Emscripten, microcontrollers, game engines. No RTTI, no exceptions, no STL allocator surprises. The tradeoff is real: we duplicate FP16 helpers, manually manage memory, and miss templates for generic quantization kernels. For a library meant to be dropped into other people's codebases, we think the portability wins. The test suite uses C++17 (Google Test) because that is a development dependency, not a deployment one.

## 6. "Your comparison with llama.cpp is unfair -- they use Q4_0 which is their worst KV quant. What about Q8_0 or K-quant KV?"

You are right, and we measured the full range. On SmolLM2 1.7B at 2K tokens: llama.cpp Q8_0 KV gives -0.4% PPL (essentially lossless), Q5_1 gives +0.9%, Q4_1 gives +3.2%, Q4_0 gives +10.6%. The README comparison against Q4_0 is the most favorable framing. At the same 4 bits/element, llama.cpp Q4_1 (+3.2%) is the fair comparison to our uniform_4b + Q4 V (-7.8% on a different baseline/sequence). We should present the full table. The honest differentiator is delta compression enabling quality-preserving 3-bit keys, not beating llama.cpp at 4-bit.

## 7. "What's the actual inference speed vs llama.cpp on the same model?"

On SmolLM2 1.7B (Apple M3, single sequence): quant.cpp gets ~80 tok/s with TQM Q4 format, llama.cpp gets ~35 tok/s with GGUF Q8 on Metal. But this is not apples-to-apples: different weight formats, different backends (our CPU vs their GPU), different quantization levels. We have not done a controlled comparison with identical weight quantization and identical hardware utilization. quant.cpp is not optimized for raw throughput -- it prioritizes memory efficiency and code simplicity. On larger models (4B+), llama.cpp with Metal will almost certainly be faster.

## 8. "Does this work with speculative decoding, continuous batching, or other advanced serving features?"

No. quant.cpp is a single-sequence autoregressive engine. No speculative decoding, no continuous batching, no beam search, no KV cache paging across requests. These are serving infrastructure features that require significant engineering beyond the core compression algorithm. The llama.cpp integration patch (874 lines) is the intended path for users who need those features -- use quant.cpp's compression inside a mature serving framework.

## 9. "The 'FP32 fallback bug' you disclosed -- how can we trust the current measurements aren't similarly flawed?"

We initially claimed 1-bit lossless KV compression, which turned out to be caused by the attention path silently reading from an unquantized FP32 key cache instead of the quantized one. After discovering it, we retracted all 1-bit claims, fixed the code path, and re-measured everything. The current results file (`bench/results/real_kv_compression.md`) explicitly states "no FP32 fallback" and shows that 1-bit actually does NOT work (cosine ~0.8, destroys attention). The fact that we publicly disclosed a bug that invalidated our best headline number is, we think, evidence of honest methodology. But you should reproduce independently -- the code and test data are all in the repo.

## 10. "Why should I use this over just waiting for the official TurboQuant implementation from Google?"

If you need a production-ready, GPU-optimized, batched serving solution -- wait. quant.cpp serves a different niche: developers who want to embed LLM inference in a C/C++ application with minimal dependencies and need to extend context length on memory-constrained hardware. The delta KV compression is our own engineering on top of the TurboQuant theory, not a port of any official implementation. If Google releases an official library with vLLM/TensorRT integration, it will likely be better for datacenter serving. quant.cpp is for the person shipping an on-device product who needs to read and control every line of the inference path.
