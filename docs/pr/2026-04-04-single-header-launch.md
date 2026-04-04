# Single-Header Launch Posts (2026-04-04)

---

## Show HN

**Title:** Show HN: quant.h -- LLM inference in a single C header (15K LOC, zero deps)

**URL:** https://github.com/quantumaikr/quant.cpp

**Text:**

quant.h is an stb-style single-header C library for running LLMs. 15K lines, 628KB, no dependencies beyond libc and pthreads. You add LLM inference to a C project the same way you add stb_image -- one `#include` and a compiler invocation.

I built this because every LLM runtime I tried required pulling in a framework. If you just want to run a 1.7B model inside an existing C application, you shouldn't need cmake, ggml, or 250K lines of C++. With quant.h you write 6 lines of code and compile with `cc app.c -lm -lpthread`.

What works today: GGUF loading, SmolLM2 1.7B, Qwen3.5, Llama-architecture models, ~25 tok/s on M3, KV cache compression (4-bit lossless, 3-bit at +1.3% PPL). What doesn't: no GPU, no MoE, no batched inference. This is deliberately slower than llama.cpp. The point is simplicity and embeddability, not speed.

Blog post with implementation details: https://github.com/quantumaikr/quant.cpp/blob/main/docs/blog/single-header.md

---

## Reddit r/C_Programming

**Title:** quant.h -- stb-style single-header library for LLM inference (15K lines, cc app.c -lm -lpthread)

**Body:**

I wanted to add LLM text generation to a C project without pulling in a build system or framework. Ended up writing a single-header library for it.

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"

int main() {
    quant_model* m = quant_load("model.gguf");
    quant_ctx*   c = quant_new(m, NULL);
    quant_generate(c, "Hello!", print_token, NULL);
    quant_free_ctx(c);
    quant_free_model(m);
}
```

```
cc app.c -o app -lm -lpthread
```

15K lines, 628KB, C11. Loads GGUF models, runs transformer forward pass, does token sampling. Supports SmolLM2 1.7B and Llama-architecture models at ~25 tok/s on M3.

No GPU support. Significantly slower than llama.cpp. The tradeoff is that you get the entire inference engine in one file you can read and modify.

Source: https://github.com/quantumaikr/quant.cpp

---

## Reddit r/programming

**Title:** We fit an LLM inference engine into a single C header file (15K LOC)

**Body:**

We packaged an LLM inference engine as an stb-style single-header C library. You `#include "quant.h"`, define `QUANT_IMPLEMENTATION` in one translation unit, and compile with `cc app.c -lm -lpthread`. No cmake, no package manager, no framework.

Why this matters: LLMs are becoming a standard building block, but the current runtimes (llama.cpp, vLLM, etc.) are large projects designed to be standalone servers. If you're building a C/C++ application and want to add local text generation as a feature -- not as a separate process -- the integration cost is high. quant.h makes it a single file copy.

The tradeoff is performance. This does ~25 tok/s on Apple M3 for a 1.7B model. No GPU, no batched inference, no speculative decoding. It also includes KV cache compression (4-bit keys are lossless on WikiText-2) which helps fit longer contexts in RAM.

Source: https://github.com/quantumaikr/quant.cpp

---

## Reddit r/LocalLLaMA

**Title:** quant.cpp v0.2: now ships as a single-header C library -- add LLM inference to your C project with one file

**Body:**

quant.cpp now has a single-header distribution: `quant.h`, 15K lines, 628KB. You include it like stb_image and compile with `cc app.c -lm -lpthread`. No build system needed.

This is aimed at app developers who want to embed a small LLM (1-3B) inside an existing C/C++ project without depending on a full inference framework. Think CLI tools, game NPCs, embedded assistants -- cases where you want local generation as a library call, not a server.

KV cache compression is the other reason to look at this. On WikiText-2 with SmolLM2 1.7B: 4-bit keys give +0.0% PPL (lossless, 4x less KV memory), 3-bit delta keys give +1.3% PPL. On an 8GB laptop with Llama 8B Q4, this extends usable context from ~16K to ~61K tokens.

Honest limitations: CPU only, ~25 tok/s on M3, no MoE, delta mode drops to 7 tok/s. This is not a llama.cpp replacement -- it's a library for embedding inference into other software.

Source: https://github.com/quantumaikr/quant.cpp
