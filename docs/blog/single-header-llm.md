# LLM inference in a single C header file

*2026-04-04 -- quantumaikr/quant.cpp*

---

What if adding LLM inference to your C project was as easy as adding PNG loading? One header, one `#define`, and `cc app.c -o app -lm -lpthread`. No CMake. No package manager. No vendoring 200K lines of C++ templates. That is what [quant.h](https://github.com/quantumaikr/quant.cpp) gives you: a 15,404-line single-header file that loads GGUF models, runs transformer inference, and generates text. It supports Llama, Qwen3.5, and Gemma architectures out of the box.

The full project is 33K lines of C. The single header is the core 15K -- everything you need to go from a GGUF file on disk to tokens coming out.

## How stb-style headers work

If you have used [stb_image.h](https://github.com/nothings/stb) or [stb_truetype.h](https://github.com/nothings/stb), you know the pattern. The header file contains both declarations and implementations. In every file that needs the API, you `#include "quant.h"` and get the function prototypes. In exactly one `.c` file, you write:

```c
#define QUANT_IMPLEMENTATION
#include "quant.h"
```

That pulls in the actual code. The linker sees one copy of each function. You get the convenience of a header-only library with the compilation model of a normal C library. No build system integration required, no shared library versioning headaches, no pkg-config files to maintain.

## What is inside 15K lines

The header breaks down roughly as follows: GGUF model loader at 2,500 lines, matrix multiplication kernels at 1,800, the transformer forward pass at 2,300, tokenizer (BPE) at 1,200, KV cache with compression at 1,600, memory arena and allocation at 800, sampling and generation at 600, and the rest is dequantization routines, type definitions, and glue. Every major component lives in a single file, which means you can read the full inference pipeline top to bottom without jumping between translation units.

There is no abstraction for the sake of abstraction. The attention computation is a function that takes pointers and dimensions. The KV cache is a flat array with an integer head pointer. The model struct holds weight pointers and hyperparameters. If you have read Karpathy's llm.c, the level of directness is similar, though we support quantized weight formats and multiple architectures where llm.c targets a single model.

## The 6-function API

The entire public API is six functions:

```c
#include "quant.h"

int main(void) {
    quant_model *model = quant_load("smollm2-1.7b-q4_k_m.gguf");
    quant_ctx   *ctx   = quant_new(model, 2048);

    // One-shot question answering
    char *answer = quant_ask(ctx, "What is the capital of France?");
    printf("%s\n", answer);

    // Streaming generation with callback
    quant_generate(ctx, "The quick brown fox", 128,
                   (quant_params){.temperature = 0.7f});

    quant_free_ctx(ctx);
    quant_free_model(model);
    return 0;
}
```

Build it: `cc app.c -o app -lm -lpthread`. Run it. That is the entire integration story. No initialization rituals, no backend selection, no device management. The context object holds the KV cache and scratch buffers. You can create multiple contexts from one model for concurrent conversations.

## What we cut to make it fit

Fitting LLM inference into a single header means saying no to a lot of things. There is no GPU support -- no CUDA, no Metal, no Vulkan. The full quant.cpp project has Metal and CUDA backends, but they do not belong in a portable C header. There is no Mixture-of-Experts routing, which rules out Mixtral and similar architectures. There is no speculative decoding, no KV cache paging across multiple sequences, no tensor parallelism.

The quantization story is deliberately narrow. The header supports only uniform min-max quantization for runtime KV cache compression, plus the standard GGUF weight quantization formats (Q4_K_M, Q8_0, etc.) for loading models. The full project implements PolarQuant, QJL, and hybrid turbo schemes for research-grade KV compression. None of that is in the header. We picked the one method that is simple enough to be correct in 200 lines of C and good enough to matter in practice.

We also do not implement Flash Attention or any fused kernel tricks. The attention is a straightforward loop: compute QK^T, apply mask, softmax, multiply by V. It is not the fastest possible implementation, but it is the one you can read and debug without a PhD in GPU programming.

## Performance: honest numbers

On an Apple M3 MacBook Pro, SmolLM2 1.7B (Q4_K_M) runs at roughly 25 tokens per second for generation. That is about 3x slower than llama.cpp on the same hardware with the same model. The gap comes from SIMD -- llama.cpp has hand-tuned NEON and AVX2 kernels for every quantized matmul variant, while quant.h uses scalar C with compiler autovectorization. For a 1.7B model on a modern laptop, 25 tok/s is fast enough to read in real time.

Prompt processing (prefill) is slower proportionally, since it is entirely compute-bound on large matrix multiplications. If you are processing long documents, you will feel it. This header is for applications where you want a small model to answer a question, classify some text, or generate a short response -- not for running 70B models at production throughput.

We tested with SmolLM2 1.7B and the prompt "What is the capital of France?" The model produces coherent output: "Paris, a city rich in history..." Greedy decoding matches the expected output token-for-token.

## KV compression: 4x longer context for free

The header includes one feature that most single-file inference engines do not: KV cache compression. When enabled, key and value vectors are quantized to 4 bits as they enter the cache. This cuts KV memory by 4x, which means 4x longer context windows at the same memory budget.

The compression is effectively lossless. On WikiText-2, 4-bit uniform KV quantization adds +0.0% perplexity versus FP32 -- the difference is within measurement noise. This is not a novel result; uniform 4-bit works well because key and value distributions are smooth and roughly symmetric within each head. But it is a practical result: your 2048-token context can become 8192 tokens without allocating more memory and without measurable quality loss.

You enable it with a single flag in the context parameters. No separate compression pass, no offline calibration, no lookup tables to ship alongside the model.

## Try it

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp

# Download a small model
curl -LO https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf

# Build and run
echo '#define QUANT_IMPLEMENTATION
#include "quant.h"
int main(void) {
    quant_model *m = quant_load("smollm2-1.7b-instruct-q4_k_m.gguf");
    quant_ctx *c = quant_new(m, 2048);
    char *a = quant_ask(c, "Explain pointers in C in two sentences.");
    printf("%s\n", a);
    quant_free_ctx(c);
    quant_free_model(m);
}' > demo.c

cc demo.c -o demo -lm -lpthread
./demo
```

The project is MIT licensed. The header works on Linux, macOS, and Windows (MSVC and MinGW). We have tested it on x86_64 and ARM64. If it does not compile on your platform with your compiler, that is a bug -- file an issue.

---

*[quant.cpp](https://github.com/quantumaikr/quant.cpp) -- Embeddable LLM inference in pure C. 33K LOC, zero dependencies.*
