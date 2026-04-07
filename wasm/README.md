# @quantcpp/wasm

> Single-header C LLM inference engine compiled to WebAssembly. **192 KB** binary. Runs GGUF models in your browser with KV cache compression.

[![npm version](https://img.shields.io/npm/v/@quantcpp/wasm.svg)](https://www.npmjs.com/package/@quantcpp/wasm)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Install

```bash
npm install @quantcpp/wasm
```

## Quick start

```html
<script type="module">
  import { Quant } from '@quantcpp/wasm';

  const q = await Quant.create({
    scriptUrl: 'node_modules/@quantcpp/wasm/quant.js',
    modelUrl: 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf',
    onStatus: (msg) => console.log('[quant]', msg),
  });

  await q.generate('The capital of France is', {
    maxTokens: 32,
    temperature: 0.0,
    onToken: (text) => document.body.append(text),
    onDone: ({ nTokens, elapsedMs }) => {
      console.log(`Generated ${nTokens} tokens in ${elapsedMs.toFixed(0)} ms`);
    },
  });

  q.free();
</script>
```

## Why?

- **192 KB binary.** The entire inference engine — tokenizer, transformer forward pass, KV cache compression — fits in less than most JPEGs.
- **Zero server.** Models load and run entirely client-side. Nothing is uploaded.
- **Real models.** Llama 3, Qwen 3.5, Gemma 3, SmolLM2, and any other GGUF model under your memory budget.
- **KV compression built in.** Run 4–7× longer context than FP16 KV cache.
- **One file at the source.** Powered by [`quant.h`](https://github.com/quantumaikr/quant.cpp), a 628 KB single-header C library you can drop into any project.

## API

See [`index.d.ts`](./index.d.ts) for the full TypeScript surface.

```ts
import { Quant } from '@quantcpp/wasm';

const q = await Quant.create({
  scriptUrl: './quant.js',           // path to the loaded WASM glue
  modelUrl: '/models/llama.gguf',    // optional eager model load
  kvType: 'uniform_4b',              // KV cache quantization
  vQuant: 'q4',                      // value cache quantization
});

await q.generate('Hello', {
  maxTokens: 64,
  temperature: 0.7,
  onToken: (text) => process.stdout.write(text),
});

q.free();
```

### Supported KV quantization types

| Type | Bits/elem | Notes |
|---|---|---|
| `fp32` | 32 | baseline |
| `uniform_4b` ⭐ | 4 | recommended; +6.3% PPL on Llama 3.2 3B |
| `uniform_2b` | 2 | maximum compression, lower quality |
| `polar_3b` / `polar_4b` | 3 / 4 | PolarQuant-style |
| `qjl_1b` | 1 | sign-hash baseline |
| `turbo_kv_3b` / `turbo_kv_4b` | 3 / 4 | TurboQuant-structure (research; see [issue #14](https://github.com/quantumaikr/quant.cpp/issues/14)) |

## Build from source

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp/wasm
bash build.sh   # requires emscripten (brew install emscripten)
```

Output: `quant.wasm` (192 KB) and `quant.js` (~30 KB glue).

## License

Apache 2.0. See [LICENSE](../LICENSE).

## Citation

If you use quant.cpp's KV compression building blocks in research, please cite the underlying papers:

- [TurboQuant — Zandieh et al., ICLR 2026](https://arxiv.org/abs/2504.19874)
- [PolarQuant — AISTATS 2026](https://arxiv.org/abs/2502.02617)
- [QJL — AAAI 2025](https://arxiv.org/abs/2406.03482)
