# Tier Benchmark — 2026-04-25

Standardized coherent-length measurement across 5 models, 3 prompts each. Run via [`tools/coh_bench.sh`](../tools/coh_bench.sh), `-T 0`, `-n 300`, single-thread deterministic.

## Prompts

1. **quantum**: "Explain quantum mechanics in simple terms with examples."
2. **poem**: "Write a short poem about a solitary lighthouse at dawn."
3. **trivia**: "What is the capital of France, and name two famous landmarks there?"

## Raw results — 14 models across family

| Model                           | quantum            | poem              | trivia             | Tier |
|:--------------------------------|:------------------:|:-----------------:|:------------------:|:----:|
| SmolLM2-135M Q8_0               | 299 (-n cap)       | 108 (EOS)         | 22 (EOS)           | 1    |
| SmolLM2-360M Q8_0               | 299 (-n cap)       | 108 (EOS)         | 22 (EOS)           | 1    |
| **Qwen2.5-0.5B Q4_K_M**         | **64, rep loop**   | **49, rep loop**  | **55, rep loop**   | **3** |
| Qwen3-0.6B Q4_K_M               | 299 (-n cap)       | 285 (EOS)         | 299 (-n cap)       | 1    |
| Qwen3.5-4B Q4_K_M               | 147 (EOS)          | 106 (EOS)         | 66 (EOS)           | 1    |
| llama-3.2-1B Q4_K_M             | 299 (-n cap)       | 133 (EOS)         | 110 (EOS)          | 1    |
| Llama-3.2-1B Q8_0               | 261 (EOS)          | 107 (EOS)         | 137 (EOS)          | 1    |
| Llama-3.2-3B Q8_0               | 299 (-n cap)       | 105 (EOS)         | 120 (EOS)          | 1    |
| Gemma-4-e2b-it Q8_0             | 299 (-n cap)       | 92 (EOS)          | 31 (EOS)           | 1    |
| Gemma-4-e4b-it Q4_0             | 299 (-n cap)       | 82 (EOS)          | 19 (EOS)           | 1    |
| Phi-3.5-mini-instruct Q4_K_M    | 299 (-n cap)       | 299 (-n cap)      | 299 (-n cap)       | 1    |
| Phi-3.5-mini-instruct Q8_0      | 299 (-n cap)       | 299 (-n cap)      | natural EOS        | 1    |
| **Qwen3.6-35B-A3B IQ4_XS**      | 149 (EOS, thinking) | **73, rep loop**  | **51, attractor**  | **2** |
| **Qwen3.6-35B-A3B Q5_K_M**      | 169 (EOS, thinking) | **68, rep loop**  | **69, rep loop**   | **2** |

**Key observations:**

- Qwen2.5-0.5B (Tier 3): model is genuinely too small — attractor under any engine. Keep in table as lower-bound reference.
- Qwen3.6-35B-A3B (both quants Tier 2): **quantization change doesn't help**. Both IQ4_XS and Q5_K_M show the same basin mismatch (attractor at ~70 tok on non-thinking prompts). Confirms basin issue is structural, not bit-width.
- 10/14 models are clean Tier 1 across 3 prompts.
- No model has been demoted from Tier 1 to 2 by our R63 cleanup. The DN_PORT fix + auto-preset cleanup was a net improvement.

## Quality verdicts (first ~200 chars)

### Qwen3-0.6B — Tier 1

quantum: *"Okay, the user wants me to explain quantum mechanics in simple terms with examples. Let me start by recalling what I know about quantum mechanics. First, I need to define it clearly. Quantum mechanics..."*

poem: *"Okay, the user wants me to write a short poem about a solitary lighthouse at dawn. Let me start by visualizing the scene. A lighthouse at dawn—so it's peaceful, maybe with some elements like sea..."*

trivia: *"Okay, so I need to figure out what the capital of France is and also name two famous landmarks there. Let me start by recalling some basic facts about France. First, the capital of France is Paris..."*

### Qwen3.5-4B — Tier 1

quantum: *"Imagine the world around you is a giant, chaotic puzzle where everything is constantly moving and changing. **Quantum mechanics** (often called 'quantum physics') is the set of rules that describes..."*

poem: *"The morning breaths its first soft sigh, As mist clings heavy to the shore; A single star above the hilltop sky, Watches over a silent shore. The sea is still, a vast expanse of gray..."*

trivia: *"The capital of France is **Paris**. Two famous landmarks located in Paris include: 1. The **Eiffel Tower** (La Tour Eiffel), 2. The **Louvre Museum**..."*

### Llama-3.2-3B — Tier 1

quantum: *"Quantum mechanics! It's a fascinating and mind-bending branch of physics that can be tricky to wrap your head around, but I'll try to explain it in simple terms with examples. **What is quantum mechanics?**..."*

poem: *"Amidst the misty morning's hush, A sentinel stands, steadfast and still. The solitary lighthouse, a beacon's rush, Cuts through the dawn's gray veil with gentle will..."*

trivia: *"The capital of France is Paris. Two famous landmarks in Paris are: 1. The Eiffel Tower (La Tour Eiffel) - a iconic iron lattice tower built for the 1889 World's Fair..."*

### Phi-3.5-mini — Tier 1

quantum: *"Quantum Mechanics is a branch of physics that deals with the behavior and interactions of particles at very small scales, such as atoms or subatomic particles..."*

poem: *"In the hush of early morn, A sentinel stands alone. Its beacon pierces through night's embrace, a steadfast guide for the lost at sea. The lighthouse watches over the waves with silent vigil..."*

trivia: *"The capital city of France is Paris. Two notable landmarks in this vibrant metropolis are the Eiffel Tower and Notre-Dame Cathedral, both iconic symbols of French culture..."*

### Qwen3.6-35B-A3B — Tier 2

quantum (thinking mode): *"Here's a thinking process: 1. **Deconstruct the Request:** - 'Explain quantum mechanics simply.' -> 'Use analogies, avoid jargon where possible, keep it engaging...'"* → 149 tok, quality reasoning trace.

poem: *"Thinking Process: . ) )56789101012345678910. Here is a poem about the lighthouse: - A solitary figure stands against the dawn's first light, casting long shadows of three45678901234567..."* → **73 tok, repetition loop**. Number-walk attractor after ~15 coherent words.

trivia: *"Thinking Process: .1. The user is asking about the location ofof Paris, France's capital city3.'4.'56789012345678901234567890"* → **51 tok, attractor**. Answer never delivered.

## Tier classification (data-driven)

Rule: model is **Tier 1** if all 3 prompts complete without attractor (natural EOS or -n cap, no repetition loop on -T 0). Otherwise **Tier 2**.

| Model | Tier | Evidence |
|-------|:----:|----------|
| Qwen3-0.6B Q4_K_M | 1 | 3/3 prompts natural |
| Qwen3.5-4B Q4_K_M | 1 | 3/3 prompts natural |
| Llama-3.2-3B Q8_0 | 1 | 3/3 prompts natural |
| Phi-3.5-mini Q4_K_M | 1 | 3/3 prompts natural |
| Qwen3.6-35B-A3B IQ4_XS | 2 | 1/3 prompts natural (thinking only); 2/3 hit attractor < 75 tok |

## Reproducibility

```bash
# Requires build/quant and relevant models under models/
./tools/coh_bench.sh \
  models/Qwen3-0.6B-Q4_K_M.gguf \
  models/Qwen3.5-4B-Q4_K_M.gguf \
  models/Llama-3.2-3B-Instruct-Q8_0.gguf \
  models/Phi-3.5-mini-instruct-Q4_K_M.gguf \
  models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
```

## Pairs well with

- [`tools/basin_compat.sh`](../tools/basin_compat.sh) — numerical divergence diagnostic (per-layer rel_diff vs llama-debug)
- [`docs/engine_basin_tiers.md`](engine_basin_tiers.md) — theory of when numerical divergence translates to quality loss
- [`docs/blog/fp32-basin-theory.md`](blog/fp32-basin-theory.md) — public-facing writeup of the finding

As noted in the theory doc, **basin compat (numerical) does not always predict coh quality**. Qwen3.5-4B has Tier 3 basin_compat score (2.30 max rel_diff) yet delivers Tier 1 coherent quality. Qwen3.6-A3B has Tier 2 basin_compat (0.41 max rel_diff) AND Tier 2 quality. The two metrics should be read together.
