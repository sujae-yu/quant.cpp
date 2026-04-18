# Qwen3.6-35B-A3B prefill bottleneck analysis — 2026-04-18 late

## Baseline (commit `b7c42dd` head)

On Qwen3.6-35B-A3B-UD-Q3_K_S, M1 Pro 16 GB, CPU 8t, `TQ_NO_METAL=1 TQ_NO_MLOCK=1`:

| Metric | quant.cpp | llama.cpp CPU | Gap |
|---|:-:|:-:|:-:|
| Decode (warm) | ~12 t/s | 4.3 t/s | **2.8× faster** ✓ |
| Prefill (pp500) | ~5 t/s | 32.8 t/s | **6.5× slower** ✗ |

## Profile breakdown (10-second sample during prefill)

### Compute (total = 24,388 samples)
| Kernel | Samples | % | Target |
|---|---:|---:|---|
| `fused_dot_iq3_xxs_int8` | 8,690 | 35.6% | routed experts (IQ3_XXS) |
| `q8_int_dot_worker` | 6,558 | 26.9% | non-expert Q8_0 (attn/shared) |
| `fused_dot_iq3_s_int8` | 4,646 | 19.0% | routed experts (IQ3_S) |
| `fused_dot_q6_k_int8` | 2,880 | 11.8% | attn.wo + select experts (Q6_K) |
| `gguf_matmul_worker` | 617 | 2.5% | misc |
| `matmul_q4_rows` | 354 | 1.5% | shared experts Q4 |
| `fused_dot_iq4_xs_int8` | 217 | 0.9% | experts (IQ4_XS) |

### Idle (total = 36,866 samples)
| Op | Samples | % |
|---|---:|---:|
| `__psynch_cvwait` | 34,941 | 94.8% (worker idle) |
| `__psynch_mutexwait` | 1,312 | 3.6% |
| `__psynch_cvbroad/drop` | 613 | 1.7% |

### Misc (non-matmul, non-sync)
| Op | Samples |
|---|---:|
| `tq_dequant_row_gguf` | 430 |
| `deltanet_forward` | 398 |
| `_platform_memmove` | 350 |
| `memcpy` | 184 |
| `tq_matmul_gguf` | 177 |
| `tq_moe_route` | 133 |
| **`self_attn_forward`** | **63** |

## Key findings

### (1) Expert compute dominates prefill (62.3%)
`iq3_xxs_int8 + iq3_s_int8 + iq4_xs_int8` (expert kernels) = 13,553 samples = 55.6% of compute.
Adding shared expert Q6_K (attn.wo routes through Q6_K, partially) + Q4 rows: routed expert total ≈ 62%.

### (2) Self-attention is a negligible slice (<0.3%)
`self_attn_forward = 63 samples` (0.17% of total sampled time).

### (3) Worker idle (60% of total) is the real headroom
`cvwait = 34,941 samples / (24,388 compute + 36,866 idle) = 60%`.

This is the same phenomenon as decode — workers spend most time waiting between matmul dispatches. But in prefill, **per-token compute volume dominates**, so reducing `cvwait` cannot meaningfully shift wall-clock by itself — we need to *consolidate* the 240 matmul dispatches per token into fewer, larger batched dispatches.

## Mission A scope re-evaluation

### Step 2 (self-attn only batched): marginal
Self-attn = 63 of 24,388 compute samples = **0.26%** of prefill cost. Even a 10× speedup on this path moves prefill < 5% overall. Not a "breakthrough" return on a 400-LOC refactor.

### Step 3 (MoE expert batched grouping): real target
MoE expert compute = 62%. The architectural complexity:

- **Expert selection is per-token** — 8 of 256 routes chosen by `tq_moe_route` for each token independently.
- Across a 500-token batch, experts #42/#15/#87 may be used by 200 tokens each, while #201 is used by 3 tokens.
- **Fix**: after routing all 500 tokens, group tokens by their shared experts; dispatch one batched matmul per (expert × weight type × activation path) with its subset of tokens.
- This is a "token-grouped MoE" pattern (similar to what vLLM / DeepSpeed-MoE do for GPU batching, but on CPU).

Estimated complexity: 800-1200 LOC (expert-slot routing table + reshape scratch + reordered MoE kernel invocation + tokenwise scatter of results).

Estimated throughput gain: expert compute 62% → amortized over 500 tokens × expert overlap ratio ≈ 3-5× effective speedup on that 62% → overall prefill **5 t/s → ~12-18 t/s**.

### Total Mission A plan
| Step | Change | Predicted gain | Cost |
|---|---|:-:|:-:|
| 1 | `tq_batched_matmul_q8_0` kernel (✅ done `b7c42dd`) | infra | 165 LOC |
| 2 | `tq_forward_batch` accepts Qwen3.6 + self-attn batched | +5-10% | 400 LOC |
| 3 | **Token-grouped MoE expert dispatch** (the real win) | 5 → 12-18 t/s | 800-1200 LOC |

3-5 focused sessions. Each step independently landable and testable against per-token baseline.

## Immediate next-session target
Step 3 first, Step 2 after. Reason: Step 3 alone solves 60% of the gap; Step 2 polishes the remaining.

Recommended entry point: `src/engine/tq_moe.c` `tq_moe_forward` — add a `tq_moe_forward_batch(..., const int* tokens_per_expert_counts, ...)` that takes an N-batch instead of N=1, and dispatches one kernel call per expert with its subset of batch indices.
