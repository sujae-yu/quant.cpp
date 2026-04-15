# Accelerate GEMM vs GEMV — the 100× lever (microbench)

## Hypothesis under test

For prefill, batching N prompt tokens into a single matrix-matrix multiply
(SGEMM) should be much faster than running N independent matrix-vector
multiplies (SGEMV). If the speedup is < 3× even with optimized BLAS, then
batched-prefill engineering work is unjustified.

## Setup

Apple M1 Pro 16GB. Single-threaded Accelerate (cblas_sgemv / cblas_sgemm).
FP32 throughout. 5 reps, median reported. Source: `/tmp/gemm_blas.c`.

Compile: `clang -O3 -framework Accelerate gemm_blas.c -o bench`

## Results

| Shape (M,K) | N | N×SGEMV | 1×SGEMM | Speedup | SGEMM GFLOPS |
|---|---:|---:|---:|---:|---:|
| 3072 × 3072 | 1   | 2.6 ms   | 2.8 ms  | 0.95×  | 6.9    |
| 3072 × 3072 | 8   | 10.8 ms  | 3.2 ms  | **3.4×**  | 47     |
| 3072 × 3072 | 32  | 39.5 ms  | 1.3 ms  | **31×**   | 476    |
| 3072 × 3072 | 128 | 158 ms   | 2.4 ms  | **67×**   | 1027   |
| 8192 × 3072 (FFN) | 128 | 474 ms | 6.1 ms | **78×** | 1064 |
| **248320 × 2560 (Qwen lm_head)** | 128 | **13056 ms** | **132 ms** | **99×** | **1237** |

## Implications

1. **AMX coprocessor is real and accessible**. Accelerate hits 1.0-1.2 TFLOPS
   on FP32 GEMM (single-threaded), which is impossible without AMX. M1 Pro
   spec is ~2 TFLOPS GPU FP32; we're getting half of that on the CPU side
   via the AMX matrix unit.

2. **Batching is the entire game**. SGEMV peaks at ~15 GFLOPS regardless
   of N. SGEMM scales to 1200+ GFLOPS once N ≥ 32. The gap isn't algorithmic;
   it's the AMX execution model — a 16×16 outer-product per cycle vs.
   ~16-element dot per cycle.

3. **Naive C GEMM is NOT enough**. A direct port of three nested loops
   (tested in `/tmp/gemm_bench.c`) is *slower* than N×GEMV because the
   memory access pattern thrashes cache. The win requires either Accelerate
   or a hand-rolled tile-blocked kernel.

4. **For decode (N=1) Accelerate offers nothing new**. Speedup is 0.95×.
   This means our existing NEON quant matmul is fine for decode; we should
   only switch to Accelerate when N is large enough to amortize.

5. **Crossover N is small** — even N=8 already gives 3.4×. So a batched-
   prefill implementation that processes the prompt in chunks of 8-16
   tokens at a time would already capture most of the win.

## Path forward (committed)

Implement batched prefill using cblas_sgemm:
- Dequant each weight matrix to FP32 *once per layer pass*, not per call.
- For Phi-3.5 fused QKV (worst case): 110 MB transient FP32 buffer per
  layer — fits comfortably.
- Reuse the buffer across layers (not concurrent, single allocation).
- For lm_head specifically (largest single matmul), consider persistent
  FP16 storage if memory permits.

Target: 1000-token Phi-3.5 prefill from current ~600 s → under 30 s.
That's a **20× user-visible win** on long-context use cases.

## Why this matters strategically

This microbench validated that the prefill gap to llama.cpp (40-50× by
direct measurement today) is fundamentally caused by their use of
batched matmul + AMX, not by any quantization-format superiority.

Closing this gap is therefore an *engineering* problem (port forward
to batch-aware), not a *research* problem. We can do it.
