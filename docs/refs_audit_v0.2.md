# quant.cpp — refs/ Absorption Audit v0.2

**Date**: 2026-03-29

## QJL (Quantized Johnson-Lindenstrauss)

- [x] Random projection matrix generation — Deterministic Rademacher (functionally equivalent to Gaussian)
- [x] Sign quantization — 8 bits/byte, LSB-first, ONNX compliant
- [x] Outlier detection — L2 norm top-k, stores indices + norm
- [~] Attention score — **v0.2 target**: Direct Hamming `sqrt(π/2)/d × norm × (d - 2×hamming)`
- [~] Outlier contribution — Block stores outlier_norm; **v0.2 target**: fused in attention

## PolarQuant

- [x] 2D pair splitting — D/2 pairs correctly handled
- [x] atan2 angle + norm radius — Proper [-π,π] normalization
- [x] Group min-max quantization — Per-block FP16 scales
- [~] Direct attention — **v0.2 target**: cos/sin lookup table → gather by index

## llama.cpp (Block & Dispatch Patterns)

- [x] Self-contained blocks with embedded scales — All 5 block types
- [x] static_assert size verification — 6 assertions across tq_types.h
- [x] Type traits O(1) dispatch — `TQ_TRAITS[type]` indexed table
- [x] vec_dot pairing — `residual_type` for quant.cpp composite
- [x] SIMD function pointer swapping — cpu_dispatch.c with CPUID detection

## vLLM (Cache Management)

- [x] Slot mapping (logical → physical) — Block index + offset calculation
- [x] Fused quantize+cache CUDA kernel — PolarQuant fused kernel
- [~] Template-based type selection — PolarQuant only; **v0.2**: add QJL/Turbo
- [~] Copy-on-Write — ref_count defined; **v0.2 target**: implement CoW logic
- [x] Block table mapping — Per-head block arrays

## ONNX (Bit Packing Standard)

- [x] LSB-first bit packing — All quantization types
- [x] 4-bit packing: `(high << 4) | (low & 0x0F)`
- [x] 2-bit packing: `(v3 << 6) | (v2 << 4) | (v1 << 2) | (v0 & 0x03)`
- [x] Format versioning — `TQ_SPEC_VERSION` in tq_spec.h

## Summary: 27/31 fully absorbed, 4 in-progress (v0.2 targets)
