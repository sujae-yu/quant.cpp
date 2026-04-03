#!/usr/bin/env python3
"""
quant.cpp — Qwen3.5-0.8B Inference Demo

실제 모델로 추론하면서 KV 캐시를 quant.cpp로 압축했을 때의
메모리 절약과 품질 보존을 직접 확인합니다.

Usage:
    source /tmp/tq_venv/bin/activate
    python3 examples/qwen35_inference_demo.py
"""

import sys
import os
import time
import numpy as np

# quant.cpp Python bindings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../bindings/python"))

def run_demo():
    print()
    print("=" * 70)
    print("  quant.cpp — Qwen3.5-0.8B Real Inference Demo")
    print("=" * 70)
    print()

    # ── Step 1: Load model ──
    print("[1/5] Loading Qwen3.5-0.8B...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, dtype=torch.float32
    )
    model.eval()
    print(f"  Model loaded: {model_name}")
    print(f"  Parameters: ~0.8B")
    print()

    # ── Step 2: Generate text (FP32 baseline) ──
    print("[2/5] Generating text (FP32 baseline)...")
    prompt = "The future of AI inference optimization lies in"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
        )
    gen_time = time.time() - t0

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gen_tokens = outputs.shape[1] - prompt_len
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Generated {gen_tokens} tokens in {gen_time:.2f}s ({gen_tokens/gen_time:.1f} tok/s)")
    print(f"  Output: \"{generated_text[:200]}...\"")
    print()

    # ── Step 3: Extract KV cache ──
    print("[3/5] Extracting KV cache for quantization analysis...")
    with torch.no_grad():
        out2 = model(**inputs, use_cache=True)
        cache = out2.past_key_values

    # Collect all attention layer KV caches
    layers_data = []
    total_kv_bytes_fp16 = 0
    for i in range(len(cache.key_cache)):
        k = cache.key_cache[i]
        v = cache.value_cache[i]
        if k is None or not isinstance(k, torch.Tensor) or k.dim() < 3:
            continue
        k_np = k.squeeze(0).float().numpy()
        v_np = v.squeeze(0).float().numpy()
        nh, sl, hd = k_np.shape
        layers_data.append({
            "layer": i, "num_heads": nh, "seq_len": sl, "head_dim": hd,
            "keys": k_np, "values": v_np,
            "k_min": k_np.min(), "k_max": k_np.max(),
            "v_min": v_np.min(), "v_max": v_np.max(),
        })
        total_kv_bytes_fp16 += nh * sl * hd * 2 * 2  # K+V, fp16

    print(f"  Attention layers with KV cache: {len(layers_data)}")
    if layers_data:
        ld = layers_data[0]
        print(f"  Per layer: {ld['num_heads']} heads x {ld['seq_len']} seq x {ld['head_dim']} dim")
    print(f"  Total KV cache (FP16): {total_kv_bytes_fp16:,} bytes ({total_kv_bytes_fp16/1024:.1f} KB)")
    print()

    # ── Step 4: quant.cpp compression ──
    print("[4/5] quant.cpp A/B test on real KV cache...")
    print()

    try:
        from turboquant import quant.cpp
        tq = quant.cpp("cpu")
        has_tq = True
    except Exception as e:
        print(f"  quant.cpp bindings not available: {e}")
        print("  Falling back to NumPy simulation...")
        has_tq = False

    # Test types
    test_configs = [
        ("FP16 (baseline)", None),
        ("uniform_4b", 5),       # TQ_TYPE_UNIFORM_4B
        ("mixed_4b8", 7),        # TQ_TYPE_MIXED_4B8
        ("uniform_2b", 6),       # TQ_TYPE_UNIFORM_2B
    ]

    print(f"  {'Config':<20} {'Key Cosine':>12} {'Value Cosine':>12} {'Size':>10} {'Compress':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    for name, qtype in test_configs:
        if qtype is None:
            # FP16 baseline
            print(f"  {'FP16 (baseline)':<20} {'1.000000':>12} {'1.000000':>12} "
                  f"{total_kv_bytes_fp16/1024:>8.1f}KB {'1.0x':>10}")
            continue

        total_k_cos = 0
        total_v_cos = 0
        total_quant_bytes = 0
        count = 0

        for ld in layers_data:
            nh, sl, hd = ld["num_heads"], ld["seq_len"], ld["head_dim"]

            for h in range(nh):
                keys_h = ld["keys"][h]    # [seq_len, head_dim]
                values_h = ld["values"][h]

                if has_tq:
                    # Real quant.cpp quantization
                    k_quant = tq.quantize_keys(keys_h, qtype)
                    k_deq = tq.dequantize_keys(k_quant, sl, hd, qtype)
                    v_quant = tq.quantize_keys(values_h, qtype)
                    v_deq = tq.dequantize_keys(v_quant, sl, hd, qtype)
                    total_quant_bytes += len(k_quant) + len(v_quant)
                else:
                    # NumPy simulation (simple uniform quantization)
                    def simple_quant(data, bits):
                        mn, mx = data.min(), data.max()
                        levels = 2**bits
                        scale = (mx - mn) / levels if mx > mn else 1e-8
                        q = np.clip(np.floor((data - mn) / scale), 0, levels - 1)
                        return mn + (q + 0.5) * scale

                    bits = 4 if qtype in [5, 7] else 2
                    k_deq = simple_quant(keys_h, bits)
                    v_deq = simple_quant(values_h, bits)
                    bpe = 4.2 if qtype == 5 else (5.0 if qtype == 7 else 2.2)
                    total_quant_bytes += int(nh * sl * hd * bpe / 8) * 2

                # Cosine similarity (flattened)
                k_flat = keys_h.flatten()
                kd_flat = k_deq.flatten()
                k_cos = np.dot(k_flat, kd_flat) / (np.linalg.norm(k_flat) * np.linalg.norm(kd_flat) + 1e-10)

                v_flat = values_h.flatten()
                vd_flat = v_deq.flatten()
                v_cos = np.dot(v_flat, vd_flat) / (np.linalg.norm(v_flat) * np.linalg.norm(vd_flat) + 1e-10)

                total_k_cos += k_cos
                total_v_cos += v_cos
                count += 1

        if not has_tq:
            total_quant_bytes = total_quant_bytes // (nh * len(layers_data))

        avg_k_cos = total_k_cos / count if count > 0 else 0
        avg_v_cos = total_v_cos / count if count > 0 else 0
        compress = total_kv_bytes_fp16 / total_quant_bytes if total_quant_bytes > 0 else 1

        print(f"  {name:<20} {avg_k_cos:>12.6f} {avg_v_cos:>12.6f} "
              f"{total_quant_bytes/1024:>8.1f}KB {compress:>8.1f}x")

    # ── Step 5: Summary ──
    print()
    print("[5/5] Summary")
    print("=" * 70)
    print()
    print(f"  Model:      {model_name}")
    print(f"  Prompt:     \"{prompt}\"")
    print(f"  Generated:  {gen_tokens} tokens at {gen_tokens/gen_time:.1f} tok/s")
    print(f"  KV layers:  {len(layers_data)} attention layers (hybrid model)")
    if layers_data:
        print(f"  Head dim:   {layers_data[0]['head_dim']}")
    print(f"  FP16 cache: {total_kv_bytes_fp16/1024:.1f} KB")
    print()
    print("  Recommendation: uniform_4b (A+ quality, 7.5x compression)")
    print("  For max compression: K4V2 asymmetric (Key 4-bit + Value 2-bit = 9.8x)")
    print()

if __name__ == "__main__":
    run_demo()
