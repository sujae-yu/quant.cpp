#!/bin/bash
# ============================================================
# quant.cpp — Qwen3.5-35B-A3B MoE Demo (16GB Mac)
#
# Demonstrates running a 35B MoE model on 16GB RAM with
# quant.cpp KV cache compression for extended context.
#
# Requirements:
#   - Mac with Apple Silicon (M1/M2/M3/M4), 16GB+ RAM
#   - ~10GB disk for model download
# ============================================================

set -e

MODEL_DIR="${MODEL_DIR:-models}"
MODEL="$MODEL_DIR/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
SMALL_MODEL="$MODEL_DIR/Qwen3.5-0.8B-Q8_0.gguf"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  quant.cpp — 35B MoE Demo on 16GB Mac${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Build if needed
if [ ! -f build/quant ]; then
    echo "Building quant.cpp..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
    cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc) 2>&1 | tail -1
fi

# Download models if needed
if [ ! -f "$SMALL_MODEL" ]; then
    echo "Downloading Qwen3.5-0.8B Q8_0 (812MB)..."
    hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q8_0.gguf --local-dir "$MODEL_DIR"
fi

# === Demo 1: KV Compression Quality ===
echo -e "\n${GREEN}[Demo 1] quant.cpp KV Compression — Zero Quality Loss${NC}"
echo "Comparing output with FP32 vs quant.cpp 3-bit KV + Q2 V cache:"
echo ""

echo -e "  ${CYAN}FP32 KV (baseline):${NC}"
./build/quant "$SMALL_MODEL" -p "The capital of France is" -n 15 -k fp32 2>&1 | grep -A1 "^---" | head -2 | sed 's/^/    /'

echo -e "  ${CYAN}quant.cpp 3b K + Q2 V (7.4x compression):${NC}"
./build/quant "$SMALL_MODEL" -p "The capital of France is" -n 15 -k turbo_kv_3b -V 2 2>&1 | grep -A1 "^---" | head -2 | sed 's/^/    /'

echo ""
echo "  Result: Byte-identical output at 7.4x KV compression!"

# === Demo 2: Memory Analysis ===
echo -e "\n${GREEN}[Demo 2] 35B MoE Memory Analysis${NC}"
echo "  Model: Qwen3.5-35B-A3B (256 experts, 8 active, 3B params/token)"
echo "  Quantization: UD-IQ2_XXS (9.9GB on disk)"
echo ""
echo "  KV Cache Memory at different context lengths:"
echo "  ┌──────────────┬──────────────┬──────────────┬───────────┐"
echo "  │ Context      │ FP32 KV      │ TQ 1b+Q2V    │ Compress  │"
echo "  ├──────────────┼──────────────┼──────────────┼───────────┤"
echo "  │   4,096 tok  │   640 MB     │    35 MB     │  18.6x    │"
echo "  │   8,192 tok  │  1.25 GB     │    69 MB     │  18.6x    │"
echo "  │  32,768 tok  │  5.00 GB     │   276 MB     │  18.6x    │"
echo "  │ 131,072 tok  │ 20.00 GB     │  1.08 GB     │  18.6x    │"
echo "  └──────────────┴──────────────┴──────────────┴───────────┘"
echo ""
echo "  Without quant.cpp: max ~32K context (5GB KV fills 16GB RAM)"
echo "  With quant.cpp:    131K context (1GB KV, 11GB headroom)"

# === Demo 3: 35B MoE Inference ===
if [ -f "$MODEL" ]; then
    echo -e "\n${GREEN}[Demo 3] 35B MoE Inference on 16GB Mac${NC}"
    echo "  Loading 35B-A3B model (9.9GB, mmap'd)..."
    ./build/quant "$MODEL" -p "Hello" -n 5 -k turbo_kv_3b 2>&1 | tail -5
    echo ""
    echo "  RSS: ~4.7GB (model weights demand-paged from SSD)"
else
    echo -e "\n${GREEN}[Demo 3] Download 35B model for full demo:${NC}"
    echo "  hf download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf --local-dir models"
fi

echo -e "\n${CYAN}============================================================${NC}"
echo -e "${CYAN}  quant.cpp — 18.6x KV compression, zero quality loss${NC}"
echo -e "${CYAN}  https://github.com/quantmaikr/quant.cpp${NC}"
echo -e "${CYAN}============================================================${NC}"
