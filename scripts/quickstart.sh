#!/bin/bash
# quant.cpp — One-command quickstart
# Downloads Qwen3.5-0.8B, builds the engine, converts the model, and runs inference.
#
# Usage:
#   bash scripts/quickstart.sh
#   bash scripts/quickstart.sh "Your prompt here"

set -e

PROMPT="${1:-What is deep learning?}"
THREADS="${2:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== quant.cpp Quickstart ==="
echo ""

# Step 1: Build
if [ ! -f build/quant ]; then
    echo "[1/4] Building..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=OFF -DTQ_BUILD_BENCH=OFF -Wno-dev 2>/dev/null
    cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" --target quant --target tq_convert 2>&1 | tail -3
    echo "      Done."
else
    echo "[1/4] Build found."
fi

# Step 2: Download model if not cached
MODEL_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B"
if [ ! -d "$MODEL_DIR" ]; then
    echo "[2/4] Downloading Qwen3.5-0.8B (~1.5 GB)..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir-use-symlinks True
    elif command -v python3 &>/dev/null; then
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B')
print('Download complete.')
" 2>/dev/null || {
            echo "      Installing huggingface_hub..."
            pip3 install --quiet huggingface_hub 2>/dev/null || pip3 install --quiet --break-system-packages huggingface_hub 2>/dev/null
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B')
print('Download complete.')
"
        }
    else
        echo "Error: python3 or huggingface-cli required for model download."
        echo "  Install: pip3 install huggingface_hub"
        echo "  Or download manually: https://huggingface.co/Qwen/Qwen3.5-0.8B"
        exit 1
    fi
    echo "      Done."
else
    echo "[2/4] Model found in cache."
fi

# Step 3: Convert to TQM
if [ ! -f model.tqm ]; then
    echo "[3/4] Converting to TQM format..."
    ./build/tq_convert -o model.tqm -j "$THREADS"
    echo "      Done."
else
    echo "[3/4] model.tqm found."
fi

# Step 4: Run inference
echo "[4/4] Running inference..."
echo ""
./build/quant model.tqm -p "$PROMPT" -j "$THREADS" -n 100
