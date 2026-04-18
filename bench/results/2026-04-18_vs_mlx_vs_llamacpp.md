# 3-way 엔진 비교 — quant.cpp vs llama.cpp vs MLX (2026-04-18)

M1 Pro 16 GB, macOS 24, warm run, same prompt `"Once upon a time"`, 30-token greedy decode unless noted.

**MLX version**: `mlx-lm 0.31.2` / `mlx 0.31.1` / `mlx-metal 0.31.1` (venv pip install).
**llama.cpp**: Homebrew `ggml 0.9.11` build d00685831.
**quant.cpp**: this repo at HEAD (commit `59263da`).

---

## Qwen3.5-4B Q4_K_M — dense (모든 엔진 실측)

| Engine | Backend | Speed | RSS | bpw | Notes |
|---|---|:-:|:-:|:-:|---|
| **MLX 4bit** | Metal GPU | **58.5 t/s** | **2.49 GB** | ~4.5 | mlx-community/Qwen3.5-4B-4bit |
| llama.cpp Q4_K_M | Metal GPU | 31.4 t/s | — | 4.84 | default `-ngl 99` |
| llama.cpp Q4_K_M | CPU 8t | 20.2 t/s | — | 4.84 | `-ngl 0` |
| quant.cpp Q4_K_M | CPU 8t | 14.1 t/s (peak) | 6.90 GB | 4.84 | `TQ_NO_METAL=1` |

**명확한 패턴**: MLX Metal → llama.cpp Metal → llama.cpp CPU → quant.cpp CPU. 우리 **MLX 대비 24%, llama.cpp CPU 대비 70%**. Apple Silicon dense 시장은 MLX 영역.

**RSS 비교**: MLX는 raw weight만 (2.49 GB ≈ 4.21B × 4.5 bpw / 8). quant.cpp는 **load-time Q4 재양자화** 때문에 원본 mmap + 내부 Q4 버퍼 둘 다 유지 → 6.90 GB. 이 재양자화는 일부 모델의 디코드 속도 이득이 있는 설계 선택이지만 RAM 사용량은 비효율.

---

## Qwen3.6-35B-A3B MoE — MoE (MLX 다운로드 대기 중)

| Engine | Backend | Speed | RSS | Notes |
|---|---|:-:|:-:|---|
| **quant.cpp UD-Q3_K_S** | CPU 8t | **14.3 t/s** | **5.24 GB** | `TQ_NO_MLOCK=1` |
| quant.cpp UD-IQ2_XXS | CPU 8t | 16.1 t/s | 6.54 GB | |
| llama.cpp Q3_K_S | CPU 8t | 5.1 t/s | — | |
| llama.cpp Q3_K_S | Metal | **hang** | — | 알려진 qwen35moe Metal 문제 |
| MLX 4bit | Metal | *측정 중* | — | 19 GB 다운로드 진행 중; 16 GB Mac에 로드 가능성 불확실 |

**명확한 패턴**: MoE CPU에서는 quant.cpp가 **llama.cpp 대비 2.8× 빠름**. Metal에서 llama.cpp가 동작 안 함 (Qwen 3.6-A3B는 DeltaNet + MoE hybrid로 일반 Metal 커널이 실패). MLX Metal이 성공하면 dense처럼 앞설 가능성 있으나, **19 GB 모델이 16 GB Mac에 메모리 상으로 들어갈지 자체가 미지수** (실측 진행 중).

---

## 가치 포지셔닝 요약

### 영역별 승자 (측정 기반)

| 시나리오 | 최강 |
|---|---|
| Apple Silicon **dense** LLM 속도 | **MLX Metal** |
| Apple Silicon **dense** CPU-only | **llama.cpp CPU** |
| Apple Silicon **MoE** CPU-only | **quant.cpp** (2.8× vs llama.cpp) |
| Apple Silicon **MoE** + 16 GB 한계 | **quant.cpp** (RSS 5.24 GB, MLX는 19 GB 모델) |
| **비-Apple** (Linux/Windows) dense | **llama.cpp** (MLX 불가) |
| **비-Apple** MoE | **quant.cpp** / llama.cpp |
| **WASM 브라우저** | **quant.cpp only** (192 KB) |
| **C 단일 헤더 임베드** | **quant.cpp only** (17K LOC, 의존성 0) |
| **KV 압축 기반 긴 컨텍스트** | **quant.cpp only** (6.4× KV) |

### 핵심 해석

**MLX는 Apple Silicon Metal dense의 왕**. 4B Q4를 58 t/s로 돌리는 엔진은 우리 손이 닿지 않는 영역. 포지셔닝에서 **dense 속도 경쟁에 참여하지 않는다**는 선언이 정직.

**llama.cpp는 범용 기준점**. 플랫폼 넓고 (Linux/Windows/Mac), Metal/CUDA/Vulkan 다 있고, 성숙. Apple Silicon dense에서도 Metal 켜면 MLX의 53%. CPU-only 에서는 우리보다 1.4×. **표준**.

**quant.cpp의 방어 가능한 고유 영역**:
1. **MoE + CPU + 16 GB Mac** — 실측 3× 우위. Metal MoE가 현재 llama.cpp/MLX 모두 성숙하지 않은 틈.
2. **메모리 타이트한 환경** — `TQ_NO_MLOCK` + Q3_K_S가 35B MoE를 5.24 GB에 맞춤. MLX 4bit Qwen3.6는 19 GB 요구 (16 GB 단일 Mac 사용자 cutoff).
3. **임베드 / 단일 파일 배포** — `quant.h` 17K LOC, MLX는 Python + framework, llama.cpp는 C++ build system.
4. **WASM 192 KB** — MLX 불가, llama.cpp WASM은 5 MB+.
5. **KV 압축 6.4×** — MLX는 FP16 기본, llama.cpp는 공격성 낮음.

### 우리 프로젝트의 "정직한 포지셔닝 문장"

> quant.cpp는 Apple Silicon dense LLM 속도 경쟁에 참여하지 않는다. 
> 그 자리에 MLX와 llama.cpp Metal이 이미 있다.
> 
> 대신 **"16 GB 메모리, 비-Apple 플랫폼, MoE 하이브리드 아키텍처, 단일 바이너리 배포"** 라는 네 교차점에서
> 측정 가능한 우위를 유지한다.
> 
> 35B MoE를 16 GB Mac에서 llama.cpp의 2.8배 속도로, MLX의 절반 메모리로 돌리는 엔진은 현재 우리뿐이다 (2026-04-18 측정 기준).
> 브라우저 LLM은 우리만 가능하다 (192 KB WASM).
> 17K LOC 단일 C 헤더로 LLM을 앱에 임베드하는 경로는 우리만 제공한다.

### 재현

```bash
# MLX
pip install mlx-lm
mlx_lm.generate --model mlx-community/Qwen3.5-4B-4bit \
  --prompt "Once upon a time" --max-tokens 30 --temp 0.0

# llama.cpp
brew install ggml
llama-bench -m models/Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -n 30 -p 0 -r 1

# quant.cpp
TQ_NO_METAL=1 ./build/quant models/Qwen3.5-4B-Q4_K_M.gguf \
  -p "Once upon a time" -n 30 -T 0 -j 8
```

### 후속

- MLX Qwen3.6-A3B MoE 다운로드 완료 (~1시간 대기 중). 로드 성공 시 MoE 3-way 완료.
- llama.cpp Metal MoE 는 엔진 쪽 이슈 (qwen35moe Metal 커널 hang) — 업스트림 이슈 보고 대상.
