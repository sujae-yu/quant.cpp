# TurboQuant.cpp

![TurboQuant Hero](docs/assets/hero.png)

**LLM 추론을 위한 극한 KV 캐시 압축. 외부 의존성 없음. 순수 C.**

동일 하드웨어에서 **3배 긴 컨텍스트** — 또는 동일 비용으로 **3배 많은 사용자**.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-38%2B%20pass-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Qwen3.5 Validated](https://img.shields.io/badge/Qwen3.5--0.8B-validated-blue)]()

---

## 한눈에 보는 결과

| | FP16 (기준) | TurboQuant |
|---|---|---|
| **KV 캐시 크기** | 7.00 GB | **0.93 GB** (87% 절약) |
| **Attention 속도** | 1.0x | **2.9-4.8배 빠름** |
| **최대 컨텍스트 (24GB GPU)** | 164K 토큰 | **540K 토큰** |
| **품질 (코사인)** | 1.000 | **0.994** (A+) |

> Llama-3.2-3B @ 64K 기준. [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 실제 추론으로 검증.

---

## 지금 바로 체험 (30초)

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp
cd TurboQuant.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON -DTQ_BUILD_BENCH=ON
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

# A/B 비교 직접 확인
./build/ab_test

# 실제 LLM 모델별 메모리 절약
./build/demo_real_model

# 속도: 정수 Attention vs FP32
./build/speed_int_vs_float
```

### Python

```bash
pip install -e bindings/python

python3 examples/python_quickstart.py
```

```python
from turboquant import TurboQuant
import numpy as np

tq = TurboQuant("cpu")
keys = np.random.randn(512, 128).astype(np.float32) * 0.15

compressed = tq.quantize_keys(keys, TurboQuant.UNIFORM_4B)
print(f"압축: {keys.nbytes:,} → {len(compressed):,} bytes ({keys.nbytes/len(compressed):.1f}x)")
```

### C

```c
#include "turboquant/turboquant.h"

tq_context_t* ctx;
tq_init(&ctx, TQ_BACKEND_CPU);

// 7.5배 압축, 한 줄
tq_quantize_keys(ctx, keys, n, dim, TQ_TYPE_UNIFORM_4B, out, size);

// 압축된 캐시에서 직접 Attention — FP32보다 2.9배 빠름
tq_attention(ctx, query, out, n, dim, TQ_TYPE_UNIFORM_4B, scores);
```

---

## 세 가지 돌파구

### 1. 작을 뿐 아니라 더 빠르다

대부분의 양자화는 작아지지만 느려집니다. TurboQuant은 정수 도메인에서 직접 계산하여 attention이 **FP32보다 2.9-4.8배 빠릅니다**.

```
FP32:    query × key = float dot       → 22.8 μs
Q4×Q8:   int_query × int_key = int_dot →  7.8 μs  (2.9배 빠름)
```

### 2. 실제 모델로 검증

합성 벤치마크가 아닌 실제 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) KV 캐시:

| 타입 | 압축률 | 품질 | 등급 |
|------|--------|------|------|
| **uniform_4b** | 7.5x | 코사인 0.994 | **A+** |
| **mixed_4b8** | 6.4x | 코사인 0.994 | **A+** |
| uniform_2b | 14.2x | 코사인 0.953 | A |

### 3. 커뮤니티 검증 아키텍처

r/LocalLLaMA 커뮤니티와 llama.cpp Discussion #20969에서 검증된 기법:

- **정수 내적** (llama.cpp `vec_dot` 패턴)
- **Random Hadamard Transform** (Qwen3.5에서 MSE 3.9배 감소)
- **K/V 비대칭** 양자화 (Key 4bit + Value 2bit = 9.8배 압축)
- **Mixed Precision** 아웃라이어 탐지 (fp16 + 4bit)

---

## 얼마나 절약되나?

| 모델 | GPU | FP16 컨텍스트 | TurboQuant | 향상 |
|------|-----|-------------|------------|------|
| Qwen3.5-0.8B | 8GB M2 Air | 87K | **286K** | 3.3x |
| Llama-3.2-1B | 16GB RTX 4060 | 445K | **1,462K** | 3.3x |
| Llama-3.2-3B | 24GB RTX 4090 | 164K | **540K** | 3.3x |

---

## 문서

| 문서 | 설명 |
|------|------|
| **[시작 가이드](docs/getting-started.md)** | **빌드, CLI, Python, C API, llama.cpp — 한 페이지에 모두** |
| [아키텍처](docs/architecture.md) | 4-layer 설계, 타입 시스템, 디스패치 |
| [Qwen3.5 검증](docs/qwen35_validation_results.md) | 실제 모델 A/B 테스트 결과 |
| [통합 가이드](docs/integration_guide.md) | llama.cpp, vLLM, Python |
| [llama.cpp 플러그인](integrations/llamacpp/README.md) | llama.cpp 통합 단계별 가이드 |
| [포맷 사양](spec/tq_format_v1.md) | 블록 구조, 비트 패킹 |
| [변경 이력](CHANGELOG.md) | 전체 릴리즈 노트 |

---

## 기술 특징

- **8개 양자화 타입** — Uniform, Mixed Precision, PolarQuant, QJL, TurboQuant
- **정수 도메인 Attention** — Q4×Q8, ARM `vdotq_s32` / x86 VNNI
- **외부 의존성 제로** — 순수 C11/C++17, libc/libm만 사용
- **스레드 안전** — pthread mutex, TSan 검증
- **38+ 테스트** — ASan + UBSan + TSan 클린
- **GPU 대응** — CUDA + Metal 커널 포함

---

## 참고 논문

- **TurboQuant** — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **QJL** — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482) (AAAI 2025)
- **PolarQuant** — [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)

---

**개발사: [QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
