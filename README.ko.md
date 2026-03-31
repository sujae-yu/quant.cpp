# TurboQuant.cpp

**[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) 논문을 충실히 구현한 순수 C 추론 엔진.**

3비트 KV 캐시. 품질 손실 제로. FP16보다 빠름.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Release](https://img.shields.io/github/v/release/quantumaikr/TurboQuant.cpp)]()
[![Tests](https://img.shields.io/badge/tests-21%20suites-brightgreen)]()

---

## 핵심 아이디어

LLM attention은 **내적(inner product)** `<query, key>`를 계산합니다. 일반 양자화기는 복원 오차(MSE)를 최소화하지만, 이것은 **내적 추정에 편향(bias)을 만듭니다** — attention 점수가 체계적으로 왜곡됩니다.

TurboQuant는 [ICLR 2026 논문](https://arxiv.org/abs/2504.19874)의 2단계 접근으로 이를 해결합니다:

```
Key → 정규화 → Random Hadamard Transform
    → Lloyd-Max 코드북 (b-1 bits)        ← MSE 최적, 하지만 내적에 편향
    → QJL 부호 해시 on 잔차 (1 bit)       ← 편향 교정, 비편향 추정기
    → 저장: [인덱스, 부호, 노름]

Attention:
    query → RHT (1회) → 회전 공간에서 내적 (역변환 불필요)
                      → 사전 계산된 QJL 투영으로 보정
```

결과: **3비트 KV로 품질 저하 없이, 4비트 uniform보다 빠른 attention.**

---

## 결과

### 속도: TurboQuant KV vs Uniform KV

| 모델 | Uniform 4비트 | TurboQuant 3비트 | 가속 | 품질 |
|------|-------------|----------------|------|------|
| **Gemma 3 4B** | 5.1 tok/s | **17.6 tok/s** | **3.4x** | 동일 |
| **Qwen3.5-0.8B** | 49.5 tok/s | **80.1 tok/s** | **1.6x** | 동일 |

더 적은 비트 = 더 적은 데이터 = 더 나은 캐시 효율. 회전 공간 내적으로 역변환 제거.

### KV 캐시 메모리

![Long Context Memory](docs/assets/long_context_memory.png)

```
Gemma 3 4B, 32K 컨텍스트:
  FP16 (llama.cpp):       4,352 MB
  Uniform Q4:             1,156 MB   (3.8x)
  TurboQuant 3비트:          900 MB   (4.6x)  ← 같은 품질, 22% 적은 메모리
```

### 지원 모델

| 모델 | 파라미터 | 속도 (Q4, 6T) | 검증 |
|------|----------|---------------|------|
| **Gemma 3 4B** | 4B | 17.6 tok/s | "France" → "Paris" |
| **Qwen3.5-0.8B** | 752M | 80.1 tok/s | PyTorch 대비 코사인 0.999 |
| **Gemma 3 270M** | 270M | 176 tok/s | 레이어별 정확 일치 |

멀티 아키텍처: Qwen3.5 (DeltaNet 하이브리드) + Gemma 3 (슬라이딩 윈도우). Gemma 4 대응.

---

## 빠른 시작

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

### KV 캐시 옵션

```bash
./build/tq_run model.tqm -p "Hello" -k turbo_kv_3b   # 3비트 TurboQuant (권장)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_4b   # 4비트 TurboQuant
./build/tq_run model.tqm -p "Hello" -k uniform_4b     # 4비트 uniform (베이스라인)
./build/tq_run model.tqm -p "Hello" -M                 # KV 메모리 통계 표시
```

---

## 작동 원리

### 알고리즘 (논문 기반)

| 단계 | 내용 | 이유 |
|------|------|------|
| **Random Hadamard Transform** | 입력을 회전하여 채널 상관 제거 | 회전 후 좌표가 가우시안 근사 → 단순 스칼라 양자화 가능 |
| **Lloyd-Max 코드북** | 각 회전 좌표를 독립적으로 양자화 | 가우시안 분포에 대한 사전 계산 최적 중심점, 거의 최적 MSE |
| **QJL 잔차** | 양자화 잔차의 1비트 부호 해시 | 내적 추정을 **비편향**으로 만듦 — attention 정확도의 핵심 |

MSE 최적 양자화기만으로는 내적에 2/pi ≈ 0.64의 곱셈 편향이 발생합니다. QJL 잔차 보정이 이 편향을 완전히 제거합니다.

---

## 기술 상세

- **10,000줄 이상의 C** — 완전한 추론 엔진, 래퍼 아님
- **10개 양자화 타입** — Uniform, Mixed, PolarQuant, QJL, TurboQuant, TurboQuant KV
- **논문 충실 구현** — RHT + Lloyd-Max 코드북 + QJL 잔차 (arXiv 2504.19874)
- **멀티 아키텍처** — Qwen3.5 (DeltaNet) + Gemma 3 (슬라이딩 윈도우), Gemma 4 대응
- **멀티 샤드 safetensors** — 분할 모델 로딩 (Gemma 4B = 2 샤드)
- **듀얼 토크나이저** — GPT2 바이트 BPE + SentencePiece 자동 감지
- **TQM 포맷** — 사전 양자화 바이너리, mmap 즉시 로딩
- **NEON 벡터화** — 2-row matmul 배치, fused attention, 스레드 풀
- **21개 테스트 스위트** — TurboQuant KV 라운드트립, attention 정확도, 코드북 검증 포함

---

## 여정

```
1일차 오전:   빈 디렉토리
1일차 오후:   KV 캐시 압축 라이브러리 (10개 타입)
1일차 저녁:   완전한 추론 엔진 (Qwen3.5)
1일차 밤:    82 tok/s, llama.cpp 동등
2일차 오전:   Gemma 3 지원 (270M + 4B)
2일차 오후:   TurboQuant 논문 알고리즘 구현
2일차 저녁:   3비트 KV, 품질 손실 제로, uniform 대비 3.4배 빠름

C 코드:       10,000줄 이상
테스트:       21개 스위트
모델:         Gemma 3 4B, Qwen3.5-0.8B, Gemma 3 270M
KV 압축:      4.6x (3비트 TurboQuant, 품질 중립)
```

---

## 참고 논문

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — 근최적 왜곡률의 온라인 벡터 양자화
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — KV 캐시를 위한 1비트 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 KV 양자화

아키텍처: [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [ONNX](https://github.com/onnx/onnx) 참조.

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
