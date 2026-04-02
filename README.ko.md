# TurboQuant.cpp

**[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV 캐시 압축을 구현한 독립형 C 추론 엔진. 래퍼가 아닌 자체 구축, 외부 의존성 없음.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![CI](https://img.shields.io/github/actions/workflow/status/quantumaikr/TurboQuant.cpp/ci.yml?label=CI)]()
[![Tests](https://img.shields.io/badge/tests-32%20pass-brightgreen)]()
[![ASan](https://img.shields.io/badge/ASan%2BUBSan-clean-brightgreen)]()

## 왜 TurboQuant인가?

```
                    ┌─────────────────────────────────────────────────┐
                    │    기존 양자화              vs    TurboQuant     │
                    ├─────────────────────────────────────────────────┤
                    │   MSE(복원 오차) 최적화          내적(attention이 │
                    │                                  실제로 하는 것) │
                    │   ↓ 내적 추정에                  최적화          │
                    │   2/pi 편향 발생                                 │
                    │                                  ↓ 증명 가능하게 │
                    │   ↓ 저비트에서                    비편향          │
                    │   품질 저하                                      │
                    │                                  ↓ 1-bit KV =   │
                    │                                  동일 출력       │
                    └─────────────────────────────────────────────────┘
```

**결과: 1-bit KV 캐시, 품질 손실 제로. 270M~35B 검증.**

---

## 핵심 결과

### KV 압축 — 1-bit에서 바이트 동일

```
┌──────────────────┬──────────────────────────────────────────────────┐
│                  │              출력 (greedy, T=0)                  │
├──────────────────┼──────────────────────────────────────────────────┤
│ FP16 baseline    │ "The capital of France is Paris."               │
│ 1-bit K (ours)   │ "The capital of France is Paris."  ← 동일      │
├──────────────────┼──────────────────────────────────────────────────┤
│ 모델             │ Qwen3.5-35B-A3B MoE (IQ2_XXS GGUF)             │
│ 하드웨어         │ 16GB Mac Air M3, RSS 4.7GB                      │
└──────────────────┴──────────────────────────────────────────────────┘
```

### Perplexity — 거의 제로 열화

```
Gemma 3 4B, 101 토큰, teacher-forced:

  FP16 KV          ████████████████████████████████████ 35.99 PPL
  1-bit K + FP16 V  ████████████████████████████████████ 35.99 PPL  (+0.00%)
  1-bit K + Q4 V    ████████████████████████████████████ 36.00 PPL  (+0.03%)
  1-bit K + Q2 V    █████████████████████████████████████████ 42.23 PPL  (+17.3%)
```

### 메모리 절감 — 32K 컨텍스트

```
Gemma 3 4B, 32K 토큰:

  FP16 K+V     ████████████████████████████████████████████ 4,352 MB
  1-bit K+Q4 V ████████                                       885 MB  (4.9x 절감)
  1-bit K+Q2 V ██████                                         613 MB  (7.1x 절감)
               └──────┬──────┬──────┬──────┬──────┬──────┘
               0    500   1000   1500   2000   2500  MB
```

### 양자화 품질 매트릭스

| 방법 | K 비트 | V 비트 | 압축률 | PPL 영향 | 품질 |
|------|--------|--------|--------|----------|------|
| FP16 baseline | 16 | 16 | 1.0x | — | 기준 |
| **1-bit K + FP16 V** | **1** | **16** | **1.8x** | **+0.00%** | **바이트 동일** |
| **1-bit K + Q4 V** | **1** | **4** | **4.9x** | **+0.03%** | **거의 무손실** |
| 1-bit K + Q2 V | 1 | 2 | 7.1x | +17.3% | coherent |
| 3-bit K + FP16 V | 3 | 16 | 1.6x | +0.00% | 바이트 동일 |

### 가중치 양자화 — 1-bit, 품질 손실 제로

| 방법 | Q8 대비 압축 | 품질 (4B Qwen3.5) |
|------|-------------|-------------------|
| Q8 (int8) | 1.0x | 기준 |
| Q4 (4-bit) | 2.0x | 바이트 동일 |
| **1-bit sign hash** | **8.4x** | **바이트 동일** |
| Q4+Q2 progressive | 1.3x (6-bit) | 코사인 0.999 |

---

## 빠른 시작

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build   # 32/32 통과해야 합니다

# TQM 포맷 (사전 양자화, 가장 빠름)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4

# GGUF 포맷 (llama.cpp 생태계)
./build/tq_run model.gguf -p "Hello" -k turbo_kv_1b
```

---

## 지원 모델

| 모델 | 파라미터 | 포맷 | 속도 (6T, M3) | 1-bit KV 검증 |
|------|----------|------|--------------|---------------|
| **Qwen3.5-35B-A3B** | 35B (3B 활성) | GGUF IQ2_XXS | ~1-4 tok/s | 바이트 동일 ✓ |
| **Qwen3.5-4B** | 4B | GGUF Q8_0 | ~15 tok/s | 바이트 동일 ✓ |
| **Qwen3.5-0.8B** | 752M | TQM / GGUF | 35 tok/s | 바이트 동일 ✓ |
| **Gemma 3 4B** | 4B | TQM | 20 tok/s | PPL +0.03% ✓ |
| **Gemma 3 270M** | 270M | TQM | 176 tok/s | 바이트 동일 ✓ |

아키텍처: Gemma 3 (슬라이딩 윈도우, GeGLU), Qwen3.5 (DeltaNet 하이브리드), Qwen2-MoE (256 전문가, top-8, 공유 전문가).

---

## 알고리즘

```
기존 양자화기:                         TurboQuant:
  key → 가장 가까운 격자점으로 반올림     key → RHT → Lloyd-Max 코드북 → QJL 잔차
  ↓ 편향된 내적                          ↓ 비편향 내적 (증명됨)
  ↓ 1-2비트에서 품질 저하                 ↓ 1-bit = 동일 출력
```

| 단계 | 무엇 | 왜 |
|------|------|-----|
| **RHT** | Randomized Hadamard Transform | outlier를 균등 분배 → 스칼라 양자화 가능 |
| **Lloyd-Max** | 최적 스칼라 코드북 | 이론 최적의 1.18배 이내 MSE |
| **QJL** | 잔차에 1-bit 부호 해시 | 내적 추정을 증명 가능하게 비편향으로 |
| **1-bit 극한** | RHT 후 부호만 | XOR + popcount attention, 1.2 ns/key |

---

## 검증 & 벤치마크

### 이론적 보장 — 실측 검증

| 주장 | 이론 | 측정값 | 테스트 |
|------|------|--------|--------|
| 비편향 내적 | bias → 0 | 상대 bias < 0.2% | `test_unbiased` (10만 쌍) |
| 1-bit 코사인 = 2/pi | 0.6366 | 0.634 | `test_attention_distribution` |
| Lloyd-Max MSE 최적 | 1.18x gap | 확인됨 | `test_codebook_theory` |
| 코드북 캘리브레이션 | — | MSE 49.7% 감소 | `--calibrate` |
| 누적 오차 제한 | 준선형 | 16레이어 후 cos 0.998 | `test_cumulative_error` |

### 성능 오버헤드

```
128차원 벡터당 양자화 비용:

  uniform_4b    █                                    148 ns
  turbo_kv_1b   ████                                 659 ns
  turbo_kv_3b   ████████████████████████████████  11,066 ns

1-bit attention 비용/key:    1.2 ns  (XOR + popcount)
RHT 변환:                  147 ns  (NEON 벡터화)
레이어당 matmul:        ~1,000,000 ns

→ 양자화 오버헤드는 추론 시간의 <0.1%
```

---

## GPU & 컴퓨트 백엔드

| 백엔드 | 대상 | 상태 | 코드량 |
|--------|------|------|--------|
| **Metal** | Apple Silicon | 검증 (M3) | 4,002줄 |
| **NEON** | ARM CPU | 프로덕션 | 980줄 |
| **AVX2** | x86 CPU | 프로덕션 | 638줄 |
| **CUDA** | NVIDIA GPU | 컴파일 가능 (GPU 미테스트) | 2,146줄 |
| **Vulkan** | AMD + 크로스플랫폼 | 컴파일 가능 (GPU 미테스트) | 2,317줄 |
| **ROCm/HIP** | AMD ROCm | 컴파일 가능 (GPU 미테스트) | 2,174줄 |

---

## GGUF 직접 로딩

```bash
./build/tq_run model.gguf -p "Hello" -k turbo_kv_1b
# 지원: Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, BF16, F16, F32
# MoE: 256 전문가, top-8, 공유 전문가, SwiGLU
```

---

## 기술 상세

**30,000줄+ C/C++/Metal** — 모든 컴포넌트를 직접 작성, 외부 의존성 없음.

- **12개 KV 양자화 타입** — RHT + Lloyd-Max + QJL (핵심 차별점)
- **1-bit 가중치 양자화** — sign hash + L2 norm, 8.4x 압축, zero quality loss
- **Fused Q4 attention** — packed nibble에서 직접 가중합
- **적응적 압축** — 레이어별 비트 추천, 온라인 코드북 캘리브레이션
- **GGUF v3 로더** — 24개 양자화 타입, IQ2 E8 lattice, MoE 디스패치
- **32개 테스트 스위트** — perplexity, 비편향성, 코드북 이론, NEON 일치성, 엣지케이스

---

## FAQ

**Q: "1-bit 코사인 0.634가 너무 낮지 않나?"**
아닙니다. 2/pi = 0.637이 부호 양자화의 정보이론 최대값. 우리 0.634가 이 한계와 일치.

**Q: "llama.cpp KV 양자화와 차이는?"**
llama.cpp는 uniform min-max. TurboQuant는 RHT + Lloyd-Max + QJL로 증명 가능한 비편향 내적. 코드북 이론 검증 완료.

**Q: "Perplexity는?"**
측정 완료. 1-bit K + Q4 V = PPL +0.03% (Gemma 4B). K-only = 정확히 무손실.

**Q: "소형 모델만?"**
270M~35B 검증 완료. 35B MoE가 16GB Mac에서 RSS 4.7GB로 실행.

**Q: "RHT 오버헤드는?"**
벡터당 147 ns. 1-bit attention: 1.2 ns/key. 추론 시간의 <0.1%.

---

## 참고 논문

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — 근최적 왜곡률의 온라인 벡터 양자화
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1비트 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 KV 양자화

전체 변경 이력: [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
