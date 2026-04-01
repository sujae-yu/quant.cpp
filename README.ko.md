# TurboQuant.cpp

**[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV 캐시 압축을 구현한 독립형 C 추론 엔진. 래퍼가 아닌 자체 구축, 외부 의존성 없음.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![Tests](https://img.shields.io/badge/tests-31%20pass-brightgreen)]()
[![ASan](https://img.shields.io/badge/ASan%2BUBSan-clean-brightgreen)]()

```
Gemma 3 4B perplexity (101 토큰, teacher-forced):
  FP16 KV:         PPL = 35.99
  1-bit K + Q4 V:  PPL = 36.00  (+0.03%)   ← 4.9x 압축, 품질 손실 거의 없음

32K 컨텍스트 메모리 (Gemma 3 4B):
  FP16 K+V:          4,352 MB
  1-bit K + Q4 V:      885 MB   (4.9x, 3.4 GB 절약)
  1-bit K + Q2 V:      613 MB   (7.1x, 3.7 GB 절약)
```

---

## 빠른 시작

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# TQM 포맷 (사전 변환)
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4

# GGUF 포맷 (llama.cpp 모델 직접 로딩)
./build/tq_run model.gguf -p "Hello" -k turbo_kv_1b -v q4
```

---

## 지원 모델

| 모델 | 파라미터 | 포맷 | 속도 | KV 압축 |
|------|----------|------|------|---------|
| **Gemma 3 4B** | 4B | TQM | 20.2 tok/s | PPL +0.03%, 모든 KV 타입 ✓ |
| **Qwen3.5-0.8B** | 752M | TQM | 80.1 tok/s | 모든 KV 타입 ✓ |
| **Qwen3.5-0.8B** | 752M | GGUF Q8_0 | 3.7 tok/s | 1b K + Q4 V ✓ |
| **Gemma 3 270M** | 270M | TQM | 176 tok/s | 모든 KV 타입 ✓ |

아키텍처: Gemma 3 (슬라이딩 윈도우, GeGLU), Qwen3.5 (DeltaNet 하이브리드).

GGUF 지원: Q8_0 검증 완료. K-quant(Q4_K, Q6_K) 및 IQ2 역양자화는 구현되었으나 품질 미검증 — 기여 환영.
MoE 아키텍처 (Qwen3.5-35B-A3B): 로딩과 라우팅 구현 완료, 품질 검증 진행 중.

---

## KV 압축

Key는 RHT + 부호 해싱(1비트) 또는 Lloyd-Max 코드북(3/4비트)으로 압축.
Value는 독립적으로 Q4 또는 Q2로 양자화.

```bash
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q4   # 4.9x 총 K+V
./build/tq_run model.tqm -p "Hello" -k turbo_kv_1b -v q2   # 7.1x 총 K+V
./build/tq_run model.tqm -p "Hello" -k turbo_kv_3b          # 3-bit keys, FP16 values
./build/tq_run model.tqm -p "Hello" -M                       # 메모리 통계
```

| 구성 | K+V/토큰 (Gemma 4B) | 압축률 | PPL 영향 |
|------|---------------------|--------|----------|
| FP16 K+V | 136.00 KB | 1.0x | 기준 |
| 1-bit K + FP16 V | 74.38 KB | 1.8x | +0.00% |
| 1-bit K + Q4 V | 27.62 KB | 4.9x | +0.03% |
| 1-bit K + Q2 V | 19.12 KB | 7.1x | +17.3% |

> K-only 양자화(V는 FP16)는 perplexity 무손실.
> Q4 V는 +0.03% PPL — 사실상 무손실. Q2 V는 눈에 띄게 저하.

---

## 알고리즘

```
Key:   key → L2 정규화 → RHT → Lloyd-Max 코드북 (b-1 bits) → QJL 부호 (1 bit)
       1-bit: 부호만 → XOR + popcount attention

Value: value → 블록별 Q4/Q2 양자화 → packed nibble에서 직접 fused 누적
```

[TurboQuant 논문](https://arxiv.org/abs/2504.19874) (ICLR 2026)은 일반 양자화기가 내적 추정에 체계적 편향을 도입함을 증명. RHT + QJL 보정으로 추정기가 증명 가능하게 비편향.

---

## 분석 도구

```bash
./build/tq_run model --ppl input.txt -k turbo_kv_1b -v q4   # perplexity
./build/tq_run model --profile-kv -k turbo_kv_1b -p "text"  # 활성값 분포
./build/tq_run model --recommend -k turbo_kv_1b -p "text"   # 레이어별 비트 할당
./build/tq_run model --calibrate -k turbo_kv_1b -p "text"   # 코드북 캘리브레이션
./build/tq_run model --attn-entropy -k turbo_kv_1b -p "text" # attention 엔트로피
bash bench/auto_profile.sh model                              # 전체 파이프라인
```

---

## 검증

| 항목 | 결과 | 재현 방법 |
|------|------|----------|
| Perplexity (1b K + Q4 V) | PPL +0.03% vs FP16 | Gemma 4B `--ppl` |
| 비편향성 | 상대 bias < 0.2%, 10만 샘플 | `test_unbiased` |
| Attention 코사인 (1-bit) | 0.634 = 이론 한계 2/pi | `test_attention_distribution` |
| Lloyd-Max 코드북 | MSE가 정보이론 최적의 1.18배 이내 | `test_codebook_theory` |
| 코드북 캘리브레이션 | 실제 활성값에서 MSE 49.7% 개선 | `--calibrate` |
| 누적 오차 (16 레이어) | 코사인 0.998 (Q4), 준선형 성장 | `test_cumulative_error` |
| NEON/스칼라 일치성 | 14개 경로 검증 | `test_neon_scalar` |
| 엣지케이스 | 29개 (NaN, Inf, n=1, dim=0) | `test_edge_cases` |
| ASan + UBSan | 31/31 클린 | `scripts/sanitize.sh` |
| Rate-distortion gap | Q4: 하한 대비 2.41배 | `test_rate_distortion` |

벤치마크: `bench/ablation_test.sh`, `bench/kv_quality_bench.sh`, `bench/long_quality_test.sh`, `bench/sampling_test.sh`

---

## FAQ

**Q: "1-bit attention 코사인 0.634는 너무 낮지 않나?"**
2/pi = 0.637이 부호 양자화의 정보이론적 최대값. 우리 0.634가 이 한계에 도달. 더 높은 코사인이 필요하면 3-bit(0.918) 사용.

**Q: "llama.cpp KV 양자화와 뭐가 다른가?"**
llama.cpp는 uniform min-max. TurboQuant는 RHT + Lloyd-Max + QJL 잔차 보정으로 증명 가능한 비편향 내적 추정. 코드북 centroid 이론 검증 완료 (`test_codebook_theory`).

**Q: "Perplexity는?"**
측정 완료. Gemma 4B 1-bit K + Q4 V: PPL = 36.00 vs 35.99 기준 (+0.03%). K-only 양자화는 정확히 무손실 (PPL 동일). `--ppl` 플래그 참조.

**Q: "NEON 코드가 정확한가?"**
모든 NEON 경로를 스칼라 참조와 비교 검증 (`test_neon_scalar`). Q4 dequant nibble 인터리빙 버그를 검증 과정에서 발견 후 수정. ASan + UBSan 31개 전체 스위트 클린.

**Q: "RHT 오버헤드는?"**
128차원 벡터당 147 ns (NEON 벡터화). 1-bit attention: 1.2 ns/key. matmul (~1ms/레이어) 대비 무시 가능. `bench/bench_kv_overhead.cpp` 참조.

**Q: "소형 모델만 지원?"**
GGUF Q8_0은 Qwen3.5 0.8B에서 검증 완료. MoE 아키텍처(35B-A3B)는 로딩과 라우팅이 구현되어 있으며, K-quant/IQ2 역양자화 품질을 안정화 중. 엔진과 KV 압축은 아키텍처 독립적 — 270M~4B에서 검증.

---

## 기술 상세

- **15,000줄+ 순수 C** — 외부 의존성 없음
- **GGUF v3 로딩** — Q8_0 검증 완료; K-quant/IQ2 역양자화 구현 (품질 WIP)
- **MoE 라우팅** — top-K expert 선택, 공유 전문가, SwiGLU (품질 WIP)
- **12개 KV 양자화 타입** — Uniform, PolarQuant, QJL, TurboQuant, TurboQuant KV (1/3/4-bit)
- **Fused Q4 attention** — packed nibble에서 직접 가중합
- **적응적 압축** — 레이어별 비트 추천, 코드북 캘리브레이션
- **NEON 벡터화** — matmul, attention, RHT, Hamming distance, Q4 dequant
- **31개 테스트 스위트** — perplexity, 비편향성, attention 분포, 코드북 이론, NEON 일치성, 엣지케이스, rate-distortion, 누적 오차

---

## 참고 논문

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — 근최적 왜곡률의 온라인 벡터 양자화
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1비트 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 KV 양자화

전체 변경 이력: [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
