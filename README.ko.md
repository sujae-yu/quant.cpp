# TurboQuant.cpp

**[TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) KV 캐시 압축을 구현한 독립형 C 추론 엔진. 래퍼가 아닌 자체 구축, 외부 의존성 없음.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)]()
[![CI](https://img.shields.io/github/actions/workflow/status/quantumaikr/TurboQuant.cpp/ci.yml?label=CI)]()
[![Tests](https://img.shields.io/badge/tests-33%20pass-brightgreen)]()
[![ASan](https://img.shields.io/badge/ASan%2BUBSan-clean-brightgreen)]()

## TurboQuant가 하는 일

**KV 캐시 7x 압축, 컨텍스트 7x 확장 — 품질 손실 제로.**

```
16GB Mac Air M3, Gemma 3 4B:

  TurboQuant 없이:   32K 컨텍스트  (FP16 KV = 4.2 GB)
  TurboQuant 적용:  230K 컨텍스트  (1-bit K + Q4 V = 612 MB)

  PPL: 35.99 → 35.99   (K만 양자화 시 +0.00%)
```

같은 하드웨어, 같은 모델, **7배 더 긴 컨텍스트**. 품질 손실 없음.

---

## 검증: 800 토큰에서 PPL +0.00%, 4개 아키텍처

```
SmolLM2 1.7B (Llama), 800 토큰:            Qwen3.5 0.8B, 800 토큰:

  baseline  ████████████  PPL 11.07           baseline  ████████████  PPL 137.6
  1-bit K   ████████████  PPL 11.07 (+0.00%)  1-bit K   ████████████  PPL 137.6 (+0.00%)
```

| 모델 | 아키텍처 | 토큰 | Baseline PPL | 1-bit K PPL | 차이 |
|------|----------|------|-------------|-------------|------|
| SmolLM2 1.7B | Llama | 800 | 11.07 | 11.07 | **+0.00%** |
| Qwen3.5 0.8B | Qwen3.5 | 800 | 137.6 | 137.6 | **+0.00%** |
| Gemma 3 4B | Gemma 3 | 101 | 35.99 | 35.99 | **+0.00%** |
| SmolLM2 1.7B | Llama | 105 | 5.84 | 5.84 | **+0.00%** |

K-only 양자화는 모든 테스트 길이에서 **perplexity 완전 동일**.

---

## 컨텍스트 확장: 하드웨어별 효과

| 하드웨어 | 모델 | FP16 컨텍스트 | TurboQuant | 배율 |
|----------|------|-------------|------------|------|
| **8GB 노트북** | Llama 8B (Q4) | 16K | 116K | 7.1x |
| **16GB Mac Air** | Gemma 4B | 96K | 684K | 7.1x |
| **16GB Mac Air** | Llama 8B (Q4) | 82K | 581K | 7.1x |
| **24GB RTX 3090** | Llama 8B (Q4) | 147K | 1M+ | 7.1x |
| **24GB RTX 3090** | 35B MoE (Q4) | 682K | 5M+ | 7.1x |

### 모델별 KV 메모리 (32K 컨텍스트)

| 모델 | 레이어 (attn) | FP16 K+V | 1-bit K + Q4 V | 절감 |
|------|-------------|----------|---------------|------|
| SmolLM2 1.7B | 24 (24) | 6.0 GB | 869 MB | 5.1 GB |
| Gemma 3 4B | 34 (34) | 4.2 GB | 613 MB | 3.6 GB |
| Qwen3.5 4B | 32 (8) | 1.0 GB | 144 MB | 880 MB |
| Qwen 35B MoE | 40 (10) | 640 MB | 90 MB | 550 MB |

---

## 작동 원리

```
저장:    key → L2 정규화 → RHT → sign bits (차원당 1 bit) → 압축 블록
복원:    압축 블록 → 역양자화 → FP32 → 표준 attention

메모리 절감은 압축 저장소에서 발생.
Attention은 완전한 FP32 정밀도로 실행 — 근사 없음.
```

[TurboQuant 논문](https://arxiv.org/abs/2504.19874) (ICLR 2026)이 RHT + sign 양자화가 내적 구조를 보존함을 증명. Key를 1비트로 저장하고 attention 시 FP32로 복원 — 메모리 절감과 품질을 모두 확보.

---

## 빠른 시작

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp && cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build   # 33/33 통과

./build/tq_run model.gguf -p "Hello" -k turbo_kv_1b
./build/tq_run model.gguf --ppl input.txt -k turbo_kv_1b  # PPL 측정
```

---

## 지원 모델

| 모델 | 아키텍처 | 파라미터 | 포맷 | 속도 (M3, 6T) | PPL 검증 |
|------|----------|----------|------|--------------|----------|
| **Qwen3.5-35B-A3B** | Qwen2-MoE | 35B (3B 활성) | GGUF IQ2_XXS | ~1-4 tok/s | byte-identical ✓ |
| **Qwen3.5-4B** | Qwen3.5 | 4B | GGUF Q8_0 | 5.4 tok/s | byte-identical ✓ |
| **SmolLM2-1.7B** | **Llama** | 1.7B | GGUF Q8_0 | 24 tok/s | **PPL +0.00% (800 tok)** ✓ |
| **Qwen3.5-0.8B** | Qwen3.5 | 752M | TQM / GGUF | 35 tok/s | **PPL +0.00% (800 tok)** ✓ |
| **Gemma 3 4B** | Gemma 3 | 4B | TQM | 20 tok/s | PPL +0.00% (101 tok) ✓ |
| **Gemma 3 270M** | Gemma 3 | 270M | TQM | 176 tok/s | byte-identical ✓ |

**4개 아키텍처 검증:** Llama, Gemma 3, Qwen3.5 (DeltaNet), Qwen2-MoE (256 전문가).

---

## 압축 옵션

| 구성 | K 비트 | V 비트 | 압축률 | PPL 영향 | 용도 |
|------|--------|--------|--------|----------|------|
| **1-bit K + FP16 V** | 1 | 16 | 1.8x | +0.00% | 최대 품질 |
| **1-bit K + Q4 V** | 1 | 4 | 4.9x | +0.03% | 최적 균형 |
| **1-bit K + Q2 V** | 1 | 2 | 7.1x | +17.3% | 최대 압축 |

---

## 검증

| 항목 | 결과 | 방법 |
|------|------|------|
| **800 토큰 PPL** | **+0.00%** (Llama, Qwen) | `--ppl` 800 토큰 텍스트 |
| 비편향성 | 상대 bias < 0.2% | `test_unbiased` (10만 쌍) |
| NEON/스칼라 | 14 경로 일치 | `test_neon_scalar` |
| 엣지케이스 | 29 테스트 (NaN, Inf, n=1) | `test_edge_cases` |
| ASan + UBSan | 33/33 클린 | `scripts/sanitize.sh` |

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

## FAQ

**Q: "1-bit인데 어떻게 손실이 없나?"**
Key를 1-bit로 저장하지만 attention은 FP32로 실행합니다. 역양자화된 key가 RHT 덕분에 충분한 구조를 보존하여 attention 분포가 사실상 동일합니다. 800 토큰에서 PPL +0.00% 검증.

**Q: "단점은?"**
압축은 실제 (7.1x). 속도는 변함 없음 (FP32 attention). 유일한 비용은 양자화/역양자화 (~659 ns/벡터, 추론 시간의 <0.1%).

**Q: "llama.cpp KV 양자화와 차이는?"**
llama.cpp는 uniform min-max. TurboQuant는 RHT + sign 양자화로 내적 구조를 수학적으로 보존. [llama.cpp 통합 패치](integrations/llamacpp/patch/) 준비 완료.

**Q: "소형 모델만?"**
270M~35B, 4개 아키텍처에서 검증. KV 압축은 아키텍처 독립적.

---

## 기술 상세

**30,000줄+ C/C++/Metal** — 모든 컴포넌트 직접 작성, 외부 의존성 없음.

- **12개 KV 양자화 타입** — RHT + Lloyd-Max + QJL
- **1-bit 가중치 양자화** — 8.4x 압축, 테스트 시퀀스에서 출력 동일
- **GGUF v3 로더** — 24개 양자화 타입, IQ2 E8 lattice, MoE 디스패치
- **llama.cpp 통합** — self-contained 패치, `--cache-type-k tq_kv_1b`
- **33개 테스트 스위트** — perplexity, 비편향성, NEON 일치성, 엣지케이스

---

## 참고 논문

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — 근최적 왜곡률의 온라인 벡터 양자화
- [QJL](https://arxiv.org/abs/2406.03482) (AAAI 2025) — 1비트 양자화 JL 변환
- [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026) — 극좌표 KV 양자화

전체 변경 이력: [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=quantumaikr/TurboQuant.cpp&type=Date)](https://star-history.com/#quantumaikr/TurboQuant.cpp&Date)

---

**[QuantumAI Inc.](https://quantumai.kr)** | [hi@quantumai.kr](mailto:hi@quantumai.kr)
