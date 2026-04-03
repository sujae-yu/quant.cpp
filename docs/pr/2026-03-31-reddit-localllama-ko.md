# r/LocalLLaMA 한글 포스팅 — 2026-03-31

## 제목

quant.cpp — KV 캐시 3.8배 압축으로 32K 컨텍스트에서 llama.cpp 대비 3.2 GB 절약하는 순수 C 추론 엔진

## 본문

llama.cpp가 하지 않는 것 하나에 집중한 C 추론 엔진을 만들었습니다: **KV 캐시 압축**.

짧은 컨텍스트에서는 KV 메모리가 큰 문제가 아닙니다. 하지만 32K 토큰 이상에서는 모델 가중치보다 KV 캐시가 더 많은 메모리를 차지합니다.

**실측 데이터 (Gemma 3 4B):**

```
컨텍스트    llama.cpp KV (FP16)    quant.cpp KV (Q4)    절약
─────────   ──────────────────     ──────────────────    ──────
4K 토큰           544 MB                145 MB           399 MB
32K 토큰        4,352 MB              1,156 MB         3,196 MB
128K 토큰      17,408 MB              4,624 MB        12,784 MB
```

3.8배 압축, PyTorch 대비 레이어별 정확도 검증 완료.

**속도는 경쟁력 있지만 핵심이 아닙니다:**
- 단일 스레드 Q4: 51.1 tok/s (llama.cpp: 50.7 tok/s) — 동등 수준
- 더 빠르다는 주장이 아닙니다

**차별점:**
- ICLR 2026 quant.cpp 논문 기반 KV 캐시 3.8배 압축
- 3개 모델 지원: Gemma 3 4B, Qwen3.5-0.8B, Gemma 3 270M
- 순수 C, 외부 의존성 없음, ~1MB 바이너리
- 멀티 아키텍처: DeltaNet (Qwen) + 슬라이딩 윈도우 (Gemma)
- Gemma 4 대응 준비 완료

```bash
git clone https://github.com/quantumaikr/quant.cpp && cd quant.cpp
bash scripts/quickstart.sh "What is deep learning?"
```

2일 만에 구축. C 9,000줄. 테스트 스위트 20개. 첫 릴리스 v0.1.0.

KV 캐시 압축은 제한된 RAM에서 긴 컨텍스트를 사용하는 시나리오에서 가장 큰 가치를 가집니다 — 로컬 LLM 사용자에게 가장 중요한 시나리오입니다.

GitHub: https://github.com/quantumaikr/quant.cpp
