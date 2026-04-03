# Hacker News 한글 포스팅 — 2026-03-31

## 제목

Show HN: 순수 C LLM 엔진, KV 캐시 3.8배 압축 (9K 줄, 의존성 없음)

## URL

https://github.com/quantumaikr/quant.cpp

## 첫 댓글

LLM 추론 시 KV 캐시를 실시간 압축하는 순수 C 엔진을 만들었습니다.

문제: 32K 토큰 이상의 긴 컨텍스트에서 KV 캐시가 가중치보다 많은 메모리를 차지합니다. 4B 모델의 32K 컨텍스트 KV 캐시는 FP16으로 4.4 GB입니다.

quant.cpp는 KV 캐시를 추론 중 Q4로 양자화하여 1.2 GB로 줄입니다 (3.8배 압축, FP16 출력 대비 코사인 유사도 0.999). 최신 논문 3편 기반: quant.cpp (ICLR '26), QJL (AAAI '25), PolarQuant (AISTATS '26).

기술 상세:
- C11 9,000줄, libc만 사용, 외부 의존성 없음
- Q4 가중치 양자화 + ARM NEON 2-row 배치
- 스레드 풀, 정수 Q4×Q8 어텐션 (vdotq_s32)
- 멀티 아키텍처: Qwen3.5 (DeltaNet) + Gemma 3 (슬라이딩 윈도우)
- 듀얼 토크나이저: GPT2 바이트 BPE + SentencePiece 자동 감지
- TQM 포맷: 사전 양자화 mmap 바이너리

llama.cpp와 단일 스레드 속도 동등 (51 vs 50.7 tok/s). 핵심 가치는 속도가 아닌 긴 컨텍스트에서의 메모리 효율입니다.

Claude Code와 함께 2일 만에 구축. v0.1.0 릴리스.
