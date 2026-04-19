# Qwen3.6-35B-A3B Full Quant Matrix on 16 GB M1 Pro

**Date**: 2026-04-19 (Round 20)
**Hardware**: MacBook Pro M1 Pro, 16 GB unified RAM, 8P+2E cores
**Engine**: quant.cpp commit c0d8d92 (Round 19, Q5_K SHL extraction landed)
**Env**: `TQ_NO_MLOCK=1`, `-T 0` (greedy), warm steady-state (2nd invocation, same prompt)

## Summary Matrix

| Quant | bpw | File | RSS | Decode (warm) | Quality |
|---|---:|---:|---:|---:|---|
| IQ2_XXS | 2.06 | 10.0 GB | ~6.5 GB | 16.1 t/s† | JSON 깨짐 |
| IQ3_XXS | 3.06 | 12.3 GB | ~6.5 GB | 14.6 t/s† | 양호 |
| Q3_K_S | 3.50 | 14.3 GB | 5.24 GB† | 14.3 t/s† | 양호 |
| **IQ4_XS** | 4.25 | 16.5 GB | **7.25 GB** | **10.6 t/s** | 우수 |
| **Q5_K_M** | 5.50 | 24.6 GB | **9.65 GB** | **7.9 t/s** | **탁월** |

† 이전 세션 측정치 (파일 삭제로 재측정 불가). **굵은 값은 Round 20 실측**.

## Round 16-17 Q5_K_M 돌파 의의

- Round 12 selective MADV 정책으로 **26 GB 파일이 16 GB RAM에 fit**.
  selective WILLNEED이 non-expert (attn/norm/embed = 2.5 GB)만 force-load,
  routed experts는 OS 페이지 캐시 자연 관리 → MoE 희소성 (K=8/N=256) 효과.
- Round 17 NEON qh SHL 추출 최적화로 Q5_K 커널 +40% (cold 1.5 → 2.1 t/s).
- **Warm steady-state 7.9 t/s**는 실용 대화 범위. Q5 양자화 품질 + 16 GB Mac
  결합은 quant.cpp 엔진 차원 최초.

## Round 18-19 실패 교훈

- **2-row parallel on Q5_K**: 레지스터 프레셔로 -14%. Q5_K는 Q4_K보다 per-row
  state가 많아 NEON 32×128-bit 한도 초과 → 스필.
- **Per-dispatch madvise(WILLNEED)**: 3× 느려짐. macOS 통합 메모리에서 madvise
  per-call은 VM 서브시스템 컨텐션 + 이미 레지던트 페이지에도 동기 검증 → flash-moe
  F_RDADVISE -73% 재현. **madvise는 load time 전용**.

## 추천 사용 가이드

| 용도 | 추천 quant | 이유 |
|---|---|---|
| 빠른 대화 (Chat) | Q3_K_S | 14 t/s, 16 GB 여유, 품질 양호 |
| 품질 우선 (One-shot) | **Q5_K_M** | 7.9 t/s, JSON/code 정확 |
| 최소 메모리 (≤12 GB Mac) | IQ3_XXS | 12.3 GB 파일, 6.5 GB RSS |
| 성능/품질 균형 | IQ4_XS | 10.6 t/s, 품질 우수 |

## 비교: llama.cpp

동일 16 GB Mac에서 llama.cpp CPU 모드로 Qwen3.6-35B-A3B 실행 시 5.1 t/s
(이전 세션 측정, Q3_K_S 기준). quant.cpp는 **2.8-3.2× 빠름** (IQ2/IQ3/Q3 tier),
Q5_K_M은 llama.cpp가 16 GB Mac에서 로드 자체 불가.
