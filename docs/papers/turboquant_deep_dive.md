# quant.cpp Deep Dive: Core Concept & Implementation Evolution

## 1. 논문이 풀려는 근본적 문제

KV 캐시에서 우리가 실제로 하는 연산은 **내적(inner product)**이다:

```
attention_score = <query, key>
```

핵심 통찰: **MSE를 최소화하는 양자화기(quantizer)는 내적 추정에 편향(bias)을 만든다.**

1-bit 양자화로 설명하면:
- MSE-optimal 양자화기는 x를 sign(x)로 변환하고, 역양자화 시 `sqrt(2/(pi*d)) * sign(x)`로 복원
- 이때 `E[<y, Q^-1(Q(x))>] = (2/pi) * <y, x>` — 원래 내적의 **2/pi ≈ 0.637배**만 복원됨
- 이 편향은 attention softmax에서 **상대적 순위를 보존하지만, 확률 분포를 왜곡**함

논문의 해법: **MSE 양자화기 + QJL 잔차 보정**으로 편향을 완전히 제거.

## 2. 현재 구현의 3가지 구조적 문제

### 문제 1: QJL의 O(d^2) 비용

현재 `compute_qjl_signs`와 attention의 QJL 보정은 모두 **O(d^2)**:

```c
// compute_qjl_signs: O(dim * sketch_dim) = O(d^2)
for (int s = 0; s < n_sketch; s++) {        // sketch_dim = d
    for (int d = 0; d < dim; d++) {          // dim = d
        proj += residual[d] * tkv_qjl_random_entry(d, s);  // random entry 매번 계산
    }
}
```

dim=128에서: 128 * 128 = **16,384 연산 per key per token**.
dim=256(Gemma 4B head_dim)에서: 256 * 256 = **65,536 연산 per key per token**.

Attention에서는 이것이 **매 key마다** 반복되므로, seq_len=1000이면 **65백만 연산**.

**논문의 해법**: S 행렬을 Rademacher (+1/-1)로 하면, `S . x`는 **부분합의 랜덤 부호 합산**이 됨. 이것은 **Structured Random Matrix** (예: SRHT = subsampled RHT)로 O(d log d)로 가속 가능. 하지만 현재 구현은 이를 활용하지 않음.

### 문제 2: Attention에서 매 key마다 RHT를 반복

```c
for (int seq = 0; seq < seq_len; seq++) {
    // 매 key마다:
    tq_rht_inverse(rotated, dim, seed);     // O(d log d) per key
    tq_rht_transform(q_rot, dim, seed);     // O(d log d) per key (query rotation)
    // QJL correction: O(d^2) per key
}
```

모든 key가 **같은 seed**를 사용하므로 (TKV_DEFAULT_SEED = 0x12345678), query의 RHT는 **한 번만 계산하면 됨**. 현재는 루프 안에서 매번 반복.

또한, RHT는 직교 변환이므로 `<q, Pi^T * k_rot> = <Pi * q, k_rot>`. **query를 한 번 회전하면, MSE 내적을 회전 공간에서 직접 계산 가능**:

```
mse_dot = <q_rot, k_mse_rot>  (RHT inverse 불필요!)
```

### 문제 3: 논문의 핵심 — RHT 공간에서 모든 연산 수행

논문 Algorithm 2의 DeQuant_prod:
```
x_tilde_mse ← DeQuant_mse(idx)           // 회전 공간에서 복원
x_tilde_qjl ← sqrt(pi/2)/d * gamma * S^T . qjl  // 회전 공간에서 QJL 복원
output: x_tilde_mse + x_tilde_qjl         // 합산 후 역회전
```

**핵심**: 내적 `<y, x_tilde>`를 계산할 때, 역회전을 하지 않아도 됨:
```
<y, Pi^T * (mse + qjl)> = <Pi * y, mse + qjl> = <y_rot, mse_rot> + <y_rot, qjl_rot>
```

즉, **query를 한 번 회전하면, 모든 key와의 내적을 회전 공간에서 직접 계산** 가능. 역 RHT가 불필요.

## 3. 최적화된 아키텍처

```
=== Quantize (per key, 1회) ===
key → normalize → RHT → codebook_quantize(b-1 bits)
                     → residual = rotated - dequant
                     → qjl_signs = sign(S . residual)
Store: [indices, qjl_signs, norm, r_norm]

=== Attention (per query, 매 토큰) ===
query → RHT (1회) → q_rot                    // O(d log d), 한 번만
q_rot → S_project → q_sketch                  // O(d log d) if SRHT, 한 번만

For each key:
  k_mse_rot = codebook_dequant(indices)        // O(d), 테이블 룩업
  mse_dot = <q_rot, k_mse_rot>                 // O(d), 벡터 내적
  qjl_dot = <q_sketch, signs>                  // O(d/8), Hamming-like XOR+popcount!
  score = norm * (mse_dot + r_norm * qjl_scale * qjl_dot)
```

### 성능 분석 (dim=128):

| 연산 | 현재 | 최적화 후 | 개선 |
|------|------|----------|------|
| RHT per key | 128*7 = 896 | **0** (query만 1회) | ∞ |
| MSE dequant | O(128) | O(128) | 동일 |
| MSE 내적 | O(128) | O(128) | 동일 |
| QJL projection | O(128^2) = 16384 | **O(128/8) = 16** (XOR+popcount) | **1000x** |
| **총 per-key** | **~17,500** | **~270** | **~65x** |

## 4. QJL을 Hamming Distance로 변환하는 핵심 트릭

논문의 QJL:
```
<y, Q_qjl^{-1}(Q_qjl(x))> = sqrt(pi/2)/d * sum_s(y_proj_s * sign_s)
```

여기서 `y_proj_s`는 query의 s번째 random projection. `sign_s`는 key의 s번째 부호.

**트릭**: query도 sign hash하면:
```
q_sign_s = sign(S_s . q)
k_sign_s = sign(S_s . key_residual)

sum_s(q_proj_s * k_sign_s) ≈ ||q_proj|| * (2 * agree(q_signs, k_signs)/d - 1)
```

여기서 `agree(a, b)` = d - hamming_distance(a, b).

**Hamming distance는 XOR + popcount로 O(d/64)에 계산 가능!**
128비트 = 2 uint64_t → 2번의 XOR + 2번의 popcount.

이것이 논문에서 말하는 **"accelerator-friendly"**의 핵심 의미.

## 5. 구현 액션 플랜

### Phase 1: Attention 경로 최적화 (최대 효과)

1. **Query를 한 번만 RHT 변환** — 루프 밖으로 이동
2. **회전 공간에서 직접 내적** — RHT inverse 제거
3. **QJL sketch를 query에 대해 1회 pre-compute** — 루프 밖

### Phase 2: QJL을 Hamming Distance로 변환

1. Query도 sign hash: `q_signs = sign(S . q_rot)`
2. Key-query 비교: `hamming = popcount(q_signs XOR k_signs)`
3. Correction: `r_norm * qjl_scale * (2*agree/d - 1) * ||q_proj||`
4. ARM NEON `vcntq_u8` popcount 사용

### Phase 3: NEON 벡터화

1. Codebook dequant: 인덱스 → centroid lookup을 NEON gather로
2. 내적: `vfmaq_f32` 사용 (이미 엔진에 있는 패턴)
3. Hamming: `veorq_u8` + `vcntq_u8` + `vpaddlq` 체인

## 6. 예상 성능 개선

| 시나리오 | 현재 turbo_kv | 최적화 후 | vs uniform_4b |
|---------|-------------|----------|---------------|
| Qwen3.5 (dim=128, seq=100) | ~2ms/attention | ~0.05ms | uniform과 동등 |
| Gemma 4B (dim=256, seq=1000) | ~800ms/attention | ~5ms | uniform보다 빠름 (캐시 효율) |

근본적으로, 최적화된 quant.cpp KV attention은 **uniform Q4 integer attention과 거의 같은 속도**로 동작하면서, **더 적은 비트(3bit vs 4bit)로 더 좋은 품질**을 달성할 수 있음.
