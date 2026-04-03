# quant.cpp — PRD v0.9: llama.cpp 속도 돌파

**Target**: 현재 15 tok/s → **40+ tok/s** (llama.cpp 수준)

## 병목 분석

```
레이어 matmul: 194 ms (94.3%) ← 이것을 4x 빨리 만들어야 함
출력 projection: 12 ms (5.7%)
나머지: 0 ms
```

24개 레이어 × 레이어당 ~8ms = 194ms. 목표: 레이어당 2ms = 48ms 총.

## 최적화 전략 (임팩트 순)

### 1. Q4 가중치 (예상 2x)
Q8 → Q4: 데이터 2x 작음 → 메모리 대역폭 2x 절약
llama.cpp Q4_K_M 패턴: int4 × int8 dot product

### 2. matmul 타일링 (예상 1.5x)
현재: 행 단위 처리 (cache miss 빈번)
개선: 타일 크기 최적화 (L1=128KB에 맞춤)

### 3. 가중치 레이아웃 전치 (예상 1.3x)
현재: row-major [n, d] → 열 방향 접근 시 cache miss
개선: 가중치를 [d, n] 전치 저장 → 순차 접근

### 4. NEON matmul 극한 최적화 (예상 1.2x)
현재: 8-wide FMA (2 accumulators)
개선: 16-wide (4 accumulators), 프리페치, 언롤링

### 목표 달성 경로
```
현재:           15 tok/s (Q8, 206ms/token)
+ Q4 가중치:   ~30 tok/s (2x, 103ms/token)
+ 타일링:      ~40 tok/s (1.3x, 79ms/token)
+ 레이아웃:    ~45 tok/s (1.1x, 72ms/token)
```
