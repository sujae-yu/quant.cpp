# Getting Started

TurboQuant.cpp를 빌드하고, 직접 체험하고, 프로젝트에 통합하는 가이드입니다.

---

## 1. 빌드

### 요구사항

- C11 / C++17 컴파일러 (GCC 9+, Clang 12+, MSVC 2019+)
- CMake 3.20+
- Python 3.8+ (Python 바인딩/CLI 사용 시)

### 빌드

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp
cd TurboQuant.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DTQ_BUILD_TESTS=ON \
      -DTQ_BUILD_BENCH=ON

cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

### 테스트

```bash
ctest --test-dir build --output-on-failure
# 17 C++ 테스트 스위트, 100% 통과
```

---

## 2. 30초 체험

빌드 후 바로 실행 가능한 데모 3개:

```bash
# A/B 비교: FP16 vs 양자화 타입별 품질 비교
./build/ab_test

# 실제 모델별 메모리 절약 (Llama, Qwen, Phi)
./build/demo_real_model

# 속도: 정수 Attention vs FP32
./build/speed_int_vs_float
```

---

## 3. CLI 도구 (`tq`)

### 설치

```bash
# Python 바인딩 설치 (CLI에 필요)
pip install -e bindings/python

# CLI 실행
python3 tools/tq info
```

### 명령어

```bash
# 양자화 타입 정보 (★ 추천 표시)
python3 tools/tq info

# 모델별 메모리 절약 계산
python3 tools/tq +memory llama-3.2-3b 65536
python3 tools/tq +memory qwen3.5-0.8b 131072

# 성능 벤치마크
python3 tools/tq bench
python3 tools/tq bench --seq-len 2048 --head-dim 256

# AI 에이전트용 JSON 출력
python3 tools/tq info --json
python3 tools/tq +memory llama-3.2-3b 65536 --json

# A/B 비교 (빌드 필요)
python3 tools/tq +compare
```

### Qwen3.5-0.8B 대화 (실제 모델 추론)

```bash
# torch + transformers 설치 (최초 1회)
python3 -m venv /tmp/tq_venv
source /tmp/tq_venv/bin/activate
pip install torch transformers numpy accelerate

# 대화 모드
python3 tools/tq_chat.py

# 단일 질문
python3 tools/tq_chat.py "What is KV cache quantization?"

# 벤치마크 모드
python3 tools/tq_chat.py --benchmark
```

---

## 4. Python API

```bash
pip install -e bindings/python
```

```python
from turboquant import TurboQuant
import numpy as np

tq = TurboQuant("cpu")

# 양자화
keys = np.random.randn(512, 128).astype(np.float32) * 0.15
compressed = tq.quantize_keys(keys, TurboQuant.UNIFORM_4B)
print(f"{keys.nbytes:,} → {len(compressed):,} bytes ({keys.nbytes/len(compressed):.1f}x)")

# 역양자화
decompressed = tq.dequantize_keys(compressed, 512, 128, TurboQuant.UNIFORM_4B)
mse = np.mean((keys - decompressed) ** 2)
print(f"MSE: {mse:.6f}")

# Attention
query = np.random.randn(128).astype(np.float32)
scores = tq.attention(query, compressed, 512, 128, TurboQuant.UNIFORM_4B)

# 타입 비교
for qtype in [TurboQuant.UNIFORM_4B, TurboQuant.MIXED_4B8, TurboQuant.UNIFORM_2B]:
    name = tq.type_name(qtype)
    bpe = tq.type_bpe(qtype)
    print(f"  {name}: {bpe:.1f} bits, {32/bpe:.1f}x compression")
```

---

## 5. C API

```c
#include "turboquant/turboquant.h"

// 초기화
tq_context_t* ctx;
tq_init(&ctx, TQ_BACKEND_CPU);

// 양자화
size_t size = tq_quantize_keys_size(seq_len, head_dim, TQ_TYPE_UNIFORM_4B);
void* buf = malloc(size);
tq_quantize_keys(ctx, keys, seq_len, head_dim, TQ_TYPE_UNIFORM_4B, buf, size);

// Attention (FP32보다 2.9-4.8x 빠름)
float scores[seq_len];
tq_attention(ctx, query, buf, seq_len, head_dim, TQ_TYPE_UNIFORM_4B, scores);

// K/V 비대칭 양자화 (Key 4bit + Value 2bit = 9.8x 압축)
tq_quantize_kv(ctx, keys, values, n, head_dim,
               TQ_TYPE_UNIFORM_4B, TQ_TYPE_UNIFORM_2B,
               key_out, key_size, val_out, val_size);

// RHT 전처리 (MSE 1.8-3.9x 추가 개선)
tq_quantize_keys_rht(ctx, keys, n, head_dim,
                      TQ_TYPE_UNIFORM_4B, seed, out, size);

// 정리
free(buf);
tq_free(ctx);
```

### CMake 통합

```cmake
add_subdirectory(TurboQuant.cpp)
target_link_libraries(my_app turboquant)
```

---

## 6. llama.cpp 통합

```bash
# 상세 가이드: integrations/llamacpp/README.md

# CMakeLists.txt에 추가
add_subdirectory(path/to/TurboQuant.cpp turboquant)
target_link_libraries(llama PRIVATE turboquant)
```

```cpp
#include "integrations/llamacpp/tq_kv_cache.cpp"

// 초기화 시 타입 등록
tq_ggml_register_types();

// CLI 파서 (21개 별칭 지원)
tq_type type = tq_parse_kv_cache_type("turbo3");  // or "tq-uniform-4b", "uniform_4b" ...

// 사용 가능한 타입 목록
tq_print_kv_cache_types();
```

---

## 7. 실제 모델 검증

Qwen3.5-0.8B의 실제 KV 캐시로 검증:

```bash
# 실제 모델에서 KV 캐시 추출
source /tmp/tq_venv/bin/activate
python3 tests/reference/dump_qwen35_kv.py

# 양자화 품질 검증
./build/qwen35_validation
```

검증 결과: [docs/qwen35_validation_results.md](qwen35_validation_results.md)

---

## 8. 벤치마크 실행

```bash
# 품질 (MSE, cosine, cross-platform)
./build/tq_quality

# 성능 (throughput, compression, SIMD)
./build/tq_bench

# 정수 vs FP32 Attention 속도
./build/speed_int_vs_float

# 개별 커널 성능
./build/bench_kernel

# 메모리 사용량
./build/bench_memory
```

---

## 9. 프로젝트 구조

```
include/turboquant/     Public C API
src/core/               알고리즘 (polar, qjl, turbo, uniform, mixed, rht)
src/cache/              페이지 캐시 + 점진적 압축
src/backend/cpu/        CPU 커널 (generic, NEON, AVX2)
src/backend/cuda/       CUDA 커널
src/backend/metal/      Metal 셰이더
tests/                  Google Test (17 스위트)
bench/                  벤치마크
tools/                  CLI 도구 (tq, tq_chat)
examples/               예제 (C, C++, Python)
integrations/           llama.cpp, vLLM 통합
bindings/python/        Python ctypes 바인딩
spec/                   포맷 사양 + 테스트 벡터
docs/                   문서
```

---

## 10. 추천 양자화 전략

실제 Qwen3.5-0.8B A/B 테스트 기반:

| 상황 | 추천 | 품질 | 압축 |
|------|------|------|------|
| **프로덕션 기본** | `uniform_4b` | A+ (0.994) | 7.5x |
| **최적 가성비** | K4V2 (key=4b, val=2b) | ~0.97 | 9.8x |
| **아웃라이어 심한 모델** | `mixed_4b8` | A+ (0.994) | 6.4x |
| **극한 압축** | `uniform_2b` | A (0.953) | 14.2x |
| **품질 극대화** | RHT + `uniform_4b` | A++ | 7.5x |

```bash
# CLI로 확인
python3 tools/tq info
```
