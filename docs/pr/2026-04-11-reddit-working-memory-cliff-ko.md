# Reddit r/LocalLLaMA — The Working Memory Cliff (한글)

**제목:** `[Research] Llama-3.2-1B/3B-Q4 모델의 working memory cliff를 측정했습니다 — 두 모델 모두 명목 128K context window의 1% 미만에서 retrieval 0%로 떨어집니다`

**Flair:** `Research`

---

## 본문

지난 달 [chunk-RAG가 7/7 환각, 6.4× KV 압축 + 전체 문서 로딩이 7/7 정답](https://github.com/quantumaikr/quant.cpp/blob/main/docs/beyond-rag-manifesto.md) 결과를 Llama-3.2-3B에서 측정해 공유했습니다. 합성 600토큰 문서였고, 일부 댓글에서 "600토큰은 stress test가 아니다"라는 정당한 지적이 있었습니다.

그래서 ctx 256–2048 범위에서 204 NIAH trial을 돌려 실제로 모델이 어디서 망가지는지 측정했습니다. 결과는 예상보다 훨씬 가파릅니다.

### Cliff

**Llama-3.2-1B-Instruct-Q8_0** (graded cliff):

| ctx | fp32 KV | 6.4× 압축 |
|---:|:-:|:-:|
| 256 | 89% | 89% |
| **512** | **100%** | **100%** |
| 1024 | 44% | 22% |
| 1536 | 0% | 0% |
| 2048 | 0% | 0% |

**Llama-3.2-3B-Instruct-Q4** (default 로더; **step function**):

| ctx | fp32 KV | 6.4× 압축 |
|---:|:-:|:-:|
| 512 | 100% | 100% |
| **1024** | **100%** | **100%** |
| **1280** | **0%** | **0%** |
| 1536–2048 | 0% | 0% |

**1024 → 1280 사이는 256토큰입니다.** 18/18 → 0/18. degradation interval이 **없습니다**. 모델이 chat-template instruction을 완벽히 따르던 상태에서 완전히 무시하는 상태로 25% context length 증가 안에서 전환됩니다.

두 모델 모두 명목 128K context window의 **1% 미만**에서 effective working memory에 도달합니다 (1B Q8 ≈ 0.4%, 3B Q4 ≈ 0.78%).

### KV 압축은 cliff와 직교 (orthogonal)

같은 grid에서 6.4× `turbo_kv_4b -v q4 --k-window 128`을 FP32 KV baseline과 비교했습니다. **20개 (model × ctx × method) cell 중 18개가 baseline과 압축 사이에 bit-for-bit 일치**합니다. 1개 disagreement도 1B cliff cell에서 둘 다 noise floor에 있는 경우입니다.

양자화 confound를 배제하기 위해 FP32 *weights* (`TQ_NO_Q4=1`)로도 cliff transition을 재측정했습니다. 동일한 cliff, 동일한 위치: ctx=1024에서 100%, ctx=1280에서 0%, FP32 weights에서도 동일. **Cliff는 model property이지 KV cache property가 아니고 weight quantization artifact도 아닙니다.**

### 실패 모드는 "모르겠다"가 아닙니다

Cliff 위에서 모델은 세 가지 중 하나를 출력합니다. 처음 두 개 (위키텍스트 이어쓰기, 헤더 echo)는 평범합니다. 세 번째가 결정적입니다.

**Synthesised hallucination, 1B fp32 ctx=1024:**

> *"In 2023 Boulter was hired as the chief financial officer..."*

Haystack은 영국 배우 Robert Boulter의 위키피디아 문서입니다. Needle은 "The chief financial officer of Northwind Logistics is Sarah Chen, hired in 2023." 모델이 **두 사실을 융합** — needle의 "2023 hire"를 Boulter의 biography에 접목한 일관된 가짜 문장을 만들어냈습니다.

이건 vector RAG가 retrieval miss에서 만들어내는 **silent hallucination 실패 모드와 동일**합니다 — 그 모드를 *제거*해야 할 regime에서 발생.

### 정직한 scope

- 2개 모델 (1B, 3B), 3개 needle, 1개 언어 (영어), 1개 도메인 (위키피디아 biography), 총 204 trial
- 8B (Llama-3.1-Q4_K_M)도 시도했으나 Metal에서 inference 1회당 ~12분이라 full grid는 비현실적. v2 작업
- 1B cliff cell의 fp32와 압축 사이 22 pp 차이는 n=9에서 통계적으로 유의하지 않음 — seed sweep 시도 중 CLI 버그 발견해 같은 라운드에서 fix했고, 적절한 stochastic robustness 확인은 v2로
- 6개 prompt format을 시도했고 가장 permissive한 것을 사용. Format sensitivity는 별도로 측정해야 할 ceiling

다른 모델/prompt/언어에서 cliff를 falsify할 수 있다면 데이터 환영합니다.

### 재현

```bash
git clone https://github.com/quantumaikr/quant.cpp
cd quant.cpp
cmake -B build_metal -DTQ_BUILD_METAL=ON && cmake --build build_metal -j8

# 1B Q8 sweep (M-series에서 ~30분)
MODEL=models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  NIAH_CONTEXTS="256 512 1024 1536 2048" \
  bash bench/niah_test.sh
```

### 링크

- **Tech report (arXiv-style draft)**: repo의 `docs/paper/working-memory-cliff.md`
- **Master table**: `bench/results/niah/master_table.md`
- **Raw CSV + per-run CLI logs**: `bench/results/niah/`
- **GitHub**: https://github.com/quantumaikr/quant.cpp

### "Beyond RAG"에 대한 재해석

정직한 재해석: **Beyond RAG는 모델의 *effective* working memory에 들어가는 문서에 대해서만 동작하며, 그 크기는 명목 context window의 100분의 1에서 1000분의 1입니다.** 메모리 측면에서는 16GB Mac에 128K context가 9.5GB로 들어갑니다. Retrieval 측면에서는 3B Q4 모델이 ~1024토큰부터 instruction을 따르지 않습니다 (압축 여부와 무관). "Long-context replaces RAG"를 주장하는 edge-device 벤더는 메모리 할당 숫자와 함께 effective working memory 측정값도 함께 발표해야 합니다 — 격차가 거대합니다.

llama.cpp / MLC / ollama default에서 1B–3B 스케일 NIAH 측정 데이터를 가진 분 계신다면 비교해 보고 싶습니다. 이것이 quant.cpp 로더의 artifact인지, 이 regime의 보편적 현상인지 확인하고 싶습니다.
