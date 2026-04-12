# Phase A-2: 대규모 문서 스트레스 테스트 — M3 실행 가이드

> **목적**: RLV 10/10 결과가 작은 문서(35K chars)에 대한 overfitting인지 검증
> **방법**: 1.3MB 문서 (37× 규모)에서 20개 질문 실행
> **기대 시간**: ~80분 (Phi-3.5 Q8_0, 20질문 × ~4분)

---

## 배경

현재 RLV 성과:
- **Acme** (1.9K chars, 5 sections): 7/7 ✅
- **Wikitext-small** (35K chars, 3 articles, 23 chunks): 19/20 ✅

**우려**: 문서가 작아서 BM25 IDF가 쉽게 작동하고, 토픽이 3개뿐이라 locator가 쉽게 구별.

**검증**: 1.3MB wikitext2_test.txt (63개 기사, 2754 chunks, ~310K tokens)에서 동일 파이프라인 테스트.

---

## 실행 순서

### 1. 코드 최신화
```bash
cd ~/dev/quantcpp_test  # 또는 프로젝트 경로
git pull origin main
```

커밋 `a922806` (test: add 1.3MB large-doc stress test) 포함 확인.

### 2. 모델 확인
```bash
ls -lh models/Phi-3.5-mini-instruct-Q8_0.gguf
# 3.8GB — 없으면 다운로드:
# curl -L -o models/Phi-3.5-mini-instruct-Q8_0.gguf \
#   "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q8_0.gguf"
```

### 3. unified 서버 빌드
```bash
cc -O2 -o build_metal/quant-server-unified tools/quant_server_unified.c -lm -lpthread
```

### 4. 서버 동작 확인 (30초)
```bash
./build_metal/quant-server-unified models/Phi-3.5-mini-instruct-Q8_0.gguf -p 8421 -j 8 &
sleep 5
curl -s http://127.0.0.1:8421/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10,"temperature":0}' | python3 -m json.tool
pkill -f quant-server-unified
```

기대 출력: `"content": "4"` (또는 유사한 정확한 답변)

### 5. 기존 테스트 regression 확인 (선택, ~15분)
```bash
# 기존 19/20 결과 유지되는지 확인
python3 bench/rlv/eval/eval_wikitext.py --systems rlv
```

### 6. 대규모 문서 스트레스 테스트 실행
```bash
# 전체 20문제 (~80분 예상)
python3 bench/rlv/eval/eval_wikitext_large.py 2>&1 | tee /tmp/eval_large.log
```

시간이 부족하면 5개만 먼저:
```bash
# 빠른 확인 (5문제, ~20분)
for i in 1 4 8 12 16; do
  python3 bench/rlv/eval/eval_wikitext_large.py --only $i
done
```

### 7. 실패 질문 디버깅
```bash
# 실패한 질문 번호로 verbose 실행
python3 bench/rlv/eval/eval_wikitext_large.py --only <번호> --verbose
```

verbose 출력에서 확인할 것:
- `[locator]` — 올바른 chunk를 골랐는지 (chunk_id와 head_text)
- `[lookup]` — 올바른 문장을 선택했는지
- `[verifier]` — CONFIDENT/UNSURE/CONTRADICTED 판정 근거
- `[researcher]` — 재시도 횟수와 결과

---

## 질문 분포

| 카테고리 | 질문 수 | 기사 |
|---|---|---|
| 기존 회귀 체크 | 3 | Boulter, Du Fu, Kiss You |
| 군사 역사 | 4 | Ise급 전함, Naktong 전투, 장갑함 |
| 스포츠 | 4 | Rifenburg, Kershaw, Ben Amos |
| 과학/기상 | 4 | Dvorak 기법, 1933 허리케인, 양서류 |
| 문학/예술 | 3 | 이미지즘, Little Gidding, Portage |
| 지리/역사 | 2 | NY Route 31B, Osbert de Bayeux |

난이도: single-hop 16개, multi-hop 4개

---

## 성공 기준

| 등급 | 정확도 | 의미 |
|---|---|---|
| **A (돌파)** | ≥ 17/20 (85%) | 대규모 문서에서도 RLV가 작동 — overfitting 아님 |
| **B (양호)** | 14-16/20 (70-80%) | 기본은 작동하지만 locator 개선 필요 |
| **C (미달)** | < 14/20 (< 70%) | 확장성 문제 심각 — 아키텍처 재설계 필요 |

### 비교 포인트
- 기존 소문서: 19/20 = 95%
- 대문서 목표: ≥ 85% (10% 이내 하락은 허용 — 37× 규모 증가 감안)

---

## 예상 실패 패턴

1. **Locator confusion**: 유사 토픽 기사 간 chunk 혼동 (예: 군사 기사들 간 "battle" 키워드 겹침)
2. **BM25 IDF 약화**: 2754 chunks에서 일반 단어의 IDF가 낮아져 구별력 저하
3. **Chunk 세분화**: paragraph chunker가 짧은 단락마다 chunk 생성 → 맥락 부족
4. **LLM 호출 타임아웃**: 긴 문서의 gist 생성 + 다수 retry로 총 시간 초과

---

## 결과 커밋
```bash
# 결과를 phase3 문서에 기록하고 커밋
git add bench/rlv/eval/eval_wikitext_large.py docs/phase3_rlv_challenge.md
git commit -m "phase 3 phase-a2: large doc stress test results (1.3MB, 2754 chunks)"
git push origin main
```

---

## Dry-run 예비 결과 (이 머신에서 LLM 없이)

Locator keyword-only 정확도: **7/8 (87.5%)** on 2754-chunk doc.

단 1개 실패: "Clayton Kershaw drafted year" — "2006"이 다른 기사에도 등장 가능.
BM25+RRF 하이브리드로 LLM 실행 시 개선 예상.
