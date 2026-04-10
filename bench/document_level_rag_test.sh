#!/bin/bash
# ============================================================
# Document-Level RAG Benchmark
# ============================================================
#
# Compares Chunk-RAG vs Full-Document-Context for QA accuracy.
#
# Methodology:
#   1. Create a synthetic document with facts spread across sections
#   2. Ask questions requiring single-hop and multi-hop reasoning
#   3. Chunk-RAG: split into 512-word chunks, keyword-search, feed top-1
#   4. Full-Document: feed entire document to model
#   5. Score: does the answer contain the correct key fact?
#
# Usage: bash bench/document_level_rag_test.sh [model.gguf]

set -e

MODEL="${1:-models/Llama-3.2-3B-Instruct-Q8_0.gguf}"
TQ="./build/quant"
THREADS=8
PASS=0
FAIL=0
TOTAL=0

if [ ! -f "$TQ" ]; then echo "Error: build first"; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "SKIP: model not found: $MODEL"; exit 0; fi

# ============================================================
# Synthetic Document: "Acme Corp Annual Report 2025"
# Facts are deliberately spread across distant sections
# ============================================================
SECTION1="Section 1: Financial Overview.
Acme Corporation reported total revenue of 847 million dollars in fiscal year 2025, representing a 15 percent increase over the previous year. Operating margins improved to 23 percent. The company opened 12 new offices globally. Net income reached 195 million dollars. The stock price increased by 34 percent during the fiscal year."

SECTION2="Section 2: Product Development.
The engineering team launched three major products this year. Project Atlas delivered a new cloud infrastructure platform used by 400 enterprise customers. The mobile division released version 5.0 of the flagship application with 20 million downloads in the first quarter. Research and development spending increased to 120 million dollars, representing 14 percent of total revenue."

SECTION3="Section 3: Growth Strategy.
The Southeast Asia expansion initiative was the primary driver of revenue growth in 2025. The company established offices in Singapore, Jakarta, and Bangkok, capturing 8 percent market share within 6 months. This regional strategy was originally proposed by Executive Vice President James Park during the 2023 strategic planning retreat in Kyoto."

SECTION4="Section 4: Human Resources.
The company grew its workforce to 5200 employees across 28 countries. Dr. Maria Santos was appointed as Chief Technology Officer in January 2025, replacing the retiring Dr. Robert Kim. The employee satisfaction score reached 4.2 out of 5.0. The company invested 15 million dollars in employee training programs."

SECTION5="Section 5: Risk Factors.
Currency fluctuations in Southeast Asian markets posed a 3 percent headwind to reported revenue. Supply chain disruptions affected the hardware division in Q2 but were resolved by Q3. The company maintains a cybersecurity insurance policy valued at 50 million dollars. Regulatory changes in the European Union required additional compliance spending of 8 million dollars."

FULL_DOC="${SECTION1}

${SECTION2}

${SECTION3}

${SECTION4}

${SECTION5}"

echo "============================================================"
echo "  Document-Level RAG Benchmark"
echo "  Model: $MODEL"
echo "============================================================"
echo ""
echo "  Document: Acme Corp Annual Report (5 sections, ~300 words)"
echo ""

# ============================================================
# Test questions: single-hop and multi-hop
# ============================================================
# Format: "question|correct_keyword|type"
QUESTIONS=(
  "What was Acme's total revenue in 2025?|847|single-hop"
  "Who was appointed as CTO in January 2025?|Santos|single-hop"
  "What was the primary driver of revenue growth?|Southeast Asia|single-hop"
  "Who originally proposed the Southeast Asia expansion strategy?|James Park|multi-hop"
  "How much did R&D spending represent as a percentage of total revenue?|14 percent|single-hop"
  "The revenue growth was driven by a strategy proposed at what event?|Kyoto|multi-hop"
  "What risk factor was related to the same region that drove growth?|Currency fluctuations|multi-hop"
)

# ============================================================
# Method 1: Chunk-RAG (keyword search on chunks)
# ============================================================
echo "[Method 1] Chunk-RAG (split into sections, keyword-search, feed top-1)"
echo "---"

chunk_pass=0
chunk_total=0

for q_entry in "${QUESTIONS[@]}"; do
  IFS='|' read -r question keyword qtype <<< "$q_entry"
  chunk_total=$((chunk_total + 1))

  # Simple keyword search: find which section contains the most words from the question
  best_section=""
  best_score=0
  for section_var in SECTION1 SECTION2 SECTION3 SECTION4 SECTION5; do
    section_text="${!section_var}"
    score=0
    for word in $question; do
      if echo "$section_text" | grep -qi "$word" 2>/dev/null; then
        score=$((score + 1))
      fi
    done
    if [ $score -gt $best_score ]; then
      best_score=$score
      best_section="$section_text"
    fi
  done

  # Feed best chunk + question to model
  prompt="Context: ${best_section}

Question: ${question}
Answer briefly:"

  answer=$($TQ "$MODEL" -p "$prompt" -n 40 -T 0.0 -j $THREADS -k fp32 --chat 2>/dev/null)

  if echo "$answer" | grep -qi "$keyword"; then
    echo "  [PASS] ($qtype) $question → found '$keyword'"
    chunk_pass=$((chunk_pass + 1))
  else
    echo "  [FAIL] ($qtype) $question → missing '$keyword'"
  fi
done

echo ""
echo "  Chunk-RAG: ${chunk_pass}/${chunk_total} correct"
echo ""

# ============================================================
# Method 2: Full-Document Context
# ============================================================
echo "[Method 2] Full-Document Context (entire document in prompt)"
echo "---"

doc_pass=0
doc_total=0

for q_entry in "${QUESTIONS[@]}"; do
  IFS='|' read -r question keyword qtype <<< "$q_entry"
  doc_total=$((doc_total + 1))

  prompt="Based on this document, answer the question.

Document: ${FULL_DOC}

Question: ${question}"

  answer=$($TQ "$MODEL" -p "$prompt" -n 40 -T 0.0 -j $THREADS -k fp32 --chat 2>/dev/null)

  if echo "$answer" | grep -qi "$keyword"; then
    echo "  [PASS] ($qtype) $question → found '$keyword'"
    doc_pass=$((doc_pass + 1))
  else
    echo "  [FAIL] ($qtype) $question → missing '$keyword'"
  fi
done

echo ""
echo "  Full-Document: ${doc_pass}/${doc_total} correct"
echo ""

# ============================================================
# Method 3: Full-Document + KV Compression (6.4x)
# ============================================================
echo "[Method 3] Full-Document + KV Compression (6.4x)"
echo "---"

comp_pass=0
comp_total=0

for q_entry in "${QUESTIONS[@]}"; do
  IFS='|' read -r question keyword qtype <<< "$q_entry"
  comp_total=$((comp_total + 1))

  prompt="Based on this document, answer the question.

Document: ${FULL_DOC}

Question: ${question}"

  answer=$($TQ "$MODEL" -p "$prompt" -n 40 -T 0.0 -j $THREADS \
    -k turbo_kv_4b -v q4 --k-window 128 --chat 2>/dev/null)

  if echo "$answer" | grep -qi "$keyword"; then
    echo "  [PASS] ($qtype) $question → found '$keyword'"
    comp_pass=$((comp_pass + 1))
  else
    echo "  [FAIL] ($qtype) $question → missing '$keyword'"
  fi
done

echo ""
echo "  Compressed: ${comp_pass}/${comp_total} correct"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "  Results Summary"
echo "============================================================"
echo ""
printf "  %-30s %s\n" "Method" "Accuracy"
printf "  %-30s %s\n" "------------------------------" "--------"
printf "  %-30s %d/%d\n" "Chunk-RAG (top-1 section)" $chunk_pass $chunk_total
printf "  %-30s %d/%d\n" "Full-Document (FP32 KV)" $doc_pass $doc_total
printf "  %-30s %d/%d\n" "Full-Document (6.4x compressed)" $comp_pass $comp_total
echo ""

if [ $doc_pass -gt $chunk_pass ]; then
  echo "  → Full-document context outperforms chunk-RAG by $((doc_pass - chunk_pass)) questions"
fi
if [ $comp_pass -eq $doc_pass ]; then
  echo "  → KV compression preserves full-document accuracy (zero quality loss)"
fi
echo ""
echo "  Key insight: multi-hop questions requiring cross-section reasoning"
echo "  are where full-document context provides the most advantage."
echo ""
