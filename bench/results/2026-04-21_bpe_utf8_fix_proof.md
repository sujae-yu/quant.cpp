# BPE UTF-8 Double-Encoding Fix — End-to-End Proof (2026-04-21)

v0.27.0 closes two symmetric bugs in the GPT-2-style byte-level BPE
encode/decode paths. Until this fix, every Llama-3 / Qwen3-family prompt
that touched international chars (accents, CJK, Cyrillic, byte-fallback
emoji) was silently fed a different token sequence than HF's reference
tokenizer used, AND output bytes got double-encoded on their way back.

## Token-level parity with HF

Our engine vs `AutoTokenizer` from `Qwen/Qwen3-0.6B`, after v0.27.0:

| Input | HF reference | Our engine (pre-fix) | Our engine (post-fix) |
|---|---|---|---|
| `café` | [924, 58858] | [68796] ✗ | [924, 58858] ✓ |
| `naïve` | [3376, 37572, 586] | [77, 523] ✗ | [3376, 37572, 586] ✓ |
| `日本語` | [101059, 102819] | [245, 250, 252] ✗ | [101059, 102819] ✓ |
| `привет` | [124436, 26991, 8178] | [222, 224] ✗ | [124436, 26991, 8178] ✓ |
| `🎉` | [144841] | — | [144841] ✓ |
| `I❤️code` | [40, 141390, 30543, 1851] | — | [40, 141390, 30543, 1851] ✓ |
| `한글 테스트` | [23573, 83291, 10764, 72509, 53189] | — | [23573, 83291, 10764, 72509, 53189] ✓ |

**11/11 HF match** across ASCII/Latin-ext/CJK/Cyrillic/4-byte emoji.

## Output-level coherence (end-to-end)

After the fix, models produce meaningful multilingual continuations
instead of silent garbage. Same CLI, same config, different prompts:

```
Llama-3.2-1B-Instruct-Q8_0, -p "한국의 수도는" -n 20 -T 0
→ "?\n세계에서 10대 tuổi 이상의 인구가 가장 많을 때까지, 195"

Qwen3-0.6B-Q4_K_M, -p "한국의 수도는" -n 20 -T 0
→ " 현재로서는 정확히 1개인칭을 지닌 국가입니다. 이국의"

Qwen3.5-4B-Q4_K_M, -p "Le café est" -n 20 -T 0
→ " une boisson très populaire dans le monde entier. Il a été cultivé et consommé depuis des"
```

Qwen3.5-4B gives grammatically correct French ("a very popular drink
around the world. It has been cultivated and consumed for …"). Korean
completions parse as Korean even when factually shaky. Before the fix,
the same prompts went through a non-training-distribution token sequence
and produced various random-token outputs.

## What was wrong

### Encode side (`tq_tokenizer.c:encode_byte_to_bpe_char`)

For GPT-2 direct-byte codepoints 0xA1-0xAC and 0xAE-0xFF, the old code
emitted the raw byte into the lookup key:

```c
out[0] = (char)byte;   // byte 0xC3 → output 0xC3 (standalone = invalid UTF-8)
out[1] = '\0';
```

The vocab stores these bytes as *UTF-8-encoded* Unicode codepoints
(`byte 0xC3` → `"Ã"` = UTF-8 `c3 83`). A standalone 0x80+ byte is
invalid UTF-8, so `str_lookup` never matched. Characters silently fell
back to wrong low-id tokens.

### Decode side (`tq_tokenizer.c:decode_bpe_token`)

For vocab pieces containing codepoints U+00A1-U+00AC and U+00AE-U+00FF,
the old code emitted the UTF-8 encoding of the codepoint instead of the
raw byte it represents in GPT-2's mapping:

```c
decode_buf[out++] = (char)p[0];  // emit c3 (utf-8 byte 0)
decode_buf[out++] = (char)p[1];  // emit 83 (utf-8 byte 1)
```

So the byte 0xC3 came out as the two bytes `c3 83`. Combined with a byte
0xA9 coming out as `c2 a9`, "café" (3 bytes `63 61 66 c3 a9`) became the
6-byte `63 61 66 c3 83 c2 a9` — renders as "cafÃ©".

Both paths now detect the direct-byte codepoint range explicitly and
apply the inverse of GPT-2's byte-to-unicode mapping.

## Regression

```
$ bash scripts/test_models.sh
  PASS: 15 / FAIL: 0 / SKIP: 2       # coherence tier unchanged
  PASS: 11 / FAIL: 0 / SKIP: 0       # new tokenizer UTF-8 tier (chained)
```

23/23 on a single command. `scripts/test_tokenizer.sh` pins all seven
international fixtures so a future refactor of
`encode_byte_to_bpe_char` fails loudly.

## Scope

- **Affected**: Llama-3.x, Qwen2.5/3.x/3.5/3.6 — anything tagged
  `is_sentencepiece=0` in the engine's `[tokenizer]` log line.
- **Not affected**: Gemma (SentencePiece path), Phi-3 (SentencePiece path).

See also:
- `docs/RELEASE_NOTES.md` v0.27.0 entry
- `tools/refparity/` (the A/B framework that surfaced this bug)
- `scripts/test_tokenizer.sh` (the 11 HF-parity fixtures)
