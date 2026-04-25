/**
 * tq_generate.c — Text generation loop with TurboQuant KV cache
 *
 * Implements:
 *   - Argmax sampling (greedy)
 *   - Top-p (nucleus) sampling with temperature
 *   - Full generation loop with streaming callback
 */

#include "turboquant/turboquant.h"
#include "turboquant/tq_engine.h"
#include "turboquant/tq_gguf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#define pthread_mutex_t SRWLOCK
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define pthread_mutex_lock(m) AcquireSRWLockExclusive(m)
#define pthread_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#else
#include <pthread.h>
#endif

/* ============================================================
 * DRY (Don't Repeat Yourself) sampler penalty.
 * Ported from llama.cpp src/llama-sampler.cpp llama_sampler_dry_apply.
 *
 * Penalizes tokens that would EXTEND a repeated n-gram pattern in the
 * recent context. Penalty is exponential: penalty = multiplier * base^(n-allowed).
 * Much more surgical than rep_penalty: rep_penalty penalizes ALL recent
 * tokens uniformly; DRY only penalizes the specific tokens that would
 * continue a detected repeating sequence.
 *
 * Algorithm uses the Z-algorithm in reverse direction to compute, for
 * each offset k from newest-going-older, the length of the longest
 * suffix match. Then for each k with match >= allowed_length, the token
 * at position k-1 (one step newer than the match start) is the token
 * that if sampled NEXT would continue the pattern — we penalize it.
 *
 * Simplified vs llama.cpp: no restart sequence breakers (we just pure
 * text continuation; breakers like punctuation are handled by logit
 * ordering implicitly).
 *
 * O(N) time where N = min(recent_count, penalty_last_n).
 * ============================================================ */
static void apply_dry_penalty(
    float* logits, int vocab_size,
    const int* recent_tokens, int recent_count, int recent_cap,
    float multiplier, float base, int allowed_length, int penalty_last_n)
{
    if (multiplier <= 0.0f || base < 1.0f || penalty_last_n == 0) return;
    int last_n = recent_count < penalty_last_n ? recent_count : penalty_last_n;
    if (last_n > recent_cap) last_n = recent_cap;
    if (last_n <= allowed_length) return;

    /* rat(i) convention: i-th token from the newest (0 = newest).
     * Ring buffer: arr index = (recent_count - 1 - i) mod recent_cap. */
    #define RAT_IDX(i) ((((recent_count - 1 - (i)) % recent_cap) + recent_cap) % recent_cap)
    #define RAT(i)     (recent_tokens[RAT_IDX(i)])

    /* rc[k] = length of suffix match starting at offset k (from newest)
     * that equals the newest suffix. Range [0, last_n). */
    int* rc = (int*)calloc((size_t)last_n, sizeof(int));
    if (!rc) return;

    int rt = 0, lt = 0;
    for (int k = 1; k < last_n; k++) {
        if (k > rt) {
            int n = 0;
            while (n + k < last_n && RAT(n) == RAT(n + k)) n++;
            rc[k] = n;
            if (n > 0) { lt = k; rt = k + n - 1; }
        } else {
            int p = k - lt;
            int right_part_len = rt - k + 1;
            if (rc[p] < right_part_len) {
                rc[k] = rc[p];
            } else {
                int i = rt + 1;
                while (i < last_n && RAT(i) == RAT(i - k)) i++;
                rc[k] = i - k;
                lt = k; rt = i - 1;
            }
        }
    }

    /* For each k with rc[k] >= allowed_length, the token at rat(k-1)
     * would extend the pattern if sampled. Track max extension per token. */
    #define DRY_TOP 512
    int   top_tok[DRY_TOP];
    int   top_len[DRY_TOP];
    int   top_count = 0;

    for (int k = 1; k < last_n; k++) {
        int rep = rc[k];
        if (rep < allowed_length) continue;
        int tok = RAT(k - 1);
        if (tok < 0 || tok >= vocab_size) continue;
        /* Linear probe — typically very few unique extenders so this is fine. */
        int found = -1;
        for (int j = 0; j < top_count; j++) {
            if (top_tok[j] == tok) { found = j; break; }
        }
        if (found >= 0) {
            if (rep > top_len[found]) top_len[found] = rep;
        } else if (top_count < DRY_TOP) {
            top_tok[top_count] = tok;
            top_len[top_count] = rep;
            top_count++;
        }
    }

    /* Apply penalty. Clamp exponent to avoid overflow of powf. */
    const float FLOAT_MAX_LOG = 88.7228391f;
    int max_exp = (base > 1.000001f) ? (int)(FLOAT_MAX_LOG / logf(base)) : 0;
    for (int j = 0; j < top_count; j++) {
        int e = top_len[j] - allowed_length;
        if (max_exp > 0 && e > max_exp) e = max_exp;
        float pen = multiplier * powf(base, (float)e);
        logits[top_tok[j]] -= pen;
    }

    free(rc);
    #undef RAT_IDX
    #undef RAT
    #undef DRY_TOP
}

/* ============================================================
 * Argmax sampling: return token with highest logit
 * ============================================================ */
int tq_sample_argmax(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* ============================================================
 * Top-p (nucleus) sampling with temperature
 *
 * 1. Apply temperature scaling
 * 2. Compute softmax probabilities
 * 3. Sort by probability (descending)
 * 4. Accumulate until cumulative prob >= top_p
 * 5. Sample from the nucleus
 * ============================================================ */

/* Simple RNG (xorshift64) for reproducible sampling */
static float random_f32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 33) / (float)(1u << 31);
}

/* Comparison for sorting (probability, index) pairs */
typedef struct {
    float prob;
    int index;
} prob_index_t;

static int compare_prob_desc(const void* a, const void* b) {
    float pa = ((const prob_index_t*)a)->prob;
    float pb = ((const prob_index_t*)b)->prob;
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    return 0;
}

/* Persistent workspace to avoid per-token malloc.
 * Protected by mutex for thread safety when multiple model instances
 * call tq_sample_topp concurrently. */
static prob_index_t* g_probindex = NULL;
static int g_probindex_size = 0;
static pthread_mutex_t g_probindex_mutex = PTHREAD_MUTEX_INITIALIZER;

int tq_sample_topp(const float* logits, int vocab_size,
                   float temperature, float top_p,
                   unsigned long long* rng) {
    if (temperature <= 0.0f || top_p <= 0.0f) {
        return tq_sample_argmax(logits, vocab_size);
    }

    /* Pre-filter: only keep logits within reasonable range of max.
     * For top-p=0.9 with temperature=0.7, logits more than ~20 below max
     * contribute negligibly. This avoids sorting 248K entries. */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float threshold = max_val - 16.0f * temperature; /* exp(-16) ≈ 1e-7 */

    /* Allocate/reuse workspace (mutex-protected for concurrent callers) */
    pthread_mutex_lock(&g_probindex_mutex);
    if (g_probindex_size < vocab_size) {
        free(g_probindex);
        g_probindex = (prob_index_t*)malloc(vocab_size * sizeof(prob_index_t));
        g_probindex_size = vocab_size;
    }
    if (!g_probindex) {
        pthread_mutex_unlock(&g_probindex_mutex);
        return tq_sample_argmax(logits, vocab_size);
    }

    /* Collect only candidates above threshold */
    int n_candidates = 0;
    float sum = 0.0f;
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] >= threshold) {
            float p = expf((logits[i] - max_val) * inv_temp);
            g_probindex[n_candidates].prob = p;
            g_probindex[n_candidates].index = i;
            sum += p;
            n_candidates++;
        }
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n_candidates; i++) {
        g_probindex[i].prob *= inv_sum;
    }

    /* Sort only candidates (typically < 1000 vs 248K) */
    qsort(g_probindex, n_candidates, sizeof(prob_index_t), compare_prob_desc);

    /* Find top-p cutoff */
    float cumulative = 0.0f;
    int n_top = 0;
    for (int i = 0; i < n_candidates; i++) {
        cumulative += g_probindex[i].prob;
        n_top = i + 1;
        if (cumulative >= top_p) break;
    }

    /* Sample from the nucleus */
    float r = random_f32(rng) * cumulative;
    float cdf = 0.0f;
    int sampled = g_probindex[0].index;
    for (int i = 0; i < n_top; i++) {
        cdf += g_probindex[i].prob;
        if (cdf >= r) {
            sampled = g_probindex[i].index;
            break;
        }
    }

    pthread_mutex_unlock(&g_probindex_mutex);
    return sampled;
}

/* ============================================================
 * Generate text from prompt
 *
 * Steps:
 * 1. Encode prompt to tokens
 * 2. Prefill: forward all prompt tokens
 * 3. Decode: sample next token, forward, repeat
 * 4. Stop on EOS or max_tokens
 * ============================================================ */
int tq_generate(tq_model_t* model, tq_tokenizer_t* tokenizer,
                const char* prompt, tq_gen_config_t* config,
                char* output, int output_size) {
    if (!model || !config) return -1;

    tq_state_t* state = tq_create_state_ex(&model->config, config->kv_type, config->value_quant_bits);
    if (!state) {
        fprintf(stderr, "tq_generate: failed to allocate state\n");
        return -1;
    }
    state->delta_kv_enabled = config->delta_kv;
    state->delta_iframe_interval = config->delta_iframe_interval;
    /* Hybrid DeltaNet models: delta KV applies only to self_attn layers.
     * DeltaNet layers don't use key_cache, so delta compression is safe. */

    /* Allocate MoE state if model uses MoE */
    if (model->config.is_moe && model->moe_config) {
        state->moe_state = tq_moe_create_state(
            (const tq_moe_config_t*)model->moe_config,
            model->config.hidden_dim);
        if (!state->moe_state) {
            fprintf(stderr, "tq_generate: failed to allocate MoE state\n");
            tq_free_state(state);
            return -1;
        }
    }

    /* Set up V highres window if requested */
    if (config->v_highres_window > 0 &&
        (config->value_quant_bits == 4 || config->value_quant_bits == 2)) {
        int n_layers = model->config.n_layers;
        int kv_dim = model->config.n_kv_heads * model->config.head_dim;
        int window = config->v_highres_window;
        state->v_highres_window = window;
        state->value_highres_fp16 = (uint16_t*)calloc(
            (size_t)n_layers * window * kv_dim, sizeof(uint16_t));
    }

    /* Set up K highres window (age-based progressive compression) */
    if (config->k_highres_window > 0 &&
        state->kv_quant_type < TQ_TYPE_COUNT && state->quant_key_cache != NULL) {
        int n_layers = model->config.n_layers;
        int kv_dim = model->config.n_kv_heads * model->config.head_dim;
        int window = config->k_highres_window;
        state->k_highres_window = window;
        state->key_highres_fp32 = (float*)calloc(
            (size_t)n_layers * window * kv_dim, sizeof(float));
    }

    /* Encode prompt.
     * Pillar 1.5 R7 fix: buffer was 4096 which truncated any prompt
     * longer than ~4096 chars of English (BPE is char-level initial
     * then merge-compressed, so the per-char cap bites before merges
     * can reduce). Bumped to 32768 to support long-doc workflows up
     * to the model's max_seq_len (typically 16384 after merges).
     * 32768 × 4 bytes = 128 KB stack — fine on macOS (default 8 MB). */
    int prompt_tokens[32768];
    int n_prompt = 0;

    if (tokenizer && prompt) {
        /* BOS token handling:
         * Gemma 3/4: model_type==1, BOS=2 (required)
         * Phi-3 / LLaMA 2: vocab has <s> as BOS (required)
         * LLaMA 3: BOS=128000 (<|begin_of_text|>) — tq_encode lookup chain handles it
         * Qwen3.5 / GPT-2 BPE: no native BOS, skip */
        int add_bos = 0;
        if (model->config.model_type == 1) {
            add_bos = 1; /* Gemma: always prepend BOS=2 */
        } else {
            /* Auto-detect BOS: check if vocab contains <s> (LLaMA 2, Phi-3)
             * or <|begin_of_text|> (LLaMA 3). Both require BOS prepending. */
            for (int i = 0; i < tokenizer->vocab_size && i < 8; i++) {
                if (tokenizer->vocab[i] && strcmp(tokenizer->vocab[i], "<s>") == 0) {
                    add_bos = 1; break;
                }
            }
            /* LLaMA 3: <|begin_of_text|> is at high token ID (128000+), not in first 8.
             * Use direct lookup instead of scanning. */
            if (!add_bos) {
                int bos_id = -1;
                for (int i = 128000; i < tokenizer->vocab_size && i < 128010; i++) {
                    if (tokenizer->vocab[i] && strcmp(tokenizer->vocab[i], "<|begin_of_text|>") == 0) {
                        bos_id = i; break;
                    }
                }
                if (bos_id >= 0) add_bos = 1;
            }
            /* Qwen3.6 family (27B dense, 35B-A3B): GGUF metadata sets
             * BOS=<|endoftext|> id 248044. tokenizer.ggml.add_bos_token=false
             * but llama-cli adds BOS by default in main, and our basin_compat
             * measurements showed missing BOS causes 100× outlier divergence
             * at L0 (tokenization mismatch with reference). Detect by
             * presence of <|endoftext|> in vocab. */
            if (!add_bos) {
                /* <|endoftext|> for Qwen3.6 lives in 248040-248050 range (vocab=248320) */
                int lo = 248040, hi = 248060;
                if (hi > tokenizer->vocab_size) hi = tokenizer->vocab_size;
                for (int i = lo; i < hi; i++) {
                    if (tokenizer->vocab[i] && strcmp(tokenizer->vocab[i], "<|endoftext|>") == 0) {
                        add_bos = 1; break;
                    }
                }
            }
        }
        /* Qwen3.6 BOS-id fix: tq_encode str_lookup chain checks <|im_start|>
         * before <|endoftext|>, picking id 248045 instead of correct 248044
         * for Qwen3.6 family (27B, 35B-A3B). For these models, override the
         * BOS to <|endoftext|> directly. Detected by large vocab (>240K) +
         * presence of <|endoftext|>. */
        int qwen36_bos_override = -1;
        if (add_bos && tokenizer->vocab_size > 240000) {
            int lo = 248040, hi = 248060;
            if (hi > tokenizer->vocab_size) hi = tokenizer->vocab_size;
            for (int i = lo; i < hi; i++) {
                if (tokenizer->vocab[i] && strcmp(tokenizer->vocab[i], "<|endoftext|>") == 0) {
                    qwen36_bos_override = i; break;
                }
            }
        }
        n_prompt = tq_encode(tokenizer, prompt, prompt_tokens,
                              (int)(sizeof(prompt_tokens)/sizeof(prompt_tokens[0])),
                              add_bos);
        /* Qwen3.6 BOS override: tq_encode picked <|im_start|> (248045) but
         * GGUF metadata BOS = <|endoftext|> (248044). Replace at index 0. */
        if (qwen36_bos_override >= 0 && n_prompt > 0 && add_bos) {
            prompt_tokens[0] = qwen36_bos_override;
        }
        if (getenv("TQ_DEBUG_TOKENS")) {
            fprintf(stderr, "[tq_encode] add_bos=%d n_prompt=%d tokens=[", add_bos, n_prompt);
            for (int i = 0; i < n_prompt && i < 20; i++) fprintf(stderr, "%d%s", prompt_tokens[i], i+1<n_prompt?",":"");
            fprintf(stderr, "]\n");
        }
    } else {
        prompt_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_prompt = 1;
    }

    if (n_prompt <= 0) {
        prompt_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_prompt = 1;
    }

    /* Debug: print tokenized prompt */
    if (getenv("TQ_DEBUG")) {
        fprintf(stderr, "[DEBUG] prompt tokens (%d): ", n_prompt);
        for (int i = 0; i < n_prompt && i < 20; i++)
            fprintf(stderr, "%d ", prompt_tokens[i]);
        fprintf(stderr, "\n");
    }

    /* Load pre-computed KV cache if available (skip prefill) */
    int pos_after_prefill = n_prompt;
    if (config->load_kv_path) {
        FILE* kv_fp = fopen(config->load_kv_path, "rb");
        if (kv_fp) {
            int32_t saved_pos = 0;
            size_t kv_dim_save = 0;
            fread(&saved_pos, sizeof(int32_t), 1, kv_fp);
            fread(&kv_dim_save, sizeof(size_t), 1, kv_fp);
            size_t kv_dim = (size_t)model->config.n_kv_heads * model->config.head_dim;
            int max_seq = model->config.max_seq_len;
            size_t layer_stride = (size_t)max_seq * kv_dim;
            /* Read per-layer, respecting stride */
            for (int l = 0; l < model->config.n_layers; l++) {
                if (state->key_cache)
                    fread(state->key_cache + l * layer_stride, sizeof(float), (size_t)saved_pos * kv_dim, kv_fp);
                if (state->value_cache_fp16)
                    fread(state->value_cache_fp16 + l * layer_stride, sizeof(uint16_t), (size_t)saved_pos * kv_dim, kv_fp);
                else if (state->value_cache)
                    fread(state->value_cache + l * layer_stride, sizeof(float), (size_t)saved_pos * kv_dim, kv_fp);
            }
            fclose(kv_fp);
            pos_after_prefill = saved_pos;
            size_t total_bytes = (size_t)model->config.n_layers * saved_pos * kv_dim * (sizeof(float) + (state->value_cache_fp16 ? sizeof(uint16_t) : sizeof(float)));
            fprintf(stderr, "[load-kv] Loaded %d tokens from %s (%.1f MB)\n",
                    saved_pos, config->load_kv_path,
                    (double)total_bytes / (1024.0 * 1024.0));
        } else {
            fprintf(stderr, "[load-kv] Cannot open %s, running normal prefill\n", config->load_kv_path);
        }
    }

    /* Prefill: process prompt tokens.
     * If KV was loaded, the loaded context occupies positions [0..pos_after_prefill).
     * The new prompt is appended starting at pos_after_prefill. */
    int prefill_start = 0;
    if (config->load_kv_path && pos_after_prefill > 0) {
        prefill_start = pos_after_prefill;
    }
    /* Batched prefill: enabled by default for supported architectures.
     * Populates both FP32 K cache and quant_key_cache (if active) so that
     * the final tq_forward's attention sees baseline-equivalent history.
     * Set TQ_NO_BATCH_PREFILL=1 to force per-token (for A/B testing). */
    int batch_ok = 0;
    int want_batched = (n_prompt >= 2) && !getenv("TQ_NO_BATCH_PREFILL");
    if (want_batched) {
        /* Qwen3.6 (MoE + DeltaNet hybrid) needs the dedicated MoE-batched
         * driver — standard tq_forward_batch bails on is_moe. Pillar 1.5
         * R5 isolated a bug inside tq_moe_forward_batch at N≥40 that
         * produces UTF-8 garbage on natural prose; per-token forward is
         * correct. Pillar 1.5 R6 mitigation: **chunked batched dispatch**
         * — call the batched driver in chunks of CHUNK tokens each, where
         * each individual call satisfies the small-N safe region. State
         * (KV cache, DeltaNet state) is persistent across driver calls so
         * chunking is semantically correct. Gives most of the speed-up
         * while avoiding the N>>1 garbage regime. TQ_MOE_BATCH_CHUNK=N
         * overrides the default chunk size; 0 = one big call (unsafe).
         * TQ_NO_MOE_BATCH=1 forces per-token fallback. */
        int is_moe_hybrid = model->config.is_moe &&
                            !model->config.is_gemma4 &&
                            model->layers[0].moe;
        int use_moe_hybrid = is_moe_hybrid && !getenv("TQ_NO_MOE_BATCH");
        int rc;
        if (use_moe_hybrid) {
            const char* chunk_env = getenv("TQ_MOE_BATCH_CHUNK");
            int chunk = chunk_env ? atoi(chunk_env) : 8;  /* 8 is safe + fast */
            if (chunk <= 0) chunk = n_prompt; /* one big call (legacy) */
            if (chunk > n_prompt) chunk = n_prompt;
            rc = prefill_start;
            for (int start = 0; start < n_prompt; start += chunk) {
                int n_chunk = n_prompt - start;
                if (n_chunk > chunk) n_chunk = chunk;
                int chunk_rc = tq_forward_batch_moe_hybrid(
                    model, state, prompt_tokens + start, n_chunk, rc);
                if (chunk_rc != rc + n_chunk) { rc = chunk_rc; break; }
                rc = chunk_rc;
            }
        } else {
            rc = tq_forward_batch(model, state, prompt_tokens, n_prompt, prefill_start);
        }
        if (getenv("TQ_DEBUG_PREFILL"))
            fprintf(stderr, "[batch_prefill] driver=%s rc=%d expected=%d (N=%d)\n",
                    use_moe_hybrid ? "moe_hybrid" : "standard",
                    rc, prefill_start + n_prompt, n_prompt);
        if (rc == prefill_start + n_prompt) {
            /* tq_forward_batch now produces logits for the last position
             * itself (so we don't double-advance DeltaNet SSM state). No
             * final tq_forward needed. */
            batch_ok = 1;
        }
    }
    if (!batch_ok) {
        for (int i = 0; i < n_prompt; i++) {
            tq_forward(model, state, prompt_tokens[i], prefill_start + i);
        }
    }
    pos_after_prefill = prefill_start + n_prompt;

    /* Save KV cache after prefill if requested */
    if (config->save_kv_path && pos_after_prefill > 0) {
        FILE* kv_fp = fopen(config->save_kv_path, "wb");
        if (kv_fp) {
            int32_t save_pos = (int32_t)pos_after_prefill;
            size_t kv_dim = (size_t)model->config.n_kv_heads * model->config.head_dim;
            int max_seq = model->config.max_seq_len;
            size_t layer_stride = (size_t)max_seq * kv_dim;
            fwrite(&save_pos, sizeof(int32_t), 1, kv_fp);
            fwrite(&kv_dim, sizeof(size_t), 1, kv_fp);
            /* Write per-layer, only saved_pos positions */
            size_t total = 0;
            for (int l = 0; l < model->config.n_layers; l++) {
                if (state->key_cache) {
                    fwrite(state->key_cache + l * layer_stride, sizeof(float), (size_t)save_pos * kv_dim, kv_fp);
                    total += (size_t)save_pos * kv_dim * sizeof(float);
                }
                if (state->value_cache_fp16) {
                    fwrite(state->value_cache_fp16 + l * layer_stride, sizeof(uint16_t), (size_t)save_pos * kv_dim, kv_fp);
                    total += (size_t)save_pos * kv_dim * sizeof(uint16_t);
                } else if (state->value_cache) {
                    fwrite(state->value_cache + l * layer_stride, sizeof(float), (size_t)save_pos * kv_dim, kv_fp);
                    total += (size_t)save_pos * kv_dim * sizeof(float);
                }
            }
            fclose(kv_fp);
            fprintf(stderr, "[save-kv] Saved %d tokens to %s (%.1f MB)\n",
                    save_pos, config->save_kv_path, (double)total / (1024.0 * 1024.0));
        }
    }

    /* Repetition penalty setup */
    int vocab_size = model->config.vocab_size;
    float rep_penalty = config->rep_penalty;
    int rep_window = config->rep_window;
    if (rep_window > 128) rep_window = 128;
    int recent_tokens[128];
    int recent_count = 0;

    /* R62 K48: Rolling context refresh. Keeps all generated token IDs
     * so we can re-prefill the most recent N at proper RoPE positions
     * (0..N-1) after a refresh trigger, avoiding the RoPE mismatch that
     * breaks the infinite-scrollback path for Qwen3.6 DeltaNet models. */
    int refresh_every = config->refresh_ctx_every;
    int refresh_keep  = config->refresh_ctx_keep;
    if (refresh_every < 0) refresh_every = 0;
    if (refresh_keep  < 0) refresh_keep  = 0;
    if (refresh_every > 0 && refresh_keep <= 0) refresh_keep = refresh_every / 2;
    /* Env override — convenient to enable without API change. */
    {
        const char* s_every = getenv("TQ_REFRESH_CTX_EVERY");
        const char* s_keep  = getenv("TQ_REFRESH_CTX_KEEP");
        if (s_every) refresh_every = atoi(s_every);
        if (s_keep)  refresh_keep  = atoi(s_keep);
        if (refresh_every > 0 && refresh_keep <= 0) refresh_keep = refresh_every / 2;
    }
    int* gen_history = NULL;
    int  gen_history_cap = 0;
    if (refresh_every > 0) {
        gen_history_cap = config->max_tokens + 32;
        gen_history = (int*)calloc((size_t)gen_history_cap, sizeof(int));
        if (!gen_history) {
            refresh_every = 0;  /* allocation failed, disable feature */
        } else {
            fprintf(stderr, "[refresh-ctx] enabled: every=%d tokens, keep=%d\n",
                    refresh_every, refresh_keep);
        }
    }
    int gen_history_len = 0;
    int next_refresh_at = refresh_every;  /* total generated count to trigger */

    /* N-gram loop detection: track recent 4-grams to detect infinite loops.
     * Small models with T=0 greedy decoding enter repetition loops where
     * the same ~30-token pattern repeats endlessly. KV quantization error
     * compounds through these repetitions, eventually collapsing output
     * into garbage. Detecting loops early prevents wasted compute. */
    uint32_t ngram_hashes[64];
    int ngram_hash_count = 0;
    int loop_detected = 0;

    /* Seed recent tokens with tail of prompt for better penalty coverage */
    for (int i = (n_prompt > rep_window ? n_prompt - rep_window : 0); i < n_prompt; i++) {
        recent_tokens[recent_count % 128] = prompt_tokens[i];
        recent_count++;
    }

    /* Apply repetition penalty to logits before first sample */
    if (rep_penalty > 1.0f) {
        int window = recent_count < rep_window ? recent_count : rep_window;
        for (int r = 0; r < window; r++) {
            int idx = (recent_count - 1 - r) % 128;
            if (idx < 0) idx += 128;
            int tok = recent_tokens[idx];
            if (tok >= 0 && tok < vocab_size) {
                if (state->logits[tok] > 0)
                    state->logits[tok] /= rep_penalty;
                else
                    state->logits[tok] *= rep_penalty;
            }
        }
    }

    /* DRY penalty before first-token sample (if enabled) */
    if (config->dry_multiplier > 0.0f) {
        apply_dry_penalty(state->logits, vocab_size,
                          recent_tokens, recent_count, 128,
                          config->dry_multiplier, config->dry_base,
                          config->dry_allowed_length, config->dry_penalty_last_n);
    }

    /* Sample first generated token. The seed is configurable via
     * config->rng_seed to support reproducible sampling sweeps; 0 falls
     * back to the historical default of 42 so existing callers that
     * never set rng_seed get bit-identical behaviour. */
    int pos = pos_after_prefill;
    unsigned long long rng_state = config->rng_seed ? config->rng_seed : 42ULL;
    int next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

    /* Record first sampled token */
    recent_tokens[recent_count % 128] = next_token;
    recent_count++;
    if (gen_history && gen_history_len < gen_history_cap) {
        gen_history[gen_history_len++] = next_token;
    }

    int generated = 0;
    int output_pos = 0;
    int prev_token = prompt_tokens[n_prompt - 1];

    /* EOS token IDs — check common values across model families.
     * Qwen3.5: eos = 248044 (<|endoftext|>), 248046 (<|im_end|>)
     * Gemma3: eos = 1
     * Gemma4: eos = 106 (<end_of_turn>)
     * LLaMA 2: eos = 2
     * LLaMA 3: eos = 128001 (<|end_of_text|>), 128009 (<|eot_id|>) */
    int eos_tokens[] = {
        1,       /* Gemma3 <eos> */
        2,       /* LLaMA 2 </s> */
        106,     /* Gemma4 <end_of_turn> */
        128001,  /* LLaMA 3 <|end_of_text|> */
        128006,  /* LLaMA 3 <|start_header_id|> (new turn = stop) */
        128007,  /* LLaMA 3 <|end_header_id|> */
        128008,  /* LLaMA 3 <|start_of_role|> */
        128009,  /* LLaMA 3 <|eot_id|> */
        248044,  /* Qwen <|endoftext|> */
        248046,  /* Qwen <|im_end|> */
    };
    int n_eos = sizeof(eos_tokens) / sizeof(eos_tokens[0]);

    /* R62 K47: TQ_IGNORE_EOS=1 forces generation past EOS tokens.
     * Used to push past model's natural termination (~268 tok in
     * thinking mode) for ultra-long coherent output. Replaces EOS
     * argmax with rank-2 token effectively. */
    static int ignore_eos = -1;
    if (ignore_eos == -1) ignore_eos = getenv("TQ_IGNORE_EOS") ? 1 : 0;

    /* Generate loop */
    while (generated < config->max_tokens) {
        int is_eos = 0;
        for (int e = 0; e < n_eos; e++) {
            if (next_token == eos_tokens[e]) { is_eos = 1; break; }
        }
        if (is_eos && !ignore_eos) break;
        if (is_eos && ignore_eos) {
            /* Replace EOS with second-best non-EOS token by searching
             * the last-computed logits (state->logits). */
            if (state->logits) {
                float best_val = -HUGE_VALF;
                int best_id = -1;
                int vs = model->config.vocab_size;
                for (int i = 0; i < vs; i++) {
                    int is_e = 0;
                    for (int e = 0; e < n_eos; e++) if (i == eos_tokens[e]) { is_e = 1; break; }
                    if (is_e) continue;
                    if (state->logits[i] > best_val) { best_val = state->logits[i]; best_id = i; }
                }
                if (best_id >= 0) next_token = best_id;
            }
        }
        /* Infinite scrollback: when context is full, shift the KV cache
         * instead of stopping. Keep the last half of the context (including
         * the FP32 hot window) and discard the oldest half. This mirrors
         * human memory: ancient context fades, recent stays sharp.
         *
         * After shift, pos is reset to keep_count and generation continues.
         * The KV cache data for discarded positions is simply overwritten
         * by future tokens — no explicit deletion needed for the quantized
         * cache (block-indexed by position modulo max_seq_len). */
        if (pos >= model->config.max_seq_len) {
            int max_seq = model->config.max_seq_len;
            int keep_count = max_seq / 2;  /* keep most recent half */
            int discard = pos - keep_count;
            if (discard <= 0) break;  /* safety: can't shift if nothing to discard */

            fprintf(stderr, "[infinite scrollback] context full at %d, "
                    "shifting: discard oldest %d, keep %d\n",
                    pos, discard, keep_count);

            /* Shift FP32 key/value caches (if present) */
            int kv_dim = model->config.n_kv_heads * model->config.head_dim;
            for (int l = 0; l < model->config.n_layers; l++) {
                size_t layer_off = (size_t)l * max_seq * kv_dim;
                if (state->key_cache) {
                    memmove(state->key_cache + layer_off,
                            state->key_cache + layer_off + (size_t)discard * kv_dim,
                            (size_t)keep_count * kv_dim * sizeof(float));
                }
                if (state->value_cache) {
                    memmove(state->value_cache + layer_off,
                            state->value_cache + layer_off + (size_t)discard * kv_dim,
                            (size_t)keep_count * kv_dim * sizeof(float));
                }
                if (state->value_cache_fp16) {
                    size_t layer_off16 = (size_t)l * max_seq * kv_dim;
                    memmove(state->value_cache_fp16 + layer_off16,
                            state->value_cache_fp16 + layer_off16 + (size_t)discard * kv_dim,
                            (size_t)keep_count * kv_dim * sizeof(uint16_t));
                }
                /* Quantized K cache: shift block-level data */
                if (state->quant_key_cache && state->kv_quant_type < TQ_TYPE_COUNT) {
                    size_t blk_sz = tq_type_type_size(state->kv_quant_type);
                    size_t q_stride = (size_t)max_seq * blk_sz;
                    uint8_t* qbase = (uint8_t*)state->quant_key_cache + (size_t)l * q_stride;
                    memmove(qbase,
                            qbase + (size_t)discard * blk_sz,
                            (size_t)keep_count * blk_sz);
                }
            }

            /* Reset position: keep absolute position for correct RoPE.
             * Keys in the KV cache have RoPE baked in at their original
             * positions. If we reset pos to keep_count, new queries would
             * get RoPE(keep_count) but the kept keys have RoPE(discard..pos),
             * giving wrong relative distances. Instead, DON'T change pos —
             * continue from the same absolute position. The attention will
             * only scan positions [discard..pos] which are now at cache
             * indices [0..keep_count]. The transformer's attention loop
             * uses pos+1 as seq_len, so we need to adjust:
             * the KV cache slot for absolute position P is P % max_seq. */
            /* For now: use the simpler approach matching llama.cpp's
             * context shift: keep pos as-is but wrap cache indices. */
            pos = keep_count;
            /* NOTE: this has a RoPE mismatch — same as llama.cpp's
             * basic context shift. Quality degrades ~2-5% per shift.
             * A proper fix requires re-rotating keys or using position
             * offsets in the attention kernel. Tracked for v0.11. */
        }

        /* Decode token to text */
        if (tokenizer) {
            const char* piece = tq_decode(tokenizer, prev_token, next_token);

            /* Skip special/thinking tokens that shouldn't appear in output.
             * Qwen3.5: <think>...</think>
             * Gemma 4: thought, <channel|>, <tool|>, <mask>, <unused*>
             * LLaMA 3: <|start_header_id|>, <|reserved_special_token_*|> */
            int should_stop = 0;
            if (piece) {
                if (strstr(piece, "<think>") || strstr(piece, "</think>") ||
                    strstr(piece, "<channel|>") || strstr(piece, "<tool|>") ||
                    strstr(piece, "<mask>") ||
                    strstr(piece, "<unused") || strstr(piece, "<|think")) {
                    piece = "";
                }
                /* Gemma 4 "thought" token: only filter if it's the EXACT piece
                 * (not a substring of normal text like "thoughtful") */
                if (piece[0] != '\0' && strcmp(piece, "thought") == 0) {
                    piece = "";
                }
                /* Stop generation on turn-boundary tokens (LLaMA 3 / Qwen only).
                 * Gemma uses token ID-based EOS (106), not text-based detection. */
                if (strstr(piece, "<|start_header_id|>") ||
                    strstr(piece, "<|eot_id|>") ||
                    strstr(piece, "<|im_end|>")) {
                    should_stop = 1;
                    piece = "";
                }
                /* Filter reserved special tokens */
                if (strstr(piece, "<|reserved_special_token") ||
                    strstr(piece, "<1st>") || strstr(piece, "<2nd>") || strstr(piece, "<3rd>")) {
                    piece = "";
                }
            }
            if (should_stop) break;

            /* Also check accumulated output for turn markers that span multiple tokens */
            if (output && output_pos > 5) {
                const char* tail = output + (output_pos > 20 ? output_pos - 20 : 0);
                if (strstr(tail, "<|start_header") || strstr(tail, "<|eot_id") ||
                    strstr(tail, "<end_of_turn") || strstr(tail, "<|im_end")) {
                    /* Trim the marker from output */
                    char* marker = strstr(output + (output_pos > 30 ? output_pos - 30 : 0), "<|");
                    if (!marker) marker = strstr(output + (output_pos > 30 ? output_pos - 30 : 0), "<end");
                    if (marker) { *marker = '\0'; output_pos = (int)(marker - output); }
                    break;
                }
            }

            int piece_len = (int)strlen(piece);

            /* Stream callback */
            if (config->on_token) {
                config->on_token(piece, config->user_data);
            }

            /* Append to output buffer */
            if (output && output_pos + piece_len < output_size - 1) {
                memcpy(output + output_pos, piece, piece_len);
                output_pos += piece_len;
            }
        }

        /* Forward pass for next token */
        prev_token = next_token;
        tq_forward(model, state, next_token, pos);
        pos++;
        generated++;

        /* Apply repetition penalty before sampling */
        if (rep_penalty > 1.0f) {
            int window = recent_count < rep_window ? recent_count : rep_window;
            for (int r = 0; r < window; r++) {
                int idx = (recent_count - 1 - r) % 64;
                if (idx < 0) idx += 64;
                int tok = recent_tokens[idx];
                if (tok >= 0 && tok < vocab_size) {
                    if (state->logits[tok] > 0)
                        state->logits[tok] /= rep_penalty;
                    else
                        state->logits[tok] *= rep_penalty;
                }
            }
        }

        /* DRY penalty before sample (if enabled) */
        if (config->dry_multiplier > 0.0f) {
            apply_dry_penalty(state->logits, vocab_size,
                              recent_tokens, recent_count, 128,
                              config->dry_multiplier, config->dry_base,
                              config->dry_allowed_length, config->dry_penalty_last_n);
        }

        /* Sample next token */
        next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

        /* Record sampled token for repetition penalty */
        recent_tokens[recent_count % 128] = next_token;
        recent_count++;
        if (gen_history && gen_history_len < gen_history_cap) {
            gen_history[gen_history_len++] = next_token;
        }

        /* R62 K48: Rolling context refresh.
         * Trigger when we've generated `refresh_every` tokens since last
         * boundary. Keep the most recent `refresh_keep` generated tokens,
         * reset all recurrent/KV state, then re-prefill those tokens at
         * positions 0..refresh_keep-1 so subsequent generation has
         * RoPE-correct context without cumulative attention drift. */
        if (refresh_every > 0 && gen_history && gen_history_len >= next_refresh_at) {
            int K = refresh_keep;
            if (K > gen_history_len) K = gen_history_len;

            /* R62 K49: attractor-aware tail trim.
             * Before re-prefilling, inspect the last 32 tokens of the
             * proposed keep window. If their unique-token diversity is
             * below threshold (model is stuck in an attractor), walk the
             * keep boundary backward to find a cleaner (more diverse)
             * 32-token window. This prevents carrying the attractor tail
             * into the fresh state, which was the root cause of cycle-3+
             * quality collapse in the base K48 implementation. */
            int keep_end = gen_history_len;  /* exclusive */
            while (K >= 64) {
                int probe_start = keep_end - 32;
                if (probe_start < gen_history_len - K) break;
                int unique = 0;
                /* simple O(32^2) uniqueness scan, 32 items — negligible */
                for (int a = probe_start; a < keep_end; a++) {
                    int dup = 0;
                    for (int b = probe_start; b < a; b++) {
                        if (gen_history[a] == gen_history[b]) { dup = 1; break; }
                    }
                    if (!dup) unique++;
                }
                /* Good window: ≥ 20/32 unique tokens (62.5%) */
                if (unique >= 20) break;
                /* Bad tail: pull keep_end back 16 tokens and retry */
                keep_end -= 16;
            }
            int keep_start = keep_end - K;
            if (keep_start < 0) { keep_start = 0; K = keep_end; }
            if (keep_end < gen_history_len) {
                fprintf(stderr, "\n[refresh-ctx] trim %d attractor tail tokens before refresh\n",
                        gen_history_len - keep_end);
            }
            if (K > 0) {
                fprintf(stderr, "[refresh-ctx] triggered at gen=%d, re-prefilling tokens [%d..%d)\n",
                        gen_history_len, keep_start, keep_end);
                int* tail = gen_history + keep_start;
                /* Discard the trimmed tail from gen_history so subsequent
                 * refresh cycles don't re-see the attractor. */
                gen_history_len = keep_end;

                /* Reset all recurrent + KV state to zero — these are bulk
                 * calloc'd arrays owned by state. tq_forward will rebuild
                 * them as we re-prefill. */
                int n_layers_ = model->config.n_layers;
                int kv_dim_   = model->config.n_kv_heads * model->config.head_dim;
                int max_seq_  = model->config.max_seq_len;
                if (state->key_cache) {
                    memset(state->key_cache, 0,
                           (size_t)n_layers_ * max_seq_ * kv_dim_ * sizeof(float));
                }
                if (state->value_cache) {
                    memset(state->value_cache, 0,
                           (size_t)n_layers_ * max_seq_ * kv_dim_ * sizeof(float));
                }
                if (state->value_cache_fp16) {
                    memset(state->value_cache_fp16, 0,
                           (size_t)n_layers_ * max_seq_ * kv_dim_ * sizeof(uint16_t));
                }
                if (state->delta_state) {
                    int dn = model->config.delta_n_heads;
                    int dk = model->config.delta_key_head_dim;
                    int dv = model->config.delta_value_head_dim;
                    if (dn > 0 && dk > 0 && dv > 0) {
                        memset(state->delta_state, 0,
                               (size_t)n_layers_ * dn * dk * dv * sizeof(float));
                        if (state->delta_state_fp64) {
                            memset(state->delta_state_fp64, 0,
                                   (size_t)n_layers_ * dn * dk * dv * sizeof(double));
                        }
                    }
                }
                if (state->conv_state) {
                    int delta_qkv_dim = 0;
                    if (model->config.delta_n_heads > 0) {
                        delta_qkv_dim = model->config.delta_n_heads * model->config.delta_value_head_dim
                                      + 2 * model->config.delta_n_kv_heads * model->config.delta_key_head_dim;
                    }
                    int conv_w = model->config.delta_conv_width > 0 ? model->config.delta_conv_width : 4;
                    if (delta_qkv_dim > 0) {
                        memset(state->conv_state, 0,
                               (size_t)n_layers_ * delta_qkv_dim * (conv_w - 1) * sizeof(float));
                    }
                }

                /* Re-prefill the kept tail at positions 0..K-1 */
                for (int i = 0; i < K; i++) {
                    tq_forward(model, state, tail[i], i);
                }
                /* After re-prefill, state->logits reflects token K-1's
                 * next-token distribution. Subsequent generation continues
                 * from pos=K with correct RoPE. */
                pos = K;
                /* Drop tokens we've "forgotten" from the recent buffer so
                 * rep-penalty and DRY don't misfire across the boundary. */
                recent_count = 0;
                for (int i = 0; i < K && i < 128; i++) {
                    recent_tokens[i] = tail[i];
                    recent_count++;
                }
                /* Reset n-gram hash history — old 4-gram hashes refer to
                 * tokens before the refresh. */
                ngram_hash_count = 0;
                /* We just sampled next_token that already got logged to
                 * gen_history above. Apply sampler to fresh state->logits. */
                if (rep_penalty > 1.0f) {
                    int window = recent_count < rep_window ? recent_count : rep_window;
                    for (int r = 0; r < window; r++) {
                        int idx = (recent_count - 1 - r) % 128;
                        if (idx < 0) idx += 128;
                        int tok = recent_tokens[idx];
                        if (tok >= 0 && tok < vocab_size) {
                            if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                            else                         state->logits[tok] *= rep_penalty;
                        }
                    }
                }
                if (config->dry_multiplier > 0.0f) {
                    apply_dry_penalty(state->logits, vocab_size,
                                      recent_tokens, recent_count, 128,
                                      config->dry_multiplier, config->dry_base,
                                      config->dry_allowed_length, config->dry_penalty_last_n);
                }
                next_token = tq_sample_topp(state->logits, vocab_size,
                                             config->temperature, config->top_p,
                                             &rng_state);
                recent_tokens[recent_count % 128] = next_token;
                recent_count++;
                if (gen_history_len < gen_history_cap) {
                    gen_history[gen_history_len++] = next_token;
                }
                /* Schedule next refresh: refresh_every tokens after current
                 * (post-trim) length. Avoids infinite trigger when trim
                 * shortens gen_history below previous boundary. */
                next_refresh_at = gen_history_len + refresh_every;
            }
        }

        /* N-gram loop detection: hash recent 4-gram and check for repeats.
         * TQ_NO_LOOP_DETECT=1 disables this early-stop so long-form benchmarks
         * can push past transient attractors. Useful when measuring how far
         * the engine can coherently go before the model itself terminates. */
        if (recent_count >= 4 && getenv("TQ_NO_LOOP_DETECT") == NULL) {
            uint32_t h = 0;
            for (int r = 0; r < 4; r++) {
                int gi = (recent_count - 4 + r) % 128;
                h = h * 31 + (uint32_t)recent_tokens[gi];
            }
            int matches = 0;
            int ring_len = ngram_hash_count < 64 ? ngram_hash_count : 64;
            for (int r = 0; r < ring_len; r++) {
                if (ngram_hashes[r] == h) matches++;
            }
            ngram_hashes[ngram_hash_count % 64] = h;
            ngram_hash_count++;
            if (matches >= 3) {
                loop_detected = 1;
                break;
            }
        }
    }

    if (loop_detected) {
        fprintf(stderr, "[generate] repetition loop detected after %d tokens, stopping\n", generated);
    }

    /* Null-terminate output */
    if (output && output_size > 0) {
        output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
    }

    if (gen_history) free(gen_history);
    tq_free_state(state);
    return generated;
}

/* ============================================================================
 * tq_generate_continue — chat-mode generation with KV cache reuse (token LCP).
 *
 * Caller-managed state: state and cached_tokens persist across calls.
 * Each call computes the longest common prefix between cached_tokens and
 * the new prompt, prefills only the diverging suffix, and updates the
 * cache record. Turns chat from O(history^2) into O(new_tokens_per_turn).
 *
 * NOTE: This is a lower-level API. It does NOT track cached_text. If a
 * sliding window triggers (n_cached_io is reset to 0), any out-of-band
 * cached_text the caller maintains becomes stale. Higher-level callers
 * should use tq_generate_chat_text instead, which handles this safely.
 * ============================================================================ */
static int tq_lcp_int(const int* a, int na, const int* b, int nb) {
    int lim = na < nb ? na : nb;
    int i = 0;
    while (i < lim && a[i] == b[i]) i++;
    return i;
}

int tq_generate_continue(tq_model_t* model,
                          tq_tokenizer_t* tokenizer,
                          tq_state_t* state,
                          const char* prompt,
                          tq_gen_config_t* config,
                          int** cached_tokens_io,
                          int*  n_cached_io,
                          int*  cached_capacity_io,
                          char* output, int output_size) {
    if (!model || !state || !config || !cached_tokens_io || !n_cached_io || !cached_capacity_io) {
        return -1;
    }

    /* Encode new prompt — use a heap buffer that grows on demand instead
     * of a fixed stack array. The previous int new_tokens[4096] silently
     * truncated long contexts (10+ turns of accumulated chat history).
     * Cap at the model's max_seq_len so we never exceed KV cache bounds. */
    int max_prompt = model->config.max_seq_len > 0
                       ? model->config.max_seq_len : 4096;
    int* new_tokens = (int*)malloc((size_t)max_prompt * sizeof(int));
    if (!new_tokens) return -1;
    int n_new = 0;
    if (tokenizer && prompt) {
        int add_bos = 0;
        if (model->config.model_type == 1) {
            add_bos = 1;
        } else {
            for (int i = 0; i < tokenizer->vocab_size && i < 8; i++) {
                if (tokenizer->vocab[i] && strcmp(tokenizer->vocab[i], "<s>") == 0) {
                    add_bos = 1; break;
                }
            }
            if (!add_bos) {
                for (int i = 128000; i < tokenizer->vocab_size && i < 128010; i++) {
                    if (tokenizer->vocab[i] && strcmp(tokenizer->vocab[i], "<|begin_of_text|>") == 0) {
                        add_bos = 1; break;
                    }
                }
            }
        }
        n_new = tq_encode(tokenizer, prompt, new_tokens, max_prompt, add_bos);
    }
    if (n_new <= 0) {
        new_tokens[0] = (model->config.model_type == 1) ? 2 : 1;
        n_new = 1;
    }

    /* Overflow check: reject prompts that won't fit. The previous
     * behavior was to silently drop oldest tokens via a sliding window,
     * but that desynced any cached_text the higher-level wrapper held
     * (cached_text claimed the full prompt, while cached_tokens only
     * had the truncated tail — next turn's text-prefix match would
     * map text bytes to the wrong KV positions). Returning -2 lets the
     * caller decide (reset chat, show error). */
    int reserve = config->max_tokens > 0 ? config->max_tokens : 256;
    int budget  = max_prompt - reserve - 32;
    if (budget < 64) budget = 64;
    if (n_new > budget) {
        free(new_tokens);
        if (getenv("TQ_CHAT_DEBUG")) {
            fprintf(stderr, "[chat] OVERFLOW n_new=%d budget=%d max=%d\n",
                    n_new, budget, max_prompt);
        }
        return -2;
    }

    int n_cached = *n_cached_io;
    int* cached_tokens = *cached_tokens_io;
    int lcp = tq_lcp_int(cached_tokens, n_cached, new_tokens, n_new);

    /* Prefill only the new suffix [lcp, n_new) */
    for (int i = lcp; i < n_new; i++) {
        tq_forward(model, state, new_tokens[i], i);
    }
    int pos = n_new;

    /* Track prefill metrics for observability */
    int prefill_tokens = n_new - lcp;
    int prefix_hit    = lcp;

    /* Grow cache buffer if needed */
    int needed_cap = n_new + config->max_tokens + 16;
    if (*cached_capacity_io < needed_cap) {
        int new_cap = needed_cap < 4096 ? 4096 : needed_cap;
        int* nb = (int*)realloc(*cached_tokens_io, (size_t)new_cap * sizeof(int));
        if (!nb) { free(new_tokens); return -1; }
        *cached_tokens_io = nb;
        *cached_capacity_io = new_cap;
        cached_tokens = nb;
    }
    memcpy(cached_tokens, new_tokens, (size_t)n_new * sizeof(int));
    *n_cached_io = n_new;
    n_cached = n_new;

    int vocab_size = model->config.vocab_size;
    float rep_penalty = config->rep_penalty;
    int rep_window = config->rep_window;
    if (rep_window > 64) rep_window = 64;
    int recent_tokens[64];
    int recent_count = 0;
    for (int i = (n_new > rep_window ? n_new - rep_window : 0); i < n_new; i++) {
        recent_tokens[recent_count % 64] = new_tokens[i];
        recent_count++;
    }

    if (rep_penalty > 1.0f) {
        int window = recent_count < rep_window ? recent_count : rep_window;
        for (int r = 0; r < window; r++) {
            int idx = (recent_count - 1 - r) % 64;
            if (idx < 0) idx += 64;
            int tok = recent_tokens[idx];
            if (tok >= 0 && tok < vocab_size && state->logits) {
                if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                else                         state->logits[tok] *= rep_penalty;
            }
        }
    }

    unsigned long long rng_state = config->rng_seed ? (unsigned long long)config->rng_seed
                                                    : (unsigned long long)time(NULL);
    int next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);

    int generated = 0;
    int output_pos = 0;
    int prev_token = new_tokens[n_new - 1];

    int eos_tokens[] = {
        1, 2, 106, 128001, 128006, 128007, 128008, 128009, 248044, 248046,
    };
    int n_eos = sizeof(eos_tokens) / sizeof(eos_tokens[0]);

    while (generated < config->max_tokens) {
        int is_eos = 0;
        for (int e = 0; e < n_eos; e++) {
            if (next_token == eos_tokens[e]) { is_eos = 1; break; }
        }
        if (is_eos) break;
        if (pos >= model->config.max_seq_len) break;

        if (tokenizer) {
            const char* piece = tq_decode(tokenizer, prev_token, next_token);
            int should_stop = 0;
            if (piece) {
                if (strstr(piece, "<|im_end|>") || strstr(piece, "<|eot_id|>") ||
                    strstr(piece, "<|start_header_id|>")) {
                    should_stop = 1; piece = "";
                }
            }
            if (should_stop) break;
            int piece_len = (int)strlen(piece ? piece : "");
            if (config->on_token && piece) config->on_token(piece, config->user_data);
            if (output && piece && output_pos + piece_len < output_size - 1) {
                memcpy(output + output_pos, piece, piece_len);
                output_pos += piece_len;
            }
        }

        if (n_cached < *cached_capacity_io) {
            cached_tokens[n_cached++] = next_token;
            *n_cached_io = n_cached;
        }

        prev_token = next_token;
        tq_forward(model, state, next_token, pos);
        pos++;
        generated++;

        if (rep_penalty > 1.0f) {
            int window = recent_count < rep_window ? recent_count : rep_window;
            for (int r = 0; r < window; r++) {
                int idx = (recent_count - 1 - r) % 64;
                if (idx < 0) idx += 64;
                int tok = recent_tokens[idx];
                if (tok >= 0 && tok < vocab_size) {
                    if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                    else                         state->logits[tok] *= rep_penalty;
                }
            }
        }

        next_token = tq_sample_topp(state->logits, vocab_size,
                                     config->temperature, config->top_p,
                                     &rng_state);
        recent_tokens[recent_count % 64] = next_token;
        recent_count++;
    }

    if (output && output_size > 0) {
        output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
    }

    /* Log cache metrics: prefix_hit / prefill_tokens / generated.
     * Useful for tuning chat clients that want to maximize KV reuse. */
    if (getenv("TQ_CHAT_DEBUG")) {
        fprintf(stderr,
            "[chat] prefix_hit=%d prefill=%d generated=%d cached=%d\n",
            prefix_hit, prefill_tokens, generated, *n_cached_io);
    }

    free(new_tokens);
    return generated;
}

/* ============================================================================
 * tq_generate_chat_text — text-prefix matching for chat reuse
 *
 * Solves the BPE re-tokenization issue: when the model generates response
 * tokens via sample_topp, those token IDs may not match what tq_encode()
 * produces from the same response text in the next turn's prompt. The
 * token-level LCP in tq_generate_continue truncates at that boundary.
 *
 * This function tracks the *text* of the last prompt (which includes the
 * model's response from previous turns, accumulated by the caller). On the
 * next call, if the new prompt starts with cached_text byte-for-byte, the
 * entire cached state is valid — we tokenize only the new SUFFIX text and
 * prefill those tokens at positions [n_cached..]. No LCP, no truncation.
 *
 * After generation, *cached_text_io is updated to:
 *   prompt + (generated tokens decoded back to text)
 * so the next call can fast-path again.
 *
 * Caller owns *cached_text_io (must free with free()).
 * Pass cached_text_io == NULL to disable text-prefix tracking and behave
 * exactly like tq_generate_continue.
 * ============================================================================ */

/* ChatML / template-marker filter ----------------------------------------
 *
 * The model can generate template tokens like `<|im_start|>`, `<|im_end|>`,
 * `<end_of_turn>`, etc. as REGULAR text bytes (not special tokens). When
 * that happens the BPE tokenizer fragments them across multiple tokens,
 * and a per-token strstr check (like the existing `should_stop` logic)
 * never matches. The user sees the marker leak into their stream.
 *
 * This filter holds the most recent CHAT_LOOKAHEAD bytes of generated
 * text in `pending` and only flushes bytes that are guaranteed to NOT
 * be the start of a marker. When a full marker is matched:
 *   - `<|im_start|>` at the very beginning of the response → header
 *     skip mode (drop until next '\n').
 *   - any END marker → emit prefix, drop the rest, set stop_requested.
 *
 * Mirrored byte-for-byte with the version in quant.h. ---------------------- */
#define CHAT_PENDING_CAP 128
#define CHAT_LOOKAHEAD   32

typedef struct {
    char*  buf;
    size_t len;
    size_t cap;
    int    tainted;
    char   pending[CHAT_PENDING_CAP];
    int    pending_len;
    int    in_header;
    int    stop_requested;
    void (*user_cb)(const char*, void*);
    void*  user_data;
} chat_accum_t;

static void chat_accum_emit(chat_accum_t* ctx, const char* p, int n) {
    if (n <= 0) return;
    char tmp[CHAT_PENDING_CAP + 1];
    if (n > CHAT_PENDING_CAP) n = CHAT_PENDING_CAP;
    memcpy(tmp, p, (size_t)n);
    tmp[n] = '\0';
    if (ctx->user_cb) ctx->user_cb(tmp, ctx->user_data);
    if (ctx->tainted) return;
    if (ctx->len + (size_t)n + 1 > ctx->cap) {
        size_t new_cap = (ctx->cap + (size_t)n + 64) * 2;
        char* nb = (char*)realloc(ctx->buf, new_cap);
        if (!nb) { ctx->tainted = 1; return; }
        ctx->buf = nb; ctx->cap = new_cap;
    }
    memcpy(ctx->buf + ctx->len, tmp, (size_t)n);
    ctx->len += (size_t)n;
    ctx->buf[ctx->len] = '\0';
}

static void chat_accum_drop(chat_accum_t* ctx, int n) {
    if (n <= 0) return;
    if (n > ctx->pending_len) n = ctx->pending_len;
    memmove(ctx->pending, ctx->pending + n,
            (size_t)(ctx->pending_len - n));
    ctx->pending_len -= n;
}

static int chat_find_marker(const char* h, int hlen, const char* m) {
    int mlen = (int)strlen(m);
    if (hlen < mlen) return -1;
    for (int p = 0; p + mlen <= hlen; p++) {
        if (h[p] == m[0] && memcmp(h + p, m, (size_t)mlen) == 0) return p;
    }
    return -1;
}

static const char* const CHAT_END_MARKERS[] = {
    "<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>",
    "<|im_start|>", "<|start_header_id|>", "<|eom_id|>",
    "</s>", "<|end|>",
    NULL,
};

static void chat_accum_callback(const char* tok, void* u) {
    chat_accum_t* ctx = (chat_accum_t*)u;
    if (!tok || ctx->stop_requested) return;
    int tlen = (int)strlen(tok);
    if (tlen == 0) return;

    if (ctx->pending_len + tlen > CHAT_PENDING_CAP) {
        int emit = ctx->pending_len - CHAT_LOOKAHEAD;
        if (emit > 0) {
            if (!ctx->in_header) chat_accum_emit(ctx, ctx->pending, emit);
            chat_accum_drop(ctx, emit);
        }
    }
    if (tlen > CHAT_PENDING_CAP) {
        if (!ctx->in_header) {
            chat_accum_emit(ctx, ctx->pending, ctx->pending_len);
            chat_accum_emit(ctx, tok, tlen);
        }
        ctx->pending_len = 0;
        return;
    }
    memcpy(ctx->pending + ctx->pending_len, tok, (size_t)tlen);
    ctx->pending_len += tlen;

    int progress = 1;
    while (progress) {
        progress = 0;
        if (ctx->in_header) {
            int nl = -1;
            for (int i = 0; i < ctx->pending_len; i++) {
                if (ctx->pending[i] == '\n') { nl = i; break; }
            }
            if (nl >= 0) {
                chat_accum_drop(ctx, nl + 1);
                ctx->in_header = 0;
                progress = 1;
            } else {
                ctx->pending_len = 0;
                return;
            }
        }
        int em_pos = -1;
        const char* em_str = NULL;
        for (int i = 0; CHAT_END_MARKERS[i]; i++) {
            int p = chat_find_marker(ctx->pending, ctx->pending_len,
                                       CHAT_END_MARKERS[i]);
            if (p >= 0 && (em_pos < 0 || p < em_pos)) {
                em_pos = p; em_str = CHAT_END_MARKERS[i];
            }
        }
        if (em_pos >= 0) {
            if (em_pos == 0 && ctx->len == 0 && em_str &&
                strcmp(em_str, "<|im_start|>") == 0) {
                chat_accum_drop(ctx, 12);
                ctx->in_header = 1;
                progress = 1;
                continue;
            }
            if (em_pos > 0) {
                chat_accum_emit(ctx, ctx->pending, em_pos);
            }
            ctx->pending_len = 0;
            ctx->stop_requested = 1;
            return;
        }
    }

    if (!ctx->in_header && ctx->pending_len > CHAT_LOOKAHEAD) {
        int emit = ctx->pending_len - CHAT_LOOKAHEAD;
        chat_accum_emit(ctx, ctx->pending, emit);
        chat_accum_drop(ctx, emit);
    }
}

static void chat_accum_finish(chat_accum_t* ctx) {
    if (ctx->in_header) {
        ctx->pending_len = 0;
        return;
    }
    if (ctx->pending_len > 0) {
        chat_accum_emit(ctx, ctx->pending, ctx->pending_len);
        ctx->pending_len = 0;
    }
}

int tq_generate_chat_text(tq_model_t* model,
                           tq_tokenizer_t* tokenizer,
                           tq_state_t* state,
                           const char* prompt,
                           tq_gen_config_t* config,
                           char** cached_text_io,
                           int** cached_tokens_io,
                           int*  n_cached_io,
                           int*  cached_capacity_io,
                           char* output, int output_size) {
    if (!model || !state || !config || !cached_tokens_io || !n_cached_io || !cached_capacity_io || !prompt) {
        return -1;
    }

    /* --- 1. Check for text-level prefix match --- */
    int matched_text_len = 0;
    int prefix_pos = 0;  /* tokens already in KV cache that we trust */

    if (cached_text_io && *cached_text_io && *n_cached_io > 0) {
        size_t cached_len = strlen(*cached_text_io);
        if (cached_len > 0 && strncmp(*cached_text_io, prompt, cached_len) == 0) {
            matched_text_len = (int)cached_len;
            prefix_pos = *n_cached_io;
        } else if (getenv("TQ_CHAT_DEBUG")) {
            /* Find where they diverge to help diagnose */
            size_t diverge = 0;
            size_t plen = strlen(prompt);
            size_t lim = cached_len < plen ? cached_len : plen;
            while (diverge < lim && (*cached_text_io)[diverge] == prompt[diverge]) diverge++;
            fprintf(stderr,
                "[chat-text] no match: cached_len=%zu prompt_len=%zu diverge_at=%zu\n"
                "  cached[%zu..]: %.40s\n"
                "  prompt[%zu..]: %.40s\n",
                cached_len, plen, diverge,
                diverge, *cached_text_io + diverge,
                diverge, prompt + diverge);
        }
    }

    /* Wrap user callback to capture generated text into a buffer for the
     * next call's cached_text update. */
    chat_accum_t accum;
    memset(&accum, 0, sizeof(accum));
    accum.user_cb = config->on_token;
    accum.user_data = config->user_data;
    void (*orig_cb)(const char*, void*) = config->on_token;
    void*  orig_ud = config->user_data;
    config->on_token = chat_accum_callback;
    config->user_data = &accum;

    int generated = 0;

    if (matched_text_len > 0) {
        /* --- Fast path: text prefix matches --- */
        const char* suffix = prompt + matched_text_len;
        int max_prompt = model->config.max_seq_len > 0
                           ? model->config.max_seq_len : 4096;
        int* suffix_toks = (int*)malloc((size_t)max_prompt * sizeof(int));
        if (!suffix_toks) {
            config->on_token = orig_cb; config->user_data = orig_ud;
            return -1;
        }
        int n_suffix = 0;
        if (*suffix != '\0') {
            n_suffix = tq_encode(tokenizer, suffix, suffix_toks, max_prompt, 0);
            if (n_suffix < 0) n_suffix = 0;
        }

        /* Context overflow check.
         * The previous "fall back to tq_generate_continue with full
         * reprefill" approach was UNSAFE: state already had the previous
         * KV at positions [0..prefix_pos), and tq_generate_continue would
         * write new positions [0..n_new), leaving stale KV at positions
         * [n_new..prefix_pos) that subsequent generation might read.
         *
         * Correct behavior: return -2 (overflow) and let the caller
         * decide — most callers should reset the chat and retry with a
         * shorter prompt. Server can return HTTP 413, Python can raise
         * an exception, WASM can show an error to the user. */
        int reserve = config->max_tokens > 0 ? config->max_tokens : 256;
        if (prefix_pos + n_suffix + reserve + 32 > max_prompt) {
            free(suffix_toks);
            config->on_token = orig_cb; config->user_data = orig_ud;
            if (accum.buf) free(accum.buf);
            if (getenv("TQ_CHAT_DEBUG")) {
                fprintf(stderr,
                    "[chat-text] OVERFLOW prefix_pos=%d n_suffix=%d reserve=%d max=%d\n",
                    prefix_pos, n_suffix, reserve, max_prompt);
            }
            return -2;
        }

        /* Grow cache buffer */
        int needed = prefix_pos + n_suffix + reserve + 16;
        if (*cached_capacity_io < needed) {
            int new_cap = needed < 4096 ? 4096 : needed;
            int* nb = (int*)realloc(*cached_tokens_io, (size_t)new_cap * sizeof(int));
            if (!nb) { free(suffix_toks); config->on_token = orig_cb; config->user_data = orig_ud; return -1; }
            *cached_tokens_io = nb;
            *cached_capacity_io = new_cap;
        }

        /* Append suffix tokens to cache + prefill at correct positions */
        int* cached = *cached_tokens_io;
        for (int i = 0; i < n_suffix; i++) {
            cached[prefix_pos + i] = suffix_toks[i];
            tq_forward(model, state, suffix_toks[i], prefix_pos + i);
        }
        *n_cached_io = prefix_pos + n_suffix;
        free(suffix_toks);

        if (getenv("TQ_CHAT_DEBUG")) {
            fprintf(stderr, "[chat-text] FAST text_match=%d new_suffix_tokens=%d\n",
                    matched_text_len, n_suffix);
        }

        /* --- Run generation loop directly. Mirrors tq_generate_continue
         *     including rep_penalty (the fast path was silently dropping
         *     it before, leaving rep_penalty inconsistent across turns). */
        int vocab_size = model->config.vocab_size;
        int n_cached = *n_cached_io;
        int pos = n_cached;
        int prev_token = n_cached > 0 ? cached[n_cached - 1] : 1;

        float rep_penalty = config->rep_penalty;
        int rep_window = config->rep_window;
        if (rep_window > 64) rep_window = 64;
        int recent_tokens[64];
        int recent_count = 0;
        for (int i = (n_cached > rep_window ? n_cached - rep_window : 0); i < n_cached; i++) {
            recent_tokens[recent_count % 64] = cached[i];
            recent_count++;
        }
        if (rep_penalty > 1.0f) {
            int window = recent_count < rep_window ? recent_count : rep_window;
            for (int r = 0; r < window; r++) {
                int idx = (recent_count - 1 - r) % 64;
                if (idx < 0) idx += 64;
                int tok = recent_tokens[idx];
                if (tok >= 0 && tok < vocab_size && state->logits) {
                    if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                    else                         state->logits[tok] *= rep_penalty;
                }
            }
        }

        unsigned long long rng_state = config->rng_seed
            ? (unsigned long long)config->rng_seed : (unsigned long long)time(NULL);
        int next_token = tq_sample_topp(state->logits, vocab_size,
                                         config->temperature, config->top_p,
                                         &rng_state);

        int output_pos = 0;
        int eos_tokens[] = { 1, 2, 106, 128001, 128006, 128007, 128008, 128009, 248044, 248046 };
        int n_eos = sizeof(eos_tokens) / sizeof(eos_tokens[0]);

        while (generated < config->max_tokens) {
            int is_eos = 0;
            for (int e = 0; e < n_eos; e++) {
                if (next_token == eos_tokens[e]) { is_eos = 1; break; }
            }
            if (is_eos) break;
            if (pos >= model->config.max_seq_len) break;

            const char* piece = tokenizer ? tq_decode(tokenizer, prev_token, next_token) : "";
            int should_stop = 0;
            if (piece) {
                if (strstr(piece, "<|im_end|>") || strstr(piece, "<|eot_id|>") ||
                    strstr(piece, "<|start_header_id|>")) {
                    should_stop = 1; piece = "";
                }
            }
            if (should_stop) break;

            int piece_len = (int)strlen(piece ? piece : "");
            if (config->on_token && piece) config->on_token(piece, config->user_data);
            /* The chat_accum filter may have detected an end marker
             * spanning multiple tokens — break before forwarding more. */
            if (accum.stop_requested) break;
            if (output && piece && output_pos + piece_len < output_size - 1) {
                memcpy(output + output_pos, piece, piece_len);
                output_pos += piece_len;
            }

            if (n_cached < *cached_capacity_io) {
                cached[n_cached++] = next_token;
                *n_cached_io = n_cached;
            }

            prev_token = next_token;
            tq_forward(model, state, next_token, pos);
            pos++;
            generated++;

            if (rep_penalty > 1.0f) {
                int window = recent_count < rep_window ? recent_count : rep_window;
                for (int r = 0; r < window; r++) {
                    int idx = (recent_count - 1 - r) % 64;
                    if (idx < 0) idx += 64;
                    int tok = recent_tokens[idx];
                    if (tok >= 0 && tok < vocab_size) {
                        if (state->logits[tok] > 0) state->logits[tok] /= rep_penalty;
                        else                         state->logits[tok] *= rep_penalty;
                    }
                }
            }

            next_token = tq_sample_topp(state->logits, vocab_size,
                                         config->temperature, config->top_p,
                                         &rng_state);
            recent_tokens[recent_count % 64] = next_token;
            recent_count++;
        }

        if (output && output_size > 0) {
            output[output_pos < output_size ? output_pos : output_size - 1] = '\0';
        }
    } else {
        /* --- Slow path: no text-prefix match, use token LCP fallback --- */
        if (getenv("TQ_CHAT_DEBUG")) {
            fprintf(stderr, "[chat-text] SLOW no text-prefix match, full tokenize\n");
        }
        generated = tq_generate_continue(
            model, tokenizer, state, prompt, config,
            cached_tokens_io, n_cached_io, cached_capacity_io,
            output, output_size);
    }

    /* Drain the marker filter's lookahead buffer before reading
     * accum.buf for the cached_text update. Without this, the last
     * ~32 bytes of clean output would be silently lost. */
    chat_accum_finish(&accum);

    /* Restore the original callback before returning to caller */
    config->on_token = orig_cb;
    config->user_data = orig_ud;

    /* Update cached_text only if we know the KV state corresponds
     * EXACTLY to (prompt + accum.buf):
     *   - generated >= 0: generation didn't error out
     *   - !accum.tainted: every generated token was captured
     * On any failure, clear cached_text so the next call falls through
     * to the slow path with a clean slate instead of trusting bytes
     * that don't match the KV cache. */
    if (cached_text_io) {
        if (generated < 0 || accum.tainted) {
            if (*cached_text_io) { free(*cached_text_io); *cached_text_io = NULL; }
        } else {
            size_t plen = strlen(prompt);
            size_t glen = accum.len;
            size_t new_len = plen + glen;
            char* nt = (char*)malloc(new_len + 1);
            if (nt) {
                memcpy(nt, prompt, plen);
                if (glen > 0 && accum.buf) memcpy(nt + plen, accum.buf, glen);
                nt[new_len] = '\0';
                if (*cached_text_io) free(*cached_text_io);
                *cached_text_io = nt;
            } else {
                /* malloc failed → can't refresh cached_text. Clearing it
                 * is safer than leaving the previous (now stale) value. */
                if (*cached_text_io) { free(*cached_text_io); *cached_text_io = NULL; }
            }
        }
    }
    if (accum.buf) free(accum.buf);

    return generated;
}
