/*
 * speculative_decode.c — Minimal speculative decoding benchmark.
 *
 * Draft: small model samples K tokens greedily.
 * Target: large model verifies via tq_forward_batch_spec(N=K),
 *         comparing per-position argmax to draft predictions.
 * Accept: prefix where draft == target argmax. +1 bonus from target.
 *
 * Usage:
 *   speculative_decode <target.gguf> <draft.gguf> "<prompt>" <n_max> <K>
 *
 * Expectation on same-family pairs (e.g. Qwen3.5-4B + Qwen3.5-0.8B):
 *   Accept rate ~70-85%. Weight-read amortization gives 2-3× decode speedup.
 */
#include "turboquant/turboquant.h"
#include "turboquant/tq_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int argmax_vec(const float* v, int n) {
    int amax = 0; float vmax = v[0];
    for (int i = 1; i < n; i++) if (v[i] > vmax) { vmax = v[i]; amax = i; }
    return amax;
}

/* Run prefill (batched if possible, else per-token) on a given model/state. */
static int do_prefill(tq_model_t* m, tq_state_t* s, const int* toks, int n) {
    if (n <= 0) return 0;
    if (n >= 2) {
        int rc = tq_forward_batch(m, s, toks, n, 0);
        if (rc == n) return n;
    }
    for (int i = 0; i < n; i++) tq_forward(m, s, toks[i], i);
    return n;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <target.gguf> <draft.gguf> \"<prompt>\" [n_max=64] [K=4]\n", argv[0]);
        return 1;
    }
    const char* target_path = argv[1];
    const char* draft_path  = argv[2];
    const char* prompt      = argv[3];
    int n_max = (argc >= 5) ? atoi(argv[4]) : 64;
    int K     = (argc >= 6) ? atoi(argv[5]) : 4;
    if (K < 1) K = 1;
    if (K > 8) K = 8;

    fprintf(stderr, "=== Speculative decoding benchmark ===\n");
    fprintf(stderr, "Target: %s\n", target_path);
    fprintf(stderr, "Draft:  %s\n", draft_path);
    fprintf(stderr, "K=%d, n_max=%d\n\n", K, n_max);

    /* Match main CLI's default: 8 threads */
    tq_set_threads(8);

    tq_model_t* target = tq_load_model(target_path);
    if (!target) { fprintf(stderr, "failed to load target\n"); return 2; }
    tq_model_t* draft  = tq_load_model(draft_path);
    if (!draft)  { fprintf(stderr, "failed to load draft\n");  return 2; }

    tq_tokenizer_t* tok = tq_load_tokenizer_from_gguf(target->gguf_ctx);
    if (!tok) { fprintf(stderr, "failed to load tokenizer\n"); return 3; }
    fprintf(stderr, "tokenizer loaded\n");

    if (target->config.vocab_size != draft->config.vocab_size) {
        fprintf(stderr, "vocab mismatch: target=%d draft=%d (must share tokenizer)\n",
                target->config.vocab_size, draft->config.vocab_size);
        return 4;
    }
    int vocab = target->config.vocab_size;

    tq_state_t* s_tgt = tq_create_state_ex(&target->config, TQ_TYPE_UNIFORM_4B, 0); /* turbo_kv_4b default */
    tq_state_t* s_drf = tq_create_state_ex(&draft->config,  TQ_TYPE_UNIFORM_4B, 0);
    if (!s_tgt || !s_drf) { fprintf(stderr, "state alloc failed\n"); return 5; }

    /* Encode prompt. */
    int prompt_toks[2048];
    int n_prompt = tq_encode(tok, prompt, prompt_toks, 2048, 1);
    if (n_prompt <= 0) { fprintf(stderr, "encode failed\n"); return 6; }
    fprintf(stderr, "Prompt: %s (%d tokens)\n---\n", prompt, n_prompt);

    /* Prefill both models. */
    double t_pre0 = now_sec();
    do_prefill(target, s_tgt, prompt_toks, n_prompt);
    do_prefill(draft,  s_drf, prompt_toks, n_prompt);
    double t_pre = now_sec() - t_pre0;

    /* ------------------------------------------------------------------
     * Baseline: target model alone, per-token decode (existing fast path).
     * This is what we're trying to beat.
     * ------------------------------------------------------------------ */
    int last_tok = prompt_toks[n_prompt - 1];
    float* logits = tq_forward(target, s_tgt, last_tok, n_prompt - 1);
    int next0 = argmax_vec(logits, vocab);

    /* Clone target state for baseline measurement by re-prefilling.
     * (simple but correct; just pay the prefill cost once.) */
    tq_state_t* s_base = tq_create_state_ex(&target->config, TQ_TYPE_COUNT, 0);
    do_prefill(target, s_base, prompt_toks, n_prompt);

    double t0 = now_sec();
    int gen_baseline = 0;
    int cur_tok = last_tok;
    int cur_pos = n_prompt - 1;
    /* First forward already done above on s_tgt; repeat on s_base for clean timing */
    {
        float* lg = tq_forward(target, s_base, cur_tok, cur_pos);
        cur_tok = argmax_vec(lg, vocab);
        cur_pos++;
        gen_baseline++;
    }
    while (gen_baseline < n_max) {
        float* lg = tq_forward(target, s_base, cur_tok, cur_pos);
        cur_tok = argmax_vec(lg, vocab);
        cur_pos++;
        gen_baseline++;
    }
    double t_base = now_sec() - t0;
    double tps_base = gen_baseline / t_base;
    fprintf(stderr, "Baseline (target only): %d tok in %.2fs → %.2f tok/s\n",
            gen_baseline, t_base, tps_base);

    /* ------------------------------------------------------------------
     * Speculative: draft K + target verify batched.
     * ------------------------------------------------------------------ */
    /* Reset draft state back to just prefill (it already has prefill from above). */
    /* s_tgt and s_drf both have prefill. s_tgt already consumed one forward
     * above for `next0`. So s_tgt is at position n_prompt; s_drf is at n_prompt-1
     * (last tq_forward was the prefill's last token or the last-token forward call).
     * To keep things clean, we re-initialize both. */
    tq_free_state(s_tgt); tq_free_state(s_drf);
    s_tgt = tq_create_state_ex(&target->config, TQ_TYPE_COUNT, 0);
    s_drf = tq_create_state_ex(&draft->config,  TQ_TYPE_COUNT, 0);
    do_prefill(target, s_tgt, prompt_toks, n_prompt);
    do_prefill(draft,  s_drf, prompt_toks, n_prompt);

    int gen_spec = 0;
    int pos_tgt = n_prompt;    /* next position to write in target */
    int pos_drf = n_prompt;    /* next position to write in draft */
    int total_drafted = 0, total_accepted = 0;

    /* Get target's prediction for the FIRST next token by running final pos logits */
    float* lg_init = tq_forward(target, s_tgt, prompt_toks[n_prompt - 1], n_prompt - 1);
    int first_next = argmax_vec(lg_init, vocab);
    /* Also advance draft to same point via a mirror forward */
    tq_forward(draft, s_drf, prompt_toks[n_prompt - 1], n_prompt - 1);

    /* current "committed" token that will feed into next spec round */
    int committed_tok = first_next;

    double t1 = now_sec();
    while (gen_spec < n_max) {
        /* Step 1: Draft K tokens starting from `committed_tok` at pos_drf.
         * Draft writes its own KV from pos_drf onward. */
        int drafts[8];
        int tok = committed_tok;
        for (int k = 0; k < K; k++) {
            float* lg = tq_forward(draft, s_drf, tok, pos_drf + k);
            tok = argmax_vec(lg, vocab);
            drafts[k] = tok;
        }

        /* Step 2: Target verifies via batched(N=K).
         * Input sequence fed to target at positions pos_tgt..pos_tgt+K-1:
         * [committed_tok, drafts[0], drafts[1], ..., drafts[K-2]]
         * i.e. each position receives the PREVIOUS round's or draft's
         * token as input. Target's argmax at position pos_tgt+i is its
         * prediction for what comes AFTER input[i]. If draft[i] == that
         * argmax, the i-th draft token is accepted. */
        int verify_input[8];
        verify_input[0] = committed_tok;
        for (int k = 1; k < K; k++) verify_input[k] = drafts[k - 1];

        int argmax_out[8];
        int rc = tq_forward_batch_spec(target, s_tgt, verify_input, K, pos_tgt, argmax_out);
        if (rc < 0) {
            /* Fallback: batched unsupported on this model — break out */
            fprintf(stderr, "\nbatched_spec unsupported (rc=%d) — stopping\n", rc);
            break;
        }

        /* Step 3: Accept prefix where drafts match target argmax, else
         * take target's argmax as the "reject" bonus. */
        int accepted = 0;
        for (int k = 0; k < K; k++) {
            if (argmax_out[k] == drafts[k]) accepted++;
            else break;
        }

        /* Emit: drafts[0..accepted-1] + target's bonus argmax_out[accepted].
         * If accepted == K, bonus = argmax_out[K-1] (target's prediction for
         * the K+1th token — "free" prediction from verification). */
        int bonus = argmax_out[accepted < K ? accepted : (K - 1)];
        int commit_count = accepted + 1;
        if (commit_count > n_max - gen_spec) commit_count = n_max - gen_spec;

        total_drafted += K;
        total_accepted += accepted;

        /* Update positions. Target KV cache from pos_tgt..pos_tgt+K-1 was
         * written. We commit only `commit_count` of those positions.
         * The rejected positions (from pos_tgt+commit_count to pos_tgt+K-1)
         * are "stale" KV entries but will be overwritten next iteration
         * when we feed new committed tokens at pos_tgt+commit_count onwards.
         * Similarly for draft. */
        pos_tgt += commit_count;
        pos_drf += commit_count;
        gen_spec += commit_count;

        /* Next iteration's committed_tok: the bonus */
        committed_tok = bonus;
    }
    double t_spec = now_sec() - t1;
    double tps_spec = gen_spec / t_spec;

    fprintf(stderr, "Speculative:            %d tok in %.2fs → %.2f tok/s\n",
            gen_spec, t_spec, tps_spec);
    if (total_drafted > 0) {
        fprintf(stderr, "Accept rate: %d/%d = %.1f%%\n",
                total_accepted, total_drafted,
                100.0 * total_accepted / total_drafted);
    }
    fprintf(stderr, "Speedup vs baseline: %.2fx\n", tps_spec / tps_base);

    tq_free_state(s_base);
    tq_free_state(s_tgt);
    tq_free_state(s_drf);
    tq_free_tokenizer(tok);
    tq_free_model(target);
    tq_free_model(draft);
    return 0;
}
