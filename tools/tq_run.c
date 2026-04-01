/**
 * tq_run — TurboQuant inference CLI
 *
 * Usage:
 *   tq_run <model.safetensors> [options]
 *
 * Options:
 *   -t <tokenizer>   Path to tokenizer binary file
 *   -p <prompt>      Input prompt (default: "Hello")
 *   -n <max_tokens>  Maximum tokens to generate (default: 256)
 *   -T <temperature> Sampling temperature (default: 0.7)
 *   -P <top_p>       Top-p nucleus sampling (default: 0.9)
 *   -k <kv_type>     KV cache type: fp32, uniform_4b, uniform_2b,
 *                     polar_3b, polar_4b, turbo_3b, turbo_4b,
 *                     turbo_kv_1b, turbo_kv_3b, turbo_kv_4b (default: uniform_4b)
 *   -v <vq>          Value cache quantization: q4 (4-bit), q2 (2-bit),
 *                     or fp16 (default: fp16 when -k is set, fp32 otherwise)
 *   -j <threads>     Number of threads for matmul (default: 4)
 *   -s <seed>        Random seed (default: 42)
 *   --info           Print model info and exit
 *   -M, --memory     Print KV cache memory stats after generation
 *   --profile-kv     Profile KV activation distributions (pre/post RHT)
 *   --ppl <file>     Compute perplexity on a text file (teacher-forced)
 *   --bench-memory   Benchmark memory bandwidth (tok/s at varying context lengths)
 */

#include "turboquant/tq_engine.h"
#include "turboquant/turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Streaming token callback */
static void print_token(const char* text, void* user_data) {
    (void)user_data;
    fputs(text, stdout);
    fflush(stdout);
}

/* Parse KV type from string */
static tq_type parse_kv_type(const char* s) {
    if (!s) return TQ_TYPE_UNIFORM_4B;
    if (strcmp(s, "fp32") == 0)       return TQ_TYPE_COUNT; /* sentinel for FP32 */
    if (strcmp(s, "uniform_4b") == 0) return TQ_TYPE_UNIFORM_4B;
    if (strcmp(s, "uniform_2b") == 0) return TQ_TYPE_UNIFORM_2B;
    if (strcmp(s, "polar_3b") == 0)   return TQ_TYPE_POLAR_3B;
    if (strcmp(s, "polar_4b") == 0)   return TQ_TYPE_POLAR_4B;
    if (strcmp(s, "turbo_3b") == 0)   return TQ_TYPE_TURBO_3B;
    if (strcmp(s, "turbo_4b") == 0)   return TQ_TYPE_TURBO_4B;
    if (strcmp(s, "turbo_kv_3b") == 0) return TQ_TYPE_TURBO_KV_3B;
    if (strcmp(s, "turbo_kv_4b") == 0) return TQ_TYPE_TURBO_KV_4B;
    if (strcmp(s, "turbo_kv_1b") == 0) return TQ_TYPE_TURBO_KV_1B;
    if (strcmp(s, "qjl_1b") == 0)     return TQ_TYPE_QJL_1B;
    if (strcmp(s, "mixed_4b8") == 0)  return TQ_TYPE_MIXED_4B8;
    fprintf(stderr, "Unknown KV type: %s (using uniform_4b)\n", s);
    return TQ_TYPE_UNIFORM_4B;
}

static void print_usage(const char* prog) {
    fprintf(stderr, "TurboQuant Inference Engine\n");
    fprintf(stderr, "Usage: %s <model.safetensors> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <tokenizer>   Tokenizer binary file\n");
    fprintf(stderr, "  -p <prompt>      Input prompt (default: \"Hello\")\n");
    fprintf(stderr, "  -n <max_tokens>  Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  -T <temperature> Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  -P <top_p>       Top-p sampling (default: 0.9)\n");
    fprintf(stderr, "  -k <kv_type>     KV cache quantization type\n");
    fprintf(stderr, "  -v <vq>          Value cache quant: q4 (4-bit), q2 (2-bit), fp16 (default)\n");
    fprintf(stderr, "  -j <threads>     Number of threads for matmul (default: 4)\n");
    fprintf(stderr, "  -s <seed>        Random seed (default: 42)\n");
    fprintf(stderr, "  -q <type>        Quantize weights: q2 (2-bit Lloyd-Max, ~12x reduction),\n");
    fprintf(stderr, "                   q4 (4-bit, ~6x reduction, default),\n");
    fprintf(stderr, "                   q8 (int8, ~3.5x reduction), or none (FP32)\n");
    fprintf(stderr, "  --info           Print model info and exit\n");
    fprintf(stderr, "  -M, --memory     Print KV cache memory stats after generation\n");
    fprintf(stderr, "  --profile-kv     Profile KV activation distributions (pre/post RHT)\n");
    fprintf(stderr, "  --ppl <file>     Compute perplexity on text file (teacher-forced)\n");
    fprintf(stderr, "  --bench-memory   Benchmark memory bandwidth at varying context lengths\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Parse arguments */
    const char* model_path = NULL;
    const char* tokenizer_path = NULL;
    const char* prompt = "Hello";
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    tq_type kv_type = TQ_TYPE_UNIFORM_4B;
    int n_threads = 4;
    int quant_mode = 0;   /* 0 = none (default), 2 = Q2, 4 = Q4, 8 = Q8 */
    int value_quant_bits = 0; /* 0 = FP16/FP32 (default), 4 = Q4, 2 = Q2 */
    int info_only = 0;
    int show_memory = 0;
    int profile_kv = 0;
    const char* ppl_file = NULL;
    int bench_memory = 0;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            model_path = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-P") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            kv_type = parse_kv_type(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            const char* varg = argv[++i];
            if (strcmp(varg, "q4") == 0 || strcmp(varg, "4") == 0) {
                value_quant_bits = 4;
            } else if (strcmp(varg, "q2") == 0 || strcmp(varg, "2") == 0) {
                value_quant_bits = 2;
            } else if (strcmp(varg, "fp16") == 0 || strcmp(varg, "none") == 0) {
                value_quant_bits = 0;
            } else {
                fprintf(stderr, "Unknown value quant type: %s (using fp16)\n", varg);
                value_quant_bits = 0;
            }
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-q") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                const char* qarg = argv[++i];
                if (strcmp(qarg, "q2") == 0 || strcmp(qarg, "2") == 0) {
                    quant_mode = 2;
                } else if (strcmp(qarg, "q4") == 0 || strcmp(qarg, "4") == 0) {
                    quant_mode = 4;
                } else if (strcmp(qarg, "q8") == 0 || strcmp(qarg, "8") == 0) {
                    quant_mode = 8;
                } else if (strcmp(qarg, "none") == 0 || strcmp(qarg, "fp32") == 0) {
                    quant_mode = 0;
                } else {
                    fprintf(stderr, "Unknown quant type: %s (using q4)\n", qarg);
                    quant_mode = 4;
                }
            } else {
                quant_mode = 4;  /* -q alone defaults to Q4 */
            }
        } else if (strcmp(argv[i], "--info") == 0) {
            info_only = 1;
        } else if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--memory") == 0) {
            show_memory = 1;
        } else if (strcmp(argv[i], "--profile-kv") == 0) {
            profile_kv = 1;
        } else if (strcmp(argv[i], "--ppl") == 0 && i + 1 < argc) {
            ppl_file = argv[++i];
        } else if (strcmp(argv[i], "--bench-memory") == 0) {
            bench_memory = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: model path required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Load model */
    fprintf(stderr, "Loading model from %s...\n", model_path);
    tq_model_t* model = tq_load_model(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Print model info */
    tq_model_config_t* c = &model->config;
    fprintf(stderr, "Model: %d layers, dim=%d, heads=%d/%d, head_dim=%d, vocab=%d, inter=%d\n",
            c->n_layers, c->hidden_dim, c->n_heads, c->n_kv_heads,
            c->head_dim, c->vocab_size, c->intermediate_dim);
    fprintf(stderr, "KV cache type: %s, V quant: %s\n",
            kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32",
            value_quant_bits == 4 ? "Q4" : (value_quant_bits == 2 ? "Q2" : "FP16"));

    if (quant_mode == 2) {
        fprintf(stderr, "Quantizing weights to Q2 (2-bit Lloyd-Max codebook)...\n");
        tq_quantize_weights_q2(model);
    } else if (quant_mode == 4) {
        fprintf(stderr, "Quantizing weights to Q4 (4-bit)...\n");
        tq_quantize_weights_q4(model);
    } else if (quant_mode == 8) {
        fprintf(stderr, "Quantizing weights to Q8 (int8)...\n");
        tq_quantize_weights(model);
    }

    if (info_only) {
        tq_free_model(model);
        return 0;
    }

    /* ================================================================
     * Mode: --ppl  (Perplexity evaluation)
     * Teacher-forced: for each token position, compute cross-entropy
     * loss against the ground truth next token.
     * ================================================================ */
    if (ppl_file) {
        /* Load tokenizer first */
        tq_tokenizer_t* tok = NULL;
        if (tokenizer_path) {
            tok = tq_load_tokenizer(tokenizer_path);
        } else {
            tok = tq_load_tokenizer_from_tqm(model_path);
        }
        if (!tok) {
            fprintf(stderr, "Error: --ppl requires a tokenizer\n");
            tq_free_model(model);
            return 1;
        }

        /* Read text file */
        FILE* fp = fopen(ppl_file, "r");
        if (!fp) {
            fprintf(stderr, "Error: cannot open %s\n", ppl_file);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }
        fseek(fp, 0, SEEK_END);
        long fsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        char* text = (char*)malloc((size_t)fsize + 1);
        if (!text) {
            fclose(fp);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }
        size_t nread = fread(text, 1, (size_t)fsize, fp);
        text[nread] = '\0';
        fclose(fp);

        /* Tokenize */
        int max_tok = (int)(nread + 256);
        if (max_tok > c->max_seq_len) max_tok = c->max_seq_len;
        int* tokens = (int*)malloc((size_t)max_tok * sizeof(int));
        if (!tokens) {
            free(text);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }
        int n_tokens = tq_encode(tok, text, tokens, max_tok, 1);
        free(text);
        fprintf(stderr, "PPL evaluation: %d tokens from %s\n", n_tokens, ppl_file);

        if (n_tokens < 2) {
            fprintf(stderr, "Error: need at least 2 tokens for perplexity\n");
            free(tokens);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }

        /* Apply weight quantization */
        if (quant_mode == 2) tq_quantize_weights_q2(model);
        else if (quant_mode == 4) tq_quantize_weights_q4(model);
        else if (quant_mode == 8) tq_quantize_weights(model);

        tq_set_threads(n_threads);

        /* Create state */
        tq_state_t* state = tq_create_state_ex(&model->config, kv_type, value_quant_bits);
        if (!state) {
            fprintf(stderr, "Error: failed to allocate state\n");
            free(tokens);
            tq_free_tokenizer(tok);
            tq_free_model(model);
            return 1;
        }

        /* Teacher-forced forward: accumulate negative log-likelihood */
        double total_nll = 0.0;
        int n_eval = 0;

        struct timespec ppl_start, ppl_end;
        clock_gettime(CLOCK_MONOTONIC, &ppl_start);

        for (int i = 0; i < n_tokens - 1; i++) {
            float* logits = tq_forward(model, state, tokens[i], i);

            /* Compute log_softmax(logits)[tokens[i+1]] */
            int target = tokens[i + 1];
            if (target < 0 || target >= c->vocab_size) continue;

            /* Find max for numerical stability */
            float max_logit = logits[0];
            for (int j = 1; j < c->vocab_size; j++) {
                if (logits[j] > max_logit) max_logit = logits[j];
            }

            /* log(sum(exp(logits - max))) */
            double log_sum = 0.0;
            for (int j = 0; j < c->vocab_size; j++) {
                log_sum += exp((double)(logits[j] - max_logit));
            }
            log_sum = log(log_sum);

            /* log_softmax[target] = (logits[target] - max) - log_sum */
            double log_prob = (double)(logits[target] - max_logit) - log_sum;
            total_nll -= log_prob;
            n_eval++;

            if ((i + 1) % 50 == 0) {
                double ppl_so_far = exp(total_nll / (double)n_eval);
                fprintf(stderr, "  [%d/%d] PPL so far: %.4f\n", i + 1, n_tokens - 1, ppl_so_far);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &ppl_end);
        double ppl_elapsed = (double)(ppl_end.tv_sec - ppl_start.tv_sec)
                           + (double)(ppl_end.tv_nsec - ppl_start.tv_nsec) / 1e9;

        double perplexity = exp(total_nll / (double)n_eval);
        double avg_nll = total_nll / (double)n_eval;

        fprintf(stderr, "\n=== Perplexity Results ===\n");
        fprintf(stderr, "File:         %s\n", ppl_file);
        fprintf(stderr, "Tokens:       %d (evaluated %d)\n", n_tokens, n_eval);
        fprintf(stderr, "KV type:      %s\n", kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32");
        fprintf(stderr, "V quant:      %s\n", value_quant_bits == 4 ? "Q4" : (value_quant_bits == 2 ? "Q2" : "FP16"));
        fprintf(stderr, "Avg NLL:      %.6f\n", avg_nll);
        fprintf(stderr, "Perplexity:   %.4f\n", perplexity);
        fprintf(stderr, "Time:         %.1fs (%.1f tok/s)\n", ppl_elapsed,
                (double)n_eval / ppl_elapsed);
        fprintf(stderr, "==========================\n");

        /* Machine-parseable */
        fprintf(stderr, "PPL_CSV:%d,%.6f,%.4f\n", n_eval, avg_nll, perplexity);

        tq_free_state(state);
        free(tokens);
        tq_free_tokenizer(tok);
        tq_free_model(model);
        return 0;
    }

    /* ================================================================
     * Mode: --bench-memory  (Memory bandwidth benchmark)
     * Runs inference at varying context lengths and measures tok/s.
     * ================================================================ */
    if (bench_memory) {
        /* Load tokenizer */
        tq_tokenizer_t* tok = NULL;
        if (tokenizer_path) {
            tok = tq_load_tokenizer(tokenizer_path);
        } else {
            tok = tq_load_tokenizer_from_tqm(model_path);
        }

        /* Apply weight quantization */
        if (quant_mode == 2) tq_quantize_weights_q2(model);
        else if (quant_mode == 4) tq_quantize_weights_q4(model);
        else if (quant_mode == 8) tq_quantize_weights(model);

        tq_set_threads(n_threads);

        /* Fixed prompt token for prefill */
        int bos_token = (c->model_type == 1) ? 2 : 1;

        /* Context lengths to test */
        int ctx_lengths[] = {10, 50, 100, 200, 500};
        int n_ctx = 5;

        fprintf(stderr, "\n=== Memory Bandwidth Benchmark ===\n");
        fprintf(stderr, "KV type: %s, V quant: %s\n",
                kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32",
                value_quant_bits == 4 ? "Q4" : (value_quant_bits == 2 ? "Q2" : "FP16"));
        fprintf(stderr, "%-12s %-12s %-12s\n", "Context", "Tok/s", "Time(s)");
        fprintf(stderr, "-------- -------- --------\n");

        for (int ci = 0; ci < n_ctx; ci++) {
            int ctx = ctx_lengths[ci];
            if (ctx >= c->max_seq_len) continue;

            tq_state_t* st = tq_create_state_ex(&model->config, kv_type, value_quant_bits);
            if (!st) continue;

            /* Prefill context with BOS tokens */
            for (int i = 0; i < ctx; i++) {
                tq_forward(model, st, bos_token, i);
            }

            /* Measure decode speed: generate 20 tokens */
            int gen_count = 20;
            if (ctx + gen_count >= c->max_seq_len) {
                gen_count = c->max_seq_len - ctx - 1;
            }
            if (gen_count < 1) { tq_free_state(st); continue; }

            struct timespec bm_start, bm_end;
            clock_gettime(CLOCK_MONOTONIC, &bm_start);

            for (int g = 0; g < gen_count; g++) {
                tq_forward(model, st, bos_token, ctx + g);
            }

            clock_gettime(CLOCK_MONOTONIC, &bm_end);
            double bm_elapsed = (double)(bm_end.tv_sec - bm_start.tv_sec)
                              + (double)(bm_end.tv_nsec - bm_start.tv_nsec) / 1e9;
            double tok_s = (double)gen_count / bm_elapsed;

            fprintf(stderr, "%-12d %-12.1f %-12.3f\n", ctx, tok_s, bm_elapsed);
            fprintf(stderr, "BENCH_CSV:%d,%.2f,%.4f\n", ctx, tok_s, bm_elapsed);

            tq_free_state(st);
        }
        fprintf(stderr, "==================================\n");

        if (tok) tq_free_tokenizer(tok);
        tq_free_model(model);
        return 0;
    }

    /* Load tokenizer */
    tq_tokenizer_t* tokenizer = NULL;
    if (tokenizer_path) {
        tokenizer = tq_load_tokenizer(tokenizer_path);
        if (!tokenizer) {
            fprintf(stderr, "Warning: failed to load tokenizer, using raw IDs\n");
        }
    } else {
        /* Try to load embedded tokenizer from TQM file */
        tokenizer = tq_load_tokenizer_from_tqm(model_path);
        if (tokenizer) {
            fprintf(stderr, "Loaded embedded tokenizer from TQM file\n");
        }
    }

    /* Set thread count for matmul parallelism */
    tq_set_threads(n_threads);
    fprintf(stderr, "Threads: %d\n", tq_get_threads());

    /* ================================================================
     * Mode: --profile-kv  (KV activation distribution profiling)
     * Runs forward on prompt tokens, collects pre/post-RHT stats per layer.
     * ================================================================ */
    if (profile_kv) {
        tq_state_t* state = tq_create_state_ex(&model->config, kv_type, value_quant_bits);
        if (!state) {
            fprintf(stderr, "Error: failed to allocate state\n");
            if (tokenizer) tq_free_tokenizer(tokenizer);
            tq_free_model(model);
            return 1;
        }

        /* Enable profiling */
        state->profile_kv = 1;
        state->profile_kv_count = 0;
        state->profile_accum = (double*)calloc((size_t)c->n_layers * 8, sizeof(double));
        state->profile_stats = (float*)calloc((size_t)c->n_layers * 8, sizeof(float));
        if (!state->profile_accum || !state->profile_stats) {
            fprintf(stderr, "Error: failed to allocate profile buffers\n");
            tq_free_state(state);
            if (tokenizer) tq_free_tokenizer(tokenizer);
            tq_free_model(model);
            return 1;
        }

        /* Encode prompt */
        int ptokens[4096];
        int n_prompt = 0;
        if (tokenizer) {
            n_prompt = tq_encode(tokenizer, prompt, ptokens, 4096, 1);
        }
        if (n_prompt <= 0) {
            ptokens[0] = (c->model_type == 1) ? 2 : 1;
            n_prompt = 1;
        }

        /* Run forward on all prompt + generated tokens */
        int total_run = n_prompt + max_tokens;
        if (total_run > c->max_seq_len) total_run = c->max_seq_len;

        fprintf(stderr, "Profiling KV for %d tokens...\n", total_run);
        for (int i = 0; i < total_run; i++) {
            int tok = (i < n_prompt) ? ptokens[i] : 1; /* use token 1 for generated positions */
            float* logits = tq_forward(model, state, tok, i);
            if (i >= n_prompt && logits) {
                /* Use argmax for the next token */
                int next = tq_sample_argmax(logits, c->vocab_size);
                (void)next; /* just forward, not generating text */
            }
        }

        /* Compute and print statistics */
        int n_tok = state->profile_kv_count;
        int head_dim = c->head_dim;
        double n_samples = (double)n_tok * head_dim; /* samples per layer */

        fprintf(stderr, "\n=== KV Activation Distribution Profile ===\n");
        fprintf(stderr, "Tokens profiled: %d, head_dim: %d, samples/layer: %.0f\n",
                n_tok, head_dim, n_samples);
        fprintf(stderr, "%-8s %-10s %-10s %-10s %-10s | %-10s %-10s %-10s %-10s %-10s\n",
                "Layer", "PreMean", "PreStd", "PreSkew", "PreKurt",
                "PostMean", "PostStd", "PostSkew", "PostKurt", "KL-div");
        fprintf(stderr, "-------- ---------- ---------- ---------- ---------- | ---------- ---------- ---------- ---------- ----------\n");

        for (int l = 0; l < c->n_layers; l++) {
            double* acc = state->profile_accum + (size_t)l * 8;
            if (n_samples < 1.0) continue;

            /* Pre-RHT stats */
            double pre_mean = acc[0] / n_samples;
            double pre_var  = acc[1] / n_samples - pre_mean * pre_mean;
            double pre_std  = (pre_var > 0) ? sqrt(pre_var) : 1e-10;
            double pre_skew = (pre_std > 1e-10) ?
                (acc[2] / n_samples - 3.0 * pre_mean * pre_var - pre_mean * pre_mean * pre_mean)
                / (pre_std * pre_std * pre_std) : 0.0;
            double pre_kurt = (pre_std > 1e-10) ?
                (acc[3] / n_samples - 4.0 * pre_mean * acc[2] / n_samples
                 + 6.0 * pre_mean * pre_mean * acc[1] / n_samples
                 - 3.0 * pre_mean * pre_mean * pre_mean * pre_mean)
                / (pre_var * pre_var) : 0.0;

            /* Post-RHT stats */
            double post_mean = acc[4] / n_samples;
            double post_var  = acc[5] / n_samples - post_mean * post_mean;
            double post_std  = (post_var > 0) ? sqrt(post_var) : 1e-10;
            double post_skew = (post_std > 1e-10) ?
                (acc[6] / n_samples - 3.0 * post_mean * post_var - post_mean * post_mean * post_mean)
                / (post_std * post_std * post_std) : 0.0;
            double post_kurt = (post_std > 1e-10) ?
                (acc[7] / n_samples - 4.0 * post_mean * acc[6] / n_samples
                 + 6.0 * post_mean * post_mean * acc[5] / n_samples
                 - 3.0 * post_mean * post_mean * post_mean * post_mean)
                / (post_var * post_var) : 0.0;

            /* Approximate KL-divergence vs N(0, post_std):
             * KL(p || N(0,1)) = 0.5 * (var + mean^2 - 1 - log(var))
             * where p is assumed Gaussian with measured mean/var. */
            double post_var_for_kl = post_var > 1e-20 ? post_var : 1e-20;
            /* Normalize: we want KL against N(0, sigma) where sigma is the observed std.
             * Actually KL against standard normal: KL = 0.5 * (sigma^2 + mu^2 - 1 - ln(sigma^2))
             * But we need to standardize first. The RHT output should be ~N(0, 1/sqrt(dim)). */
            double scaled_var = post_var * (double)head_dim; /* should be ~1.0 after RHT */
            double scaled_mean = post_mean * sqrt((double)head_dim);
            double kl_div = 0.5 * (scaled_var + scaled_mean * scaled_mean - 1.0
                                   - log(scaled_var > 1e-20 ? scaled_var : 1e-20));
            if (kl_div < 0.0) kl_div = 0.0; /* numerical guard */

            fprintf(stderr, "%-8d %-10.4f %-10.4f %-10.4f %-10.2f | %-10.4f %-10.4f %-10.4f %-10.2f %-10.4f\n",
                    l, pre_mean, pre_std, pre_skew, pre_kurt,
                    post_mean, post_std, post_skew, post_kurt, kl_div);
        }
        fprintf(stderr, "==========================================\n");
        fprintf(stderr, "Target: post-RHT kurtosis ~ 3.0 (normal), KL-div < 0.05\n");

        free(state->profile_accum);
        state->profile_accum = NULL;
        free(state->profile_stats);
        state->profile_stats = NULL;
        tq_free_state(state);
        if (tokenizer) tq_free_tokenizer(tokenizer);
        tq_free_model(model);
        return 0;
    }

    /* Configure generation */
    tq_gen_config_t config = tq_default_gen_config();
    config.temperature = temperature;
    config.top_p = top_p;
    config.max_tokens = max_tokens;
    config.kv_type = kv_type;
    config.value_quant_bits = value_quant_bits;
    config.on_token = print_token;
    config.user_data = NULL;

    /* Generate */
    fprintf(stderr, "Prompt: %s\n", prompt);
    fprintf(stderr, "---\n");

    char output[65536];

    /* Measure generation time for tok/s reporting */
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    int n_generated = tq_generate(model, tokenizer, prompt, &config,
                                   output, sizeof(output));

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (double)(ts_end.tv_sec - ts_start.tv_sec)
                   + (double)(ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    fprintf(stderr, "\n---\n");
    if (n_generated > 0 && elapsed > 0.0) {
        double tok_per_sec = (double)n_generated / elapsed;
        const char* wq_name = model->use_q2_weights ? "Q2" : (model->use_q4_weights ? "Q4" : (model->use_q8_weights ? "Q8" : "FP32"));
        fprintf(stderr, "%d tokens in %.1fs (%.1f tok/s, %d threads, weights=%s, kv=%s)\n",
                n_generated, elapsed, tok_per_sec, tq_get_threads(), wq_name,
                kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32");
    } else {
        fprintf(stderr, "Generated %d tokens\n", n_generated);
    }

    /* Print KV cache memory stats if requested */
    if (show_memory && n_generated > 0) {
        int total_tokens = n_generated;

        /* FP16 KV baseline (llama.cpp default):
         * 2 (K+V) * n_layers * n_kv_heads * head_dim * 2 bytes per token */
        size_t fp16_per_token = (size_t)2 * c->n_layers * c->n_kv_heads * c->head_dim * 2;

        /* Compressed KV: keys quantized, values remain FP32.
         * K: blocks_per_head * type_size bytes per head per layer
         * V: n_kv_heads * head_dim * 4 bytes (FP32) per layer */
        size_t block_size = tq_type_block_size(kv_type);
        size_t type_size_bytes = tq_type_type_size(kv_type);
        if (block_size == 0) { block_size = TQ_BK; }
        if (type_size_bytes == 0) { type_size_bytes = sizeof(block_tq_uniform_4b); }
        size_t blocks_per_head = ((size_t)c->head_dim + block_size - 1) / block_size;

        /* K (compressed) + V (Q4/Q2/FP16/FP32) per token */
        size_t k_per_token = (size_t)c->n_layers * c->n_kv_heads
                            * blocks_per_head * type_size_bytes;
        size_t v_per_token;
        const char* v_format_name;
        if (value_quant_bits == 4) {
            /* Q4 V: 16 packed bytes + 4 byte scale per block of 32 */
            int v_blocks = (c->head_dim + 31) / 32;
            v_per_token = (size_t)c->n_layers * c->n_kv_heads * v_blocks * (16 + sizeof(float));
            v_format_name = "Q4";
        } else if (value_quant_bits == 2) {
            /* Q2 V: 8 packed bytes + 4 byte scale per block of 32 */
            int v_blocks = (c->head_dim + 31) / 32;
            v_per_token = (size_t)c->n_layers * c->n_kv_heads * v_blocks * (8 + sizeof(float));
            v_format_name = "Q2";
        } else if (kv_type < TQ_TYPE_COUNT) {
            v_per_token = (size_t)c->n_layers * c->n_kv_heads * c->head_dim * sizeof(uint16_t);
            v_format_name = "FP16";
        } else {
            v_per_token = (size_t)c->n_layers * c->n_kv_heads * c->head_dim * sizeof(float);
            v_format_name = "FP32";
        }
        size_t compressed_per_token = k_per_token + v_per_token;

        /* If kv_type is fp32 (sentinel), both key and value are FP32 */
        if (kv_type >= TQ_TYPE_COUNT) {
            compressed_per_token = (size_t)2 * c->n_layers * c->n_kv_heads
                                 * c->head_dim * sizeof(float);
        }

        /* Total bytes for all generated tokens */
        size_t total_compressed = compressed_per_token * (size_t)total_tokens;
        size_t total_fp16 = fp16_per_token * (size_t)total_tokens;

        float ratio = (total_compressed > 0) ? (float)total_fp16 / (float)total_compressed : 0.0f;

        fprintf(stderr, "\n=== KV Cache Memory Stats ===\n");
        fprintf(stderr, "Tokens in cache:      %d\n", total_tokens);
        fprintf(stderr, "Model config:         %d layers, %d kv_heads, head_dim=%d\n",
                c->n_layers, c->n_kv_heads, c->head_dim);
        fprintf(stderr, "KV type:              %s\n",
                kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32");
        fprintf(stderr, "Per-token K (%s): %.2f KB\n",
                kv_type < TQ_TYPE_COUNT ? tq_type_name(kv_type) : "fp32",
                (double)k_per_token / 1024.0);
        fprintf(stderr, "Per-token V (%s):   %.2f KB\n",
                v_format_name,
                (double)v_per_token / 1024.0);
        fprintf(stderr, "Per-token K+V total:  %.2f KB\n",
                (double)compressed_per_token / 1024.0);
        fprintf(stderr, "Per-token K+V (FP16): %.2f KB\n",
                (double)fp16_per_token / 1024.0);
        fprintf(stderr, "Total K+V:            %.2f MB\n",
                (double)total_compressed / (1024.0 * 1024.0));
        fprintf(stderr, "Total K+V (FP16):     %.2f MB\n",
                (double)total_fp16 / (1024.0 * 1024.0));
        fprintf(stderr, "Compression ratio:    %.2fx (K+V combined)\n", ratio);
        fprintf(stderr, "Memory saved:         %.2f MB\n",
                (double)(total_fp16 - total_compressed) / (1024.0 * 1024.0));
        fprintf(stderr, "=============================\n");

        /* Machine-parseable line for scripts */
        fprintf(stderr, "MEMORY_CSV:%d,%zu,%zu,%.4f\n",
                total_tokens, total_compressed, total_fp16, ratio);
    }

    /* Cleanup */
    if (tokenizer) tq_free_tokenizer(tokenizer);
    tq_free_model(model);

    return 0;
}
