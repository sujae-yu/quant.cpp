/**
 * quant.cpp real-model KV cache validation benchmark
 *
 * Reads binary KV cache files (dumped from a real LLM or realistic synthetic)
 * and measures quantization quality across all 7 quant.cpp types.
 *
 * Binary format per file:
 *   Header: magic(4B) + layer_idx(4B) + num_heads(4B) + seq_len(4B) + head_dim(4B) = 20 bytes
 *   Data:   num_heads * seq_len * head_dim * sizeof(float32)
 *
 * Outputs machine-readable metrics:
 *   real_roundtrip_mse_<type>=X.XXXXXX
 *   real_attention_cosine_<type>=X.XXXXXX
 *   real_layer<N>_mse_<type>=X.XXXXXX
 *   real_layer<N>_cosine_<type>=X.XXXXXX
 */

extern "C" {
#include "turboquant/turboquant.h"

/* Reference quantize/dequantize functions */
void tq_polar_quantize_ref(const float* src, void* dst, int n);
void tq_polar_dequantize_ref(const void* src, float* dst, int n);
void tq_polar_attention_ref(const float* query, const void* kv,
                            float* scores, int seq_len, int head_dim);

void tq_qjl_quantize_ref(const float* src, void* dst, int n);
void tq_qjl_dequantize_ref(const void* src, float* dst, int n);
void tq_qjl_attention_ref(const float* query, const void* kv,
                           float* scores, int seq_len, int head_dim);

void tq_turbo_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_dequantize_ref(const void* src, float* dst, int n);
void tq_turbo_attention_ref(const float* query, const void* kv,
                             float* scores, int seq_len, int head_dim);

void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_4b_dequantize_ref(const void* src, float* dst, int n);
void tq_uniform_2b_quantize_ref(const float* src, void* dst, int n);
void tq_uniform_2b_dequantize_ref(const void* src, float* dst, int n);
}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ============================================================
 * Binary file reader
 * ============================================================ */

static const uint32_t KV_MAGIC = 0x544B5651;  /* "QVKT" */

struct KVData {
    uint32_t layer_idx;
    uint32_t num_heads;
    uint32_t seq_len;
    uint32_t head_dim;
    std::vector<float> data;  /* [num_heads][seq_len][head_dim] flat */

    /* Access element: data[h * seq_len * head_dim + s * head_dim + d] */
    const float* head_seq(int h, int s) const {
        return data.data() + (size_t)h * seq_len * head_dim + (size_t)s * head_dim;
    }
};

static bool load_kv_file(const char* path, KVData& out) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "WARNING: Cannot open %s\n", path);
        return false;
    }

    uint32_t header[5];
    if (fread(header, sizeof(uint32_t), 5, f) != 5) {
        fprintf(stderr, "WARNING: Failed to read header from %s\n", path);
        fclose(f);
        return false;
    }

    if (header[0] != KV_MAGIC) {
        fprintf(stderr, "WARNING: Bad magic in %s (got 0x%08X, expected 0x%08X)\n",
                path, header[0], KV_MAGIC);
        fclose(f);
        return false;
    }

    out.layer_idx = header[1];
    out.num_heads = header[2];
    out.seq_len   = header[3];
    out.head_dim  = header[4];

    size_t n_floats = (size_t)out.num_heads * out.seq_len * out.head_dim;
    out.data.resize(n_floats);

    size_t read = fread(out.data.data(), sizeof(float), n_floats, f);
    fclose(f);

    if (read != n_floats) {
        fprintf(stderr, "WARNING: Short read from %s (got %zu, expected %zu floats)\n",
                path, read, n_floats);
        return false;
    }

    return true;
}

/* ============================================================
 * Quantization type descriptors
 * ============================================================ */

struct QuantType {
    const char*      name;
    tq_quantize_fn   quantize;
    tq_dequantize_fn dequantize;
    size_t           block_bytes;
    int              block_elems;  /* elements per block */
};

static QuantType quant_types[] = {
    { "polar_3b",    tq_polar_quantize_ref,      tq_polar_dequantize_ref,
      sizeof(block_tq_polar),      TQ_BK },
    { "polar_4b",    tq_polar_quantize_ref,      tq_polar_dequantize_ref,
      sizeof(block_tq_polar),      TQ_BK },
    { "qjl_1b",     tq_qjl_quantize_ref,        tq_qjl_dequantize_ref,
      sizeof(block_tq_qjl),        TQ_BK },
    { "turbo_3b",   tq_turbo_quantize_ref,       tq_turbo_dequantize_ref,
      sizeof(block_tq_turbo),      TQ_BK },
    { "turbo_4b",   tq_turbo_quantize_ref,       tq_turbo_dequantize_ref,
      sizeof(block_tq_turbo),      TQ_BK },
    { "uniform_4b", tq_uniform_4b_quantize_ref,  tq_uniform_4b_dequantize_ref,
      sizeof(block_tq_uniform_4b), TQ_BK },
    { "uniform_2b", tq_uniform_2b_quantize_ref,  tq_uniform_2b_dequantize_ref,
      sizeof(block_tq_uniform_2b), TQ_BK },
};
static const int N_QUANT_TYPES = sizeof(quant_types) / sizeof(quant_types[0]);

/* ============================================================
 * Math helpers
 * ============================================================ */

static double compute_mse(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    return mse / n;
}

static double cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; i++) {
        dot    += (double)a[i] * (double)b[i];
        norm_a += (double)a[i] * (double)a[i];
        norm_b += (double)b[i] * (double)b[i];
    }
    if (norm_a < 1e-30 || norm_b < 1e-30) return 0.0;
    return dot / (sqrt(norm_a) * sqrt(norm_b));
}

/* ============================================================
 * Roundtrip MSE measurement
 * ============================================================ */

static double measure_roundtrip_mse(const KVData& kv, const QuantType& qt) {
    int head_dim = (int)kv.head_dim;
    /* Pad to block size if needed */
    int padded_dim = ((head_dim + qt.block_elems - 1) / qt.block_elems) * qt.block_elems;

    std::vector<uint8_t> qbuf(qt.block_bytes);
    std::vector<float> padded(padded_dim, 0.0f);
    std::vector<float> recon(padded_dim, 0.0f);

    double total_mse = 0.0;
    int n_vectors = 0;

    for (uint32_t h = 0; h < kv.num_heads; h++) {
        for (uint32_t s = 0; s < kv.seq_len; s++) {
            const float* src = kv.head_seq(h, s);

            /* Zero-pad to block size */
            memset(padded.data(), 0, padded_dim * sizeof(float));
            memcpy(padded.data(), src, head_dim * sizeof(float));

            /* Quantize + dequantize */
            qt.quantize(padded.data(), qbuf.data(), padded_dim);
            qt.dequantize(qbuf.data(), recon.data(), padded_dim);

            /* MSE on original dimensions only */
            total_mse += compute_mse(src, recon.data(), head_dim);
            n_vectors++;
        }
    }

    return (n_vectors > 0) ? total_mse / n_vectors : 0.0;
}

/* ============================================================
 * Attention cosine similarity measurement
 * ============================================================ */

static double measure_attention_cosine(const KVData& kv, const QuantType& qt) {
    int head_dim = (int)kv.head_dim;
    int seq_len  = (int)kv.seq_len;
    int padded_dim = ((head_dim + qt.block_elems - 1) / qt.block_elems) * qt.block_elems;

    std::vector<float> fp32_scores(seq_len);
    std::vector<float> quant_scores(seq_len);
    std::vector<uint8_t> qbuf(qt.block_bytes);
    std::vector<float> padded(padded_dim, 0.0f);
    std::vector<float> recon(padded_dim, 0.0f);

    /* Use a pseudo-random query per head (deterministic) */
    std::vector<float> query(head_dim);

    double sum_cosine = 0.0;
    int n_heads = 0;

    for (uint32_t h = 0; h < kv.num_heads; h++) {
        /* Generate deterministic query for this head */
        for (int d = 0; d < head_dim; d++) {
            uint32_t seed = (uint32_t)(h * 2654435761u + d * 340573321u);
            seed ^= seed >> 16;
            seed *= 0x45d9f3b;
            seed ^= seed >> 16;
            query[d] = ((float)(seed & 0xFFFF) / 32768.0f) - 1.0f;
        }

        /* Compute FP32 attention scores: dot(query, key[s]) for each s */
        for (int s = 0; s < seq_len; s++) {
            const float* key = kv.head_seq(h, s);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += query[d] * key[d];
            }
            fp32_scores[s] = dot;
        }

        /* Compute quantized attention scores: quantize each key, dequant, dot */
        for (int s = 0; s < seq_len; s++) {
            const float* key = kv.head_seq(h, s);

            memset(padded.data(), 0, padded_dim * sizeof(float));
            memcpy(padded.data(), key, head_dim * sizeof(float));

            qt.quantize(padded.data(), qbuf.data(), padded_dim);
            qt.dequantize(qbuf.data(), recon.data(), padded_dim);

            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += query[d] * recon[d];
            }
            quant_scores[s] = dot;
        }

        /* Cosine similarity of the score vectors */
        double cos = cosine_similarity(fp32_scores.data(), quant_scores.data(), seq_len);
        sum_cosine += cos;
        n_heads++;
    }

    return (n_heads > 0) ? sum_cosine / n_heads : 0.0;
}

/* ============================================================
 * Synthetic baseline for comparison
 * ============================================================ */

static uint32_t synth_rng = 42;
static float synth_rand() {
    synth_rng = synth_rng * 1664525u + 1013904223u;
    return ((float)(synth_rng >> 8) / (float)(1 << 24)) * 2.0f - 1.0f;
}

static void measure_synthetic_baseline(int head_dim, int seq_len) {
    /* Generate uniform random data for comparison */
    int padded_dim = ((head_dim + TQ_BK - 1) / TQ_BK) * TQ_BK;
    std::vector<float> vec(padded_dim, 0.0f);
    std::vector<float> recon(padded_dim, 0.0f);
    std::vector<float> query(head_dim);

    for (int d = 0; d < head_dim; d++) query[d] = synth_rand();

    printf("\n=== Synthetic Baseline (uniform random, dim=%d, seq=%d) ===\n",
           head_dim, seq_len);

    for (int ti = 0; ti < N_QUANT_TYPES; ti++) {
        const QuantType& qt = quant_types[ti];
        std::vector<uint8_t> qbuf(qt.block_bytes);

        /* MSE */
        double total_mse = 0.0;
        std::vector<float> fp32_scores(seq_len);
        std::vector<float> quant_scores(seq_len);

        for (int s = 0; s < seq_len; s++) {
            memset(vec.data(), 0, padded_dim * sizeof(float));
            for (int d = 0; d < head_dim; d++) vec[d] = synth_rand();

            qt.quantize(vec.data(), qbuf.data(), padded_dim);
            qt.dequantize(qbuf.data(), recon.data(), padded_dim);

            total_mse += compute_mse(vec.data(), recon.data(), head_dim);

            float dot_fp32 = 0.0f, dot_quant = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot_fp32 += query[d] * vec[d];
                dot_quant += query[d] * recon[d];
            }
            fp32_scores[s] = dot_fp32;
            quant_scores[s] = dot_quant;
        }

        double mse = total_mse / seq_len;
        double cos = cosine_similarity(fp32_scores.data(), quant_scores.data(), seq_len);
        printf("synth_mse_%s=%.6f\n", qt.name, mse);
        printf("synth_cosine_%s=%.6f\n", qt.name, cos);
    }
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char** argv) {
    /* Determine data directory */
    const char* data_dir = "spec/test_vectors/real_kv";
    if (argc > 1) {
        data_dir = argv[1];
    }

    printf("=== quant.cpp Real-Model KV Cache Validation ===\n");
    printf("Data directory: %s\n\n", data_dir);

    /* Try to load layers 0-3 */
    std::vector<KVData> layers;
    for (int l = 0; l < 4; l++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/layer%d_keys.bin", data_dir, l);

        KVData kv;
        if (load_kv_file(path, kv)) {
            printf("Loaded layer %d: %u heads x %u seq x %u dim\n",
                   l, kv.num_heads, kv.seq_len, kv.head_dim);
            layers.push_back(kv);
        }
    }

    if (layers.empty()) {
        fprintf(stderr, "ERROR: No KV cache data found in %s\n", data_dir);
        fprintf(stderr, "Run: python3 tests/reference/dump_real_kv_cache.py\n");
        return 1;
    }

    printf("\n");

    /* ============================================================
     * Per-layer, per-type quality measurement
     * ============================================================ */

    /* Accumulators for aggregate metrics */
    double agg_mse[7] = {};
    double agg_cosine[7] = {};
    int agg_count = 0;

    for (size_t li = 0; li < layers.size(); li++) {
        const KVData& kv = layers[li];
        printf("--- Layer %u (heads=%u, seq=%u, dim=%u) ---\n",
               kv.layer_idx, kv.num_heads, kv.seq_len, kv.head_dim);

        for (int ti = 0; ti < N_QUANT_TYPES; ti++) {
            const QuantType& qt = quant_types[ti];

            double mse = measure_roundtrip_mse(kv, qt);
            double cos = measure_attention_cosine(kv, qt);

            printf("real_layer%u_mse_%s=%.6f\n", kv.layer_idx, qt.name, mse);
            printf("real_layer%u_cosine_%s=%.6f\n", kv.layer_idx, qt.name, cos);

            agg_mse[ti] += mse;
            agg_cosine[ti] += cos;
        }
        agg_count++;
        printf("\n");
    }

    /* ============================================================
     * Aggregate results across all layers
     * ============================================================ */

    printf("=== Aggregate Results (averaged over %d layers) ===\n", agg_count);

    if (agg_count > 0) {
        printf("\n%-14s  %12s  %12s\n", "Type", "MSE", "Attn Cosine");
        printf("%-14s  %12s  %12s\n", "--------------", "------------", "------------");

        for (int ti = 0; ti < N_QUANT_TYPES; ti++) {
            double mse = agg_mse[ti] / agg_count;
            double cos = agg_cosine[ti] / agg_count;

            printf("%-14s  %12.6f  %12.6f\n", quant_types[ti].name, mse, cos);

            /* Machine-readable output */
            printf("real_roundtrip_mse_%s=%.6f\n", quant_types[ti].name, mse);
            printf("real_attention_cosine_%s=%.6f\n", quant_types[ti].name, cos);
        }
    }

    /* ============================================================
     * Per-layer trend analysis
     * ============================================================ */

    if (layers.size() > 1) {
        printf("\n=== Per-Layer Trend (MSE by layer depth) ===\n");
        printf("%-14s", "Type");
        for (size_t li = 0; li < layers.size(); li++) {
            printf("  Layer%-2u    ", layers[li].layer_idx);
        }
        printf("  Trend\n");

        for (int ti = 0; ti < N_QUANT_TYPES; ti++) {
            printf("%-14s", quant_types[ti].name);
            double first_mse = 0.0, last_mse = 0.0;
            for (size_t li = 0; li < layers.size(); li++) {
                double mse = measure_roundtrip_mse(layers[li], quant_types[ti]);
                printf("  %10.6f", mse);
                if (li == 0) first_mse = mse;
                if (li == layers.size() - 1) last_mse = mse;
            }
            /* Trend: does quality degrade in later layers? */
            if (first_mse > 1e-10) {
                double ratio = last_mse / first_mse;
                if (ratio > 1.1) printf("  WORSE (%.1fx)", ratio);
                else if (ratio < 0.9) printf("  BETTER (%.1fx)", ratio);
                else printf("  STABLE");
            } else {
                printf("  N/A");
            }
            printf("\n");
        }
    }

    /* ============================================================
     * Synthetic baseline comparison
     * ============================================================ */

    if (!layers.empty()) {
        measure_synthetic_baseline(
            (int)layers[0].head_dim,
            (int)layers[0].seq_len
        );
    }

    printf("\n=== Validation Complete ===\n");
    return 0;
}
