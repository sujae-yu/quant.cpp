/**
 * bench_kv_overhead.cpp -- KV cache quantization time microbenchmark
 *
 * Measures wall-clock time for:
 * - uniform_4b quantize per vector
 * - turbo_kv_3b quantize per vector
 * - turbo_kv_1b quantize per vector
 * - turbo_kv_1b attention per key
 *
 * Reports ns/vector for each operation.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <chrono>

extern "C" {
#include "turboquant/turboquant.h"

void tq_uniform_4b_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_kv_3b_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_kv_1b_quantize_ref(const float* src, void* dst, int n);
void tq_turbo_kv_1b_attention_ref(const float* query, const void* kv,
                                    float* scores, int seq_len, int head_dim);
void tq_turbo_kv_3b_attention_ref(const float* query, const void* kv,
                                    float* scores, int seq_len, int head_dim);
}

static const int DIM = 128;
static const int N_VECTORS = 10000;
static const int N_WARMUP = 100;

int main(void) {
    printf("=== quant.cpp KV Cache Quantization Overhead ===\n");
    printf("dim=%d, vectors=%d\n\n", DIM, N_VECTORS);

    /* Generate random input vectors */
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> vectors(N_VECTORS);
    for (int i = 0; i < N_VECTORS; i++) {
        vectors[i].resize(DIM);
        for (int d = 0; d < DIM; d++) vectors[i][d] = dist(rng);
    }

    std::vector<float> query(DIM);
    for (int d = 0; d < DIM; d++) query[d] = dist(rng);

    /* === Uniform 4-bit quantize === */
    {
        std::vector<block_tq_uniform_4b> blocks(N_VECTORS);

        /* Warmup */
        for (int i = 0; i < N_WARMUP; i++) {
            tq_uniform_4b_quantize_ref(vectors[i].data(), &blocks[i], DIM);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_VECTORS; i++) {
            tq_uniform_4b_quantize_ref(vectors[i].data(), &blocks[i], DIM);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        printf("  uniform_4b quantize:    %8.1f ns/vector\n", ns / N_VECTORS);
    }

    /* === TurboKV 3-bit quantize === */
    {
        std::vector<block_tq_turbo_kv_3b> blocks(N_VECTORS);

        for (int i = 0; i < N_WARMUP; i++) {
            tq_turbo_kv_3b_quantize_ref(vectors[i].data(), &blocks[i], DIM);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_VECTORS; i++) {
            tq_turbo_kv_3b_quantize_ref(vectors[i].data(), &blocks[i], DIM);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        printf("  turbo_kv_3b quantize:   %8.1f ns/vector\n", ns / N_VECTORS);
    }

    /* === TurboKV 1-bit quantize === */
    {
        std::vector<block_tq_turbo_kv_1b> blocks(N_VECTORS);

        for (int i = 0; i < N_WARMUP; i++) {
            tq_turbo_kv_1b_quantize_ref(vectors[i].data(), &blocks[i], DIM);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_VECTORS; i++) {
            tq_turbo_kv_1b_quantize_ref(vectors[i].data(), &blocks[i], DIM);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        printf("  turbo_kv_1b quantize:   %8.1f ns/vector\n", ns / N_VECTORS);
    }

    /* === TurboKV 1-bit attention per key === */
    {
        /* Pre-quantize all keys */
        std::vector<block_tq_turbo_kv_1b> blocks(N_VECTORS);
        for (int i = 0; i < N_VECTORS; i++) {
            tq_turbo_kv_1b_quantize_ref(vectors[i].data(), &blocks[i], DIM);
        }

        std::vector<float> scores(N_VECTORS);

        /* Warmup */
        tq_turbo_kv_1b_attention_ref(query.data(), blocks.data(),
                                       scores.data(), N_WARMUP, DIM);

        auto t0 = std::chrono::high_resolution_clock::now();
        tq_turbo_kv_1b_attention_ref(query.data(), blocks.data(),
                                       scores.data(), N_VECTORS, DIM);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        printf("  turbo_kv_1b attention:  %8.1f ns/key\n", ns / N_VECTORS);
    }

    /* === TurboKV 3-bit attention per key === */
    {
        std::vector<block_tq_turbo_kv_3b> blocks(N_VECTORS);
        for (int i = 0; i < N_VECTORS; i++) {
            tq_turbo_kv_3b_quantize_ref(vectors[i].data(), &blocks[i], DIM);
        }

        std::vector<float> scores(N_VECTORS);

        tq_turbo_kv_3b_attention_ref(query.data(), blocks.data(),
                                       scores.data(), N_WARMUP, DIM);

        auto t0 = std::chrono::high_resolution_clock::now();
        tq_turbo_kv_3b_attention_ref(query.data(), blocks.data(),
                                       scores.data(), N_VECTORS, DIM);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        printf("  turbo_kv_3b attention:  %8.1f ns/key\n", ns / N_VECTORS);
    }

    /* === RHT overhead (isolated) === */
    {
        std::vector<float> buf(DIM);

        /* Warmup */
        for (int i = 0; i < N_WARMUP; i++) {
            std::copy(vectors[i].begin(), vectors[i].end(), buf.begin());
            tq_rht_transform(buf.data(), DIM, 0x12345678u);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_VECTORS; i++) {
            std::copy(vectors[i % 1000].begin(), vectors[i % 1000].end(), buf.begin());
            tq_rht_transform(buf.data(), DIM, 0x12345678u);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        printf("  RHT transform:          %8.1f ns/vector (dim=%d)\n", ns / N_VECTORS, DIM);
    }

    printf("\nAll measurements include function call overhead.\n");
    printf("RHT is O(d log d) per vector; matmul is ~O(d^2) per layer.\n");

    return 0;
}
