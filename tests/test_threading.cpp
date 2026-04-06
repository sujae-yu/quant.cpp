#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <cmath>
#include <atomic>
extern "C" {
#include "turboquant/turboquant.h"
}

TEST(Threading, ConcurrentQuantize) {
    tq_context_t* ctx = nullptr;
    tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
    ASSERT_EQ(status, TQ_OK);

    std::atomic<int> errors{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&ctx, &errors, t]() {
            float key[128];
            for (int i = 0; i < 128; i++) key[i] = sinf(i * 0.1f + t);

            block_tq_uniform_4b block;
            tq_status s = tq_quantize_keys(ctx, key, 1, 128,
                                           TQ_TYPE_UNIFORM_4B,
                                           &block, sizeof(block));
            if (s != TQ_OK) {
                errors.fetch_add(1);
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0);

    tq_free(ctx);
}

TEST(Threading, ConcurrentQuantizeMultipleRounds) {
    tq_context_t* ctx = nullptr;
    tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
    ASSERT_EQ(status, TQ_OK);

    std::atomic<int> errors{0};
    const int num_threads = 4;
    const int rounds = 10;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&ctx, &errors, t, rounds]() {
            for (int r = 0; r < rounds; r++) {
                float key[128];
                for (int i = 0; i < 128; i++)
                    key[i] = sinf(i * 0.1f + t * 100 + r);

                block_tq_uniform_4b block;
                tq_status s = tq_quantize_keys(ctx, key, 1, 128,
                                               TQ_TYPE_UNIFORM_4B,
                                               &block, sizeof(block));
                if (s != TQ_OK) {
                    errors.fetch_add(1);
                }
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0);

    tq_free(ctx);
}

TEST(Threading, ConcurrentAttention) {
    tq_context_t* ctx = nullptr;
    tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
    ASSERT_EQ(status, TQ_OK);

    // Use POLAR_3B which has an attention function implemented
    const int seq_len = 4;
    const int head_dim = 128;
    float keys[seq_len * head_dim];
    for (int i = 0; i < seq_len * head_dim; i++)
        keys[i] = sinf(i * 0.05f);

    size_t needed = tq_quantize_keys_size(seq_len, head_dim, TQ_TYPE_POLAR_3B);
    std::vector<uint8_t> blocks(needed);
    status = tq_quantize_keys(ctx, keys, seq_len, head_dim,
                              TQ_TYPE_POLAR_3B, blocks.data(), needed);
    ASSERT_EQ(status, TQ_OK);

    std::atomic<int> errors{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&ctx, &blocks, &errors, t, seq_len, head_dim]() {
            float query[128];
            for (int i = 0; i < 128; i++)
                query[i] = cosf(i * 0.1f + t);

            std::vector<float> scores_vec(seq_len);
            float* scores = scores_vec.data();
            tq_status s = tq_attention(ctx, query, blocks.data(),
                                       seq_len, head_dim,
                                       TQ_TYPE_POLAR_3B, scores);
            if (s != TQ_OK) {
                errors.fetch_add(1);
            }
        });
    }

    for (auto& th : threads) th.join();
    EXPECT_EQ(errors.load(), 0);

    tq_free(ctx);
}

TEST(Threading, InitFreeThreadSafe) {
    // Verify init/free don't crash when called sequentially
    for (int i = 0; i < 10; i++) {
        tq_context_t* ctx = nullptr;
        tq_status status = tq_init(&ctx, TQ_BACKEND_CPU);
        ASSERT_EQ(status, TQ_OK);
        ASSERT_NE(ctx, nullptr);
        tq_free(ctx);
    }
}
