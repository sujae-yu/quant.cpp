/**
 * TurboQuant -- ROCm/HIP dispatch and initialization
 *
 * Initializes the ROCm backend, detects AMD devices, and registers
 * HIP kernel wrappers into the type traits function pointer table.
 * Manages HIP streams and events for async execution.
 *
 * Mechanically converted from CUDA (tq_cuda_dispatch.cu).
 */
#ifdef TQ_BUILD_ROCM

#include "tq_rocm_common.h"
#include <cstdio>
#include <cstring>

/* ============================================================
 * Forward declarations of ROCm kernel wrappers
 * ============================================================ */

extern "C" void tq_polar_quantize_rocm(
    const float* d_keys, void* d_out,
    int n, int head_dim, hipStream_t stream);

extern "C" void tq_polar_attention_rocm(
    const float* d_query, const void* d_keys,
    float* d_scores, int seq_len, int head_dim, hipStream_t stream);

extern "C" void tq_qjl_quantize_rocm(
    const float* d_keys, void* d_out,
    int num_keys, int emb_dim, hipStream_t stream);

extern "C" void tq_qjl_attention_rocm(
    const float* d_query, const void* d_keys,
    float* d_scores, int seq_len, int head_dim, hipStream_t stream);

extern "C" void tq_turbo_quantize_rocm(
    const float* d_keys, void* d_out,
    int n, int head_dim, hipStream_t stream);

extern "C" void tq_turbo_attention_rocm(
    const float* d_query, const void* d_keys,
    float* d_scores, int seq_len, int head_dim, hipStream_t stream);

extern "C" void tq_value_quantize_4b_rocm(
    const float* d_values, void* d_out, int n, hipStream_t stream);

extern "C" void tq_value_quantize_2b_rocm(
    const float* d_values, void* d_out, int n, hipStream_t stream);

extern "C" void tq_fused_polar_cache_write_rocm(
    const float* d_keys, void* d_cache, const int* d_slot_mapping,
    int num_tokens, int num_heads, int head_dim, hipStream_t stream);

/* ============================================================
 * ROCm backend state
 * ============================================================ */

typedef struct {
    int              initialized;
    int              device_id;
    int              compute_major;
    int              compute_minor;
    size_t           total_mem;
    hipStream_t      default_stream;
    hipStream_t      quant_stream;    /* stream for quantization ops */
    hipStream_t      attn_stream;     /* stream for attention ops */
    hipEvent_t       quant_done;      /* event to sync quant -> attn */
    char             device_name[256];
} tq_rocm_state_t;

static tq_rocm_state_t g_rocm_state = {0};

/* ============================================================
 * Wrapper functions matching tq_quantize_fn / tq_attention_fn
 * signatures from tq_types.h
 *
 * These wrappers handle device memory allocation and data
 * transfer when called with host pointers. For device-to-device
 * operation, use the _rocm functions directly.
 * ============================================================ */

static void tq_polar_quantize_rocm_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    int num_blocks = (n + TQ_BK_ROCM - 1) / TQ_BK_ROCM;
    size_t out_size = num_blocks * sizeof(tq_polar_block_d);

    hipMalloc(&d_src, n * sizeof(float));
    hipMalloc(&d_dst, out_size);
    hipMemcpy(d_src, src, n * sizeof(float), hipMemcpyHostToDevice);

    tq_polar_quantize_rocm(d_src, d_dst, n, n, g_rocm_state.default_stream);
    hipStreamSynchronize(g_rocm_state.default_stream);

    hipMemcpy(dst, d_dst, out_size, hipMemcpyDeviceToHost);
    hipFree(d_src);
    hipFree(d_dst);
}

static void tq_polar_attention_rocm_wrapper(
    const float* query, const void* kv_cache,
    float* scores, int seq_len, int head_dim)
{
    float* d_query  = NULL;
    void*  d_cache  = NULL;
    float* d_scores = NULL;
    size_t cache_size = seq_len * sizeof(tq_polar_block_d);

    hipMalloc(&d_query,  head_dim * sizeof(float));
    hipMalloc(&d_cache,  cache_size);
    hipMalloc(&d_scores, seq_len * sizeof(float));

    hipMemcpy(d_query, query, head_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_cache, kv_cache, cache_size, hipMemcpyHostToDevice);

    tq_polar_attention_rocm(d_query, d_cache, d_scores,
                            seq_len, head_dim, g_rocm_state.default_stream);
    hipStreamSynchronize(g_rocm_state.default_stream);

    hipMemcpy(scores, d_scores, seq_len * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_query);
    hipFree(d_cache);
    hipFree(d_scores);
}

static void tq_qjl_quantize_rocm_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    size_t out_size = sizeof(tq_qjl_block_d);

    hipMalloc(&d_src, n * sizeof(float));
    hipMalloc(&d_dst, out_size);
    hipMemcpy(d_src, src, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemsetAsync(d_dst, 0, out_size, g_rocm_state.default_stream);

    tq_qjl_quantize_rocm(d_src, d_dst, 1, n, g_rocm_state.default_stream);
    hipStreamSynchronize(g_rocm_state.default_stream);

    hipMemcpy(dst, d_dst, out_size, hipMemcpyDeviceToHost);
    hipFree(d_src);
    hipFree(d_dst);
}

static void tq_qjl_attention_rocm_wrapper(
    const float* query, const void* kv_cache,
    float* scores, int seq_len, int head_dim)
{
    float* d_query  = NULL;
    void*  d_cache  = NULL;
    float* d_scores = NULL;
    size_t cache_size = seq_len * sizeof(tq_qjl_block_d);

    hipMalloc(&d_query,  head_dim * sizeof(float));
    hipMalloc(&d_cache,  cache_size);
    hipMalloc(&d_scores, seq_len * sizeof(float));

    hipMemcpy(d_query, query, head_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_cache, kv_cache, cache_size, hipMemcpyHostToDevice);

    tq_qjl_attention_rocm(d_query, d_cache, d_scores,
                           seq_len, head_dim, g_rocm_state.default_stream);
    hipStreamSynchronize(g_rocm_state.default_stream);

    hipMemcpy(scores, d_scores, seq_len * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_query);
    hipFree(d_cache);
    hipFree(d_scores);
}

static void tq_turbo_quantize_rocm_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    int num_blocks = (n + TQ_BK_ROCM - 1) / TQ_BK_ROCM;
    size_t out_size = num_blocks * sizeof(tq_turbo_block_d);

    hipMalloc(&d_src, n * sizeof(float));
    hipMalloc(&d_dst, out_size);
    hipMemcpy(d_src, src, n * sizeof(float), hipMemcpyHostToDevice);

    tq_turbo_quantize_rocm(d_src, d_dst, n, n, g_rocm_state.default_stream);
    hipStreamSynchronize(g_rocm_state.default_stream);

    hipMemcpy(dst, d_dst, out_size, hipMemcpyDeviceToHost);
    hipFree(d_src);
    hipFree(d_dst);
}

static void tq_turbo_attention_rocm_wrapper(
    const float* query, const void* kv_cache,
    float* scores, int seq_len, int head_dim)
{
    float* d_query  = NULL;
    void*  d_cache  = NULL;
    float* d_scores = NULL;
    size_t cache_size = seq_len * sizeof(tq_turbo_block_d);

    hipMalloc(&d_query,  head_dim * sizeof(float));
    hipMalloc(&d_cache,  cache_size);
    hipMalloc(&d_scores, seq_len * sizeof(float));

    hipMemcpy(d_query, query, head_dim * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_cache, kv_cache, cache_size, hipMemcpyHostToDevice);

    tq_turbo_attention_rocm(d_query, d_cache, d_scores,
                            seq_len, head_dim, g_rocm_state.default_stream);
    hipStreamSynchronize(g_rocm_state.default_stream);

    hipMemcpy(scores, d_scores, seq_len * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_query);
    hipFree(d_cache);
    hipFree(d_scores);
}

static void tq_uniform_4b_quantize_rocm_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    int num_blocks = (n + TQ_BK_ROCM - 1) / TQ_BK_ROCM;
    size_t out_size = num_blocks * sizeof(tq_uniform_4b_block_d);

    hipMalloc(&d_src, n * sizeof(float));
    hipMalloc(&d_dst, out_size);
    hipMemcpy(d_src, src, n * sizeof(float), hipMemcpyHostToDevice);

    tq_value_quantize_4b_rocm(d_src, d_dst, n, g_rocm_state.default_stream);
    hipStreamSynchronize(g_rocm_state.default_stream);

    hipMemcpy(dst, d_dst, out_size, hipMemcpyDeviceToHost);
    hipFree(d_src);
    hipFree(d_dst);
}

static void tq_uniform_2b_quantize_rocm_wrapper(const float* src, void* dst, int n) {
    float* d_src = NULL;
    void*  d_dst = NULL;
    int num_blocks = (n + TQ_BK_ROCM - 1) / TQ_BK_ROCM;
    size_t out_size = num_blocks * sizeof(tq_uniform_2b_block_d);

    hipMalloc(&d_src, n * sizeof(float));
    hipMalloc(&d_dst, out_size);
    hipMemcpy(d_src, src, n * sizeof(float), hipMemcpyHostToDevice);

    tq_value_quantize_2b_rocm(d_src, d_dst, n, g_rocm_state.default_stream);
    hipStreamSynchronize(g_rocm_state.default_stream);

    hipMemcpy(dst, d_dst, out_size, hipMemcpyDeviceToHost);
    hipFree(d_src);
    hipFree(d_dst);
}

/* ============================================================
 * Backend initialization
 * ============================================================ */

extern "C" int tq_init_rocm_backend(void) {
    if (g_rocm_state.initialized) return 0;

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        fprintf(stderr, "TQ ROCm: No HIP devices found (%s)\n",
                hipGetErrorString(err));
        return -1;
    }

    /* Select device 0 by default */
    g_rocm_state.device_id = 0;
    TQ_HIP_CHECK_STATUS(hipSetDevice(0));

    /* Query device properties */
    hipDeviceProp_t prop;
    TQ_HIP_CHECK_STATUS(hipGetDeviceProperties(&prop, 0));

    g_rocm_state.compute_major = prop.major;
    g_rocm_state.compute_minor = prop.minor;
    g_rocm_state.total_mem     = prop.totalGlobalMem;
    strncpy(g_rocm_state.device_name, prop.name, sizeof(g_rocm_state.device_name) - 1);

    printf("TQ ROCm: Initialized on %s (GCN %d.%d, %.1f GB)\n",
           g_rocm_state.device_name,
           g_rocm_state.compute_major,
           g_rocm_state.compute_minor,
           (double)g_rocm_state.total_mem / (1024.0 * 1024.0 * 1024.0));

    /* Create streams and events */
    TQ_HIP_CHECK_STATUS(hipStreamCreate(&g_rocm_state.default_stream));
    TQ_HIP_CHECK_STATUS(hipStreamCreate(&g_rocm_state.quant_stream));
    TQ_HIP_CHECK_STATUS(hipStreamCreate(&g_rocm_state.attn_stream));
    TQ_HIP_CHECK_STATUS(hipEventCreate(&g_rocm_state.quant_done));

    g_rocm_state.initialized = 1;
    return 0;
}

extern "C" void tq_shutdown_rocm_backend(void) {
    if (!g_rocm_state.initialized) return;

    hipEventDestroy(g_rocm_state.quant_done);
    hipStreamDestroy(g_rocm_state.attn_stream);
    hipStreamDestroy(g_rocm_state.quant_stream);
    hipStreamDestroy(g_rocm_state.default_stream);

    memset(&g_rocm_state, 0, sizeof(g_rocm_state));
}

extern "C" int tq_rocm_is_available(void) {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    return (err == hipSuccess && count > 0) ? 1 : 0;
}

extern "C" const char* tq_rocm_device_name(void) {
    if (!g_rocm_state.initialized) return "N/A";
    return g_rocm_state.device_name;
}

extern "C" hipStream_t tq_rocm_get_stream(void) {
    return g_rocm_state.default_stream;
}

/* ============================================================
 * Dispatch table registration
 *
 * Call this after tq_init_rocm_backend() to override the default
 * CPU function pointers in TQ_TRAITS with ROCm-accelerated versions.
 *
 * NOTE: The actual traits table is defined in tq_traits.c as const.
 * In practice, the ROCm backend would use a mutable dispatch table
 * or function pointer overrides. This provides the mechanism.
 * ============================================================ */

typedef struct {
    void (*quantize)(const float*, void*, int);
    void (*attention)(const float*, const void*, float*, int, int);
} tq_rocm_dispatch_entry_t;

static tq_rocm_dispatch_entry_t g_rocm_dispatch[7] = {
    /* TQ_TYPE_POLAR_3B */
    { tq_polar_quantize_rocm_wrapper,  tq_polar_attention_rocm_wrapper },
    /* TQ_TYPE_POLAR_4B */
    { tq_polar_quantize_rocm_wrapper,  tq_polar_attention_rocm_wrapper },
    /* TQ_TYPE_QJL_1B */
    { tq_qjl_quantize_rocm_wrapper,    tq_qjl_attention_rocm_wrapper },
    /* TQ_TYPE_TURBO_3B */
    { tq_turbo_quantize_rocm_wrapper,  tq_turbo_attention_rocm_wrapper },
    /* TQ_TYPE_TURBO_4B */
    { tq_turbo_quantize_rocm_wrapper,  tq_turbo_attention_rocm_wrapper },
    /* TQ_TYPE_UNIFORM_4B */
    { tq_uniform_4b_quantize_rocm_wrapper, NULL },
    /* TQ_TYPE_UNIFORM_2B */
    { tq_uniform_2b_quantize_rocm_wrapper, NULL },
};

extern "C" void* tq_rocm_get_quantize_fn(int type_id) {
    if (type_id < 0 || type_id >= 7) return NULL;
    return (void*)g_rocm_dispatch[type_id].quantize;
}

extern "C" void* tq_rocm_get_attention_fn(int type_id) {
    if (type_id < 0 || type_id >= 7) return NULL;
    return (void*)g_rocm_dispatch[type_id].attention;
}

#endif /* TQ_BUILD_ROCM */
