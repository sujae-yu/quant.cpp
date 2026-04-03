/**
 * quant.cpp — llama.cpp integration header
 *
 * Defines GGML type registration for quant.cpp quantization types.
 * Include this in your llama.cpp build to add TQ KV cache support.
 */
#ifndef TQ_GGML_TYPE_H
#define TQ_GGML_TYPE_H

#include "turboquant/turboquant.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Register quant.cpp types with GGML type system.
 * Call once during initialization.
 * Returns TQ_OK on success.
 */
tq_status tq_ggml_register_types(void);

#ifdef __cplusplus
}
#endif

#endif /* TQ_GGML_TYPE_H */
