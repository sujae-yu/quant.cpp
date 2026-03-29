"""
TurboQuant -- Cross-platform KV cache compression for LLM inference.

Python bindings using ctypes to interface with the C library.

Usage:
    import turboquant

    ctx = turboquant.TurboQuantContext()
    quantized = ctx.quantize_keys(keys_np, type=turboquant.TURBO_3B)
    scores = ctx.attention(query_np, quantized, type=turboquant.TURBO_3B)
    ctx.close()
"""

__version__ = "0.1.0"
__author__ = "TurboQuant Contributors"

from turboquant._core import (
    TurboQuantContext,
    TurboQuantError,
    quantize_keys,
    quantize_values,
    dequantize_keys,
    attention,
    type_name,
    type_bpe,
    get_library_path,
)

# Quantization type constants
POLAR_3B   = 0
POLAR_4B   = 1
QJL_1B     = 2
TURBO_3B   = 3
TURBO_4B   = 4
UNIFORM_4B = 5
UNIFORM_2B = 6

# Backend constants
BACKEND_CPU   = 0
BACKEND_CUDA  = 1
BACKEND_METAL = 2
BACKEND_AUTO  = 99

class TurboQuant:
    """Simplified TurboQuant API for quick experimentation.

    Wraps TurboQuantContext with a convenient interface matching the
    type constants defined on the class itself.
    """

    # Type constants
    POLAR_3B = POLAR_3B
    POLAR_4B = POLAR_4B
    QJL_1B = QJL_1B
    TURBO_3B = TURBO_3B
    TURBO_4B = TURBO_4B
    UNIFORM_4B = UNIFORM_4B
    UNIFORM_2B = UNIFORM_2B

    TYPE_NAMES = {
        0: "polar_3b", 1: "polar_4b", 2: "qjl_1b",
        3: "turbo_3b", 4: "turbo_4b", 5: "uniform_4b", 6: "uniform_2b",
    }

    def __init__(self, backend="cpu"):
        import numpy as _np
        self._np = _np
        backend_map = {"cpu": 0, "cuda": 1, "metal": 2, "auto": 99}
        backend_int = backend_map.get(backend, 0) if isinstance(backend, str) else backend
        self._ctx = TurboQuantContext(backend=backend_int)

    def quantize_keys(self, keys, qtype=UNIFORM_4B):
        """Quantize key vectors. keys: [n, head_dim] float32 array."""
        return self._ctx.quantize_keys(keys, qtype)

    def dequantize_keys(self, quantized, n, head_dim, qtype=UNIFORM_4B):
        """Dequantize back to float32. Returns [n, head_dim] array."""
        return self._ctx.dequantize_keys(quantized, n, head_dim, qtype)

    def attention(self, query, quantized_keys, seq_len, head_dim, qtype=UNIFORM_4B):
        """Compute attention scores. Returns [seq_len] array."""
        return self._ctx.attention(query, quantized_keys, seq_len, qtype)

    def type_name(self, qtype):
        return type_name(qtype)

    def type_bpe(self, qtype):
        return type_bpe(qtype)

    def compression_ratio(self, qtype):
        return 32.0 / self.type_bpe(qtype)

    def close(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._ctx.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return "TurboQuant(backend=cpu)"


__all__ = [
    "TurboQuant",
    "TurboQuantContext",
    "TurboQuantError",
    "quantize_keys",
    "quantize_values",
    "dequantize_keys",
    "attention",
    "type_name",
    "type_bpe",
    "get_library_path",
    "POLAR_3B",
    "POLAR_4B",
    "QJL_1B",
    "TURBO_3B",
    "TURBO_4B",
    "UNIFORM_4B",
    "UNIFORM_2B",
    "BACKEND_CPU",
    "BACKEND_CUDA",
    "BACKEND_METAL",
    "BACKEND_AUTO",
]
