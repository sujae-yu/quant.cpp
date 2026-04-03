"""
quant.cpp ctypes-based Python bindings.

Loads the compiled quant.cpp shared library (libturboquant.so on Linux,
libturboquant.dylib on macOS) and provides a Pythonic interface with
NumPy array support.
"""

import ctypes
import ctypes.util
import os
import sys
import platform
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.ctypeslib import ndpointer


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _find_library() -> Optional[str]:
    """Locate the quant.cpp shared library on disk."""
    # Platform-specific library name
    system = platform.system()
    if system == "Darwin":
        lib_names = ["libturboquant.dylib", "libturboquant.so"]
    elif system == "Windows":
        lib_names = ["turboquant.dll", "libturboquant.dll"]
    else:
        lib_names = ["libturboquant.so"]

    # Search paths (in priority order)
    search_dirs = []

    # 1. Environment variable
    env_path = os.environ.get("QUANT_LIB_PATH")
    if env_path:
        search_dirs.append(env_path)

    # 2. Relative to this file (../../build/)
    this_dir = Path(__file__).resolve().parent
    search_dirs.append(str(this_dir.parent.parent.parent / "build"))
    search_dirs.append(str(this_dir.parent.parent.parent / "build" / "lib"))

    # 3. System library paths
    search_dirs.append("/usr/local/lib")
    search_dirs.append("/usr/lib")

    for d in search_dirs:
        for name in lib_names:
            full = os.path.join(d, name)
            if os.path.isfile(full):
                return full

    # 4. ctypes.util fallback
    found = ctypes.util.find_library("turboquant")
    return found


def get_library_path() -> Optional[str]:
    """Return the resolved path to the quant.cpp shared library, or None."""
    return _find_library()


class quant.cppError(Exception):
    """Exception raised when a quant.cpp C API call fails."""
    pass


# ---------------------------------------------------------------------------
# C library wrapper
# ---------------------------------------------------------------------------

class _TQLib:
    """Singleton wrapper around the ctypes-loaded library."""

    _instance: Optional["_TQLib"] = None
    _lib = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return

        lib_path = _find_library()
        if lib_path is None:
            raise quant.cppError(
                "Could not find quant.cpp shared library. "
                "Build with -DBUILD_SHARED_LIBS=ON or set QUANT_LIB_PATH."
            )

        self._lib = ctypes.CDLL(lib_path)
        self._setup_prototypes()
        self._loaded = True

    def _setup_prototypes(self) -> None:
        lib = self._lib

        # tq_init(tq_context_t** ctx, tq_backend backend) -> tq_status
        lib.tq_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # tq_context_t**
            ctypes.c_int,                      # tq_backend
        ]
        lib.tq_init.restype = ctypes.c_int

        # tq_free(tq_context_t* ctx)
        lib.tq_free.argtypes = [ctypes.c_void_p]
        lib.tq_free.restype = None

        # tq_get_backend(const tq_context_t* ctx) -> tq_backend
        lib.tq_get_backend.argtypes = [ctypes.c_void_p]
        lib.tq_get_backend.restype = ctypes.c_int

        # tq_quantize_keys(ctx, keys, n, head_dim, type, out, out_size) -> status
        lib.tq_quantize_keys.argtypes = [
            ctypes.c_void_p,                    # ctx
            ctypes.POINTER(ctypes.c_float),     # keys
            ctypes.c_int,                       # n
            ctypes.c_int,                       # head_dim
            ctypes.c_int,                       # type
            ctypes.c_void_p,                    # out
            ctypes.c_size_t,                    # out_size
        ]
        lib.tq_quantize_keys.restype = ctypes.c_int

        # tq_quantize_keys_size(n, head_dim, type) -> size_t
        lib.tq_quantize_keys_size.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.tq_quantize_keys_size.restype = ctypes.c_size_t

        # tq_quantize_values(ctx, values, n, head_dim, bits, out, out_size)
        lib.tq_quantize_values.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.tq_quantize_values.restype = ctypes.c_int

        # tq_quantize_values_size(n, head_dim, bits) -> size_t
        lib.tq_quantize_values_size.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.tq_quantize_values_size.restype = ctypes.c_size_t

        # tq_dequantize_keys(ctx, quantized, n, head_dim, type, out) -> status
        lib.tq_dequantize_keys.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.tq_dequantize_keys.restype = ctypes.c_int

        # tq_attention(ctx, query, kv_cache, seq_len, head_dim, type, scores)
        lib.tq_attention.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.tq_attention.restype = ctypes.c_int

        # tq_type_name(type) -> const char*
        lib.tq_type_name.argtypes = [ctypes.c_int]
        lib.tq_type_name.restype = ctypes.c_char_p

        # tq_type_bpe(type) -> float
        lib.tq_type_bpe.argtypes = [ctypes.c_int]
        lib.tq_type_bpe.restype = ctypes.c_float

        # tq_status_string(status) -> const char*
        lib.tq_status_string.argtypes = [ctypes.c_int]
        lib.tq_status_string.restype = ctypes.c_char_p

        # tq_recommend_strategy(head_dim, target_bits, quality) -> tq_type
        lib.tq_recommend_strategy.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_float
        ]
        lib.tq_recommend_strategy.restype = ctypes.c_int

    @property
    def lib(self):
        if not self._loaded:
            self.load()
        return self._lib


def _get_lib():
    """Get the loaded library instance."""
    tqlib = _TQLib()
    tqlib.load()
    return tqlib.lib


def _check_status(status: int, operation: str = ""):
    """Raise quant.cppError if status is not TQ_OK (0)."""
    if status != 0:
        lib = _get_lib()
        msg = lib.tq_status_string(status)
        msg_str = msg.decode("utf-8") if msg else f"error code {status}"
        raise quant.cppError(f"{operation}: {msg_str}" if operation else msg_str)


# ---------------------------------------------------------------------------
# Utility functions (module-level)
# ---------------------------------------------------------------------------

def type_name(tq_type: int) -> str:
    """Return the human-readable name for a quant.cpp type."""
    lib = _get_lib()
    result = lib.tq_type_name(tq_type)
    return result.decode("utf-8") if result else "unknown"


def type_bpe(tq_type: int) -> float:
    """Return bits-per-element for a quant.cpp type."""
    lib = _get_lib()
    return float(lib.tq_type_bpe(tq_type))


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_ctx: Optional["quant.cppContext"] = None


def _get_default_ctx() -> "quant.cppContext":
    global _default_ctx
    if _default_ctx is None:
        _default_ctx = quant.cppContext()
    return _default_ctx


def quantize_keys(keys: np.ndarray, tq_type: int = 3) -> bytes:
    """Quantize key vectors using a default context.

    Args:
        keys: FP32 array of shape [n, head_dim] or [head_dim].
        tq_type: Quantization type (default: TURBO_3B = 3).

    Returns:
        Quantized data as bytes.
    """
    return _get_default_ctx().quantize_keys(keys, tq_type)


def quantize_values(values: np.ndarray, bits: int = 4) -> bytes:
    """Quantize value vectors using a default context.

    Args:
        values: FP32 array of shape [n, head_dim] or [head_dim].
        bits: Quantization bits (2 or 4, default: 4).

    Returns:
        Quantized data as bytes.
    """
    return _get_default_ctx().quantize_values(values, bits)


def dequantize_keys(quantized: bytes, n: int, head_dim: int,
                    tq_type: int = 3) -> np.ndarray:
    """Dequantize keys back to FP32 using a default context.

    Args:
        quantized: Quantized key data.
        n: Number of key vectors.
        head_dim: Dimension per key.
        tq_type: Quantization type used.

    Returns:
        FP32 array of shape [n, head_dim].
    """
    return _get_default_ctx().dequantize_keys(quantized, n, head_dim, tq_type)


def attention(query: np.ndarray, kv_cache: bytes,
              seq_len: int, tq_type: int = 3) -> np.ndarray:
    """Compute attention scores using a default context.

    Args:
        query: FP32 query vector [head_dim].
        kv_cache: Quantized key cache.
        seq_len: Number of cached keys.
        tq_type: Quantization type.

    Returns:
        FP32 attention scores [seq_len].
    """
    return _get_default_ctx().attention(query, kv_cache, seq_len, tq_type)


# ---------------------------------------------------------------------------
# quant.cppContext class
# ---------------------------------------------------------------------------

class quant.cppContext:
    """High-level Python wrapper around the quant.cpp C context.

    Manages the lifecycle of a tq_context_t handle and provides methods
    for quantizing keys/values and computing attention scores.

    Usage:
        ctx = quant.cppContext(backend=BACKEND_CPU)
        quantized = ctx.quantize_keys(keys_np, tq_type=TURBO_3B)
        scores = ctx.attention(query_np, quantized, seq_len=100, tq_type=TURBO_3B)
        ctx.close()

    Or as a context manager:
        with quant.cppContext() as ctx:
            quantized = ctx.quantize_keys(keys_np)
    """

    def __init__(self, backend: int = 99):
        """Create a quant.cpp context.

        Args:
            backend: Backend to use. One of:
                     BACKEND_CPU (0), BACKEND_CUDA (1),
                     BACKEND_METAL (2), BACKEND_AUTO (99, default).
        """
        self._lib = _get_lib()
        self._ctx = ctypes.c_void_p()
        status = self._lib.tq_init(ctypes.byref(self._ctx), backend)
        _check_status(status, "tq_init")

    def close(self) -> None:
        """Release the underlying C context."""
        if self._ctx:
            self._lib.tq_free(self._ctx)
            self._ctx = ctypes.c_void_p()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def backend(self) -> int:
        """Return the active backend."""
        return self._lib.tq_get_backend(self._ctx)

    def quantize_keys(self, keys: np.ndarray, tq_type: int = 3) -> bytes:
        """Quantize FP32 key vectors.

        Args:
            keys: NumPy array of shape [n, head_dim] or [head_dim].
                  Will be converted to float32 C-contiguous if needed.
            tq_type: Quantization type (0-6). Default TURBO_3B (3).

        Returns:
            Quantized data as a bytes object.
        """
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)
        if keys.ndim != 2:
            raise ValueError(f"keys must be 1D or 2D, got {keys.ndim}D")

        n, head_dim = keys.shape
        out_size = self._lib.tq_quantize_keys_size(n, head_dim, tq_type)
        if out_size == 0:
            raise quant.cppError(f"Invalid type {tq_type} or dimensions")

        out_buf = (ctypes.c_uint8 * out_size)()
        keys_ptr = keys.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        status = self._lib.tq_quantize_keys(
            self._ctx, keys_ptr, n, head_dim, tq_type,
            ctypes.cast(out_buf, ctypes.c_void_p), out_size
        )
        _check_status(status, "tq_quantize_keys")

        return bytes(out_buf)

    def quantize_values(self, values: np.ndarray, bits: int = 4) -> bytes:
        """Quantize FP32 value vectors.

        Args:
            values: NumPy array of shape [n, head_dim] or [head_dim].
            bits: Quantization bits (2 or 4).

        Returns:
            Quantized data as bytes.
        """
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(f"values must be 1D or 2D, got {values.ndim}D")

        n, head_dim = values.shape
        out_size = self._lib.tq_quantize_values_size(n, head_dim, bits)
        if out_size == 0:
            raise quant.cppError(f"Invalid bits={bits} or dimensions")

        out_buf = (ctypes.c_uint8 * out_size)()
        values_ptr = values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        status = self._lib.tq_quantize_values(
            self._ctx, values_ptr, n, head_dim, bits,
            ctypes.cast(out_buf, ctypes.c_void_p), out_size
        )
        _check_status(status, "tq_quantize_values")

        return bytes(out_buf)

    def dequantize_keys(self, quantized: bytes, n: int, head_dim: int,
                        tq_type: int = 3) -> np.ndarray:
        """Dequantize keys back to FP32.

        Args:
            quantized: Quantized key data from quantize_keys().
            n: Number of key vectors.
            head_dim: Dimension per key vector.
            tq_type: Quantization type that was used.

        Returns:
            NumPy FP32 array of shape [n, head_dim].
        """
        out = np.empty((n, head_dim), dtype=np.float32)
        q_buf = (ctypes.c_uint8 * len(quantized)).from_buffer_copy(quantized)

        status = self._lib.tq_dequantize_keys(
            self._ctx,
            ctypes.cast(q_buf, ctypes.c_void_p),
            n, head_dim, tq_type,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        _check_status(status, "tq_dequantize_keys")

        return out

    def attention(self, query: np.ndarray, kv_cache: bytes,
                  seq_len: int, tq_type: int = 3) -> np.ndarray:
        """Compute attention scores from quantized KV cache.

        Args:
            query: FP32 query vector [head_dim].
            kv_cache: Quantized key cache data.
            seq_len: Number of cached key vectors.
            tq_type: Quantization type used for the cache.

        Returns:
            FP32 attention scores array [seq_len].
        """
        query = np.ascontiguousarray(query, dtype=np.float32).ravel()
        head_dim = query.shape[0]

        scores = np.empty(seq_len, dtype=np.float32)
        kv_buf = (ctypes.c_uint8 * len(kv_cache)).from_buffer_copy(kv_cache)

        status = self._lib.tq_attention(
            self._ctx,
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(kv_buf, ctypes.c_void_p),
            seq_len, head_dim, tq_type,
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        _check_status(status, "tq_attention")

        return scores

    def recommend_strategy(self, head_dim: int, target_bits: int,
                           quality_threshold: float = 0.99) -> int:
        """Recommend a quantization strategy.

        Args:
            head_dim: Dimension per attention head.
            target_bits: Target bits per element.
            quality_threshold: Minimum quality threshold (0-1).

        Returns:
            Recommended tq_type integer.
        """
        return self._lib.tq_recommend_strategy(
            head_dim, target_bits, ctypes.c_float(quality_threshold)
        )
