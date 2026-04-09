"""
quantcpp -- Python bindings for quant.cpp LLM inference engine.

Usage:
    from quantcpp import Model

    m = Model("model.gguf")
    answer = m.ask("What is 2+2?")
    print(answer)

    # Streaming:
    for token in m.generate("Hello"):
        print(token, end="", flush=True)

    # Context manager:
    with Model("model.gguf") as m:
        print(m.ask("Explain gravity"))
"""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("quantcpp")
except Exception:
    __version__ = "0.8.2"  # fallback for editable / source-tree imports

import os
import threading
from typing import Iterator, Optional

from quantcpp._binding import (
    QuantConfig,
    ON_TOKEN_CB,
    get_lib,
    load_model,
    new_context,
    free_ctx,
    free_model,
    version as _c_version,
)


class Model:
    """High-level Python interface to quant.cpp inference.

    Parameters
    ----------
    path : str
        Path to a GGUF model file.
    temperature : float
        Sampling temperature (default 0.7). Use 0.0 for greedy.
    top_p : float
        Nucleus sampling threshold (default 0.9).
    max_tokens : int
        Maximum tokens per generation (default 256).
    n_threads : int
        CPU thread count (default 4).
    kv_compress : int
        KV cache compression: 0=off, 1=4-bit (default), 2=delta+3-bit.

    Examples
    --------
    >>> m = Model("model.gguf")
    >>> m.ask("What is the capital of France?")
    'The capital of France is Paris.'

    >>> with Model("model.gguf", kv_compress=2) as m:
    ...     for tok in m.generate("Once upon a time"):
    ...         print(tok, end="")
    """

    def __init__(
        self,
        path: str,
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        n_threads: int = 4,
        kv_compress: int = 0,
    ):
        """
        .. note::
           ``kv_compress=1`` and ``kv_compress=2`` are temporarily disabled in
           the Python bindings (v0.8.x) — the bundled ``quant.h`` single
           header carries an older KV compression path that aborts on Llama
           architectures. The CLI ``quant`` binary uses the multi-file engine
           and works with all KV types. KV compression will be re-enabled in
           the bindings once ``quant.h`` is re-generated against the v0.8.0+
           tree (tracked as v0.8.1: WASM SIMD / un-stub turbo_kv).
        """
        if kv_compress not in (0,):
            import warnings
            warnings.warn(
                "kv_compress != 0 is not supported in the Python bindings of "
                "quantcpp 0.8.x — falling back to kv_compress=0. Use the CLI "
                "binary for KV compression until v0.8.2.",
                RuntimeWarning,
                stacklevel=2,
            )
            kv_compress = 0
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self._path = path
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._n_threads = n_threads
        self._kv_compress = kv_compress

        self._model = load_model(path)
        self._ctx = new_context(
            self._model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_threads=n_threads,
            kv_compress=kv_compress,
        )
        self._lock = threading.Lock()

    # -- Context manager ---------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    # -- Core API ----------------------------------------------------------

    def ask(self, prompt: str) -> str:
        """Send a prompt and return the full response as a string.

        Parameters
        ----------
        prompt : str
            The input prompt / question.

        Returns
        -------
        str
            The model's complete response.
        """
        self._ensure_open()
        lib = get_lib()
        import ctypes
        import sys

        with self._lock:
            ptr = lib.quant_ask(self._ctx, prompt.encode("utf-8"))

        if not ptr:
            return ""

        result = ctypes.cast(ptr, ctypes.c_char_p).value
        text = result.decode("utf-8", errors="replace") if result else ""

        # Free via the dylib's own free wrapper (added in v0.8.2). Falls back
        # to a leak if the loaded library is an older single-header that
        # doesn't export quant_free_string — preserves binary compat.
        if hasattr(lib, "quant_free_string"):
            lib.quant_free_string(ptr)

        return text

    def generate(self, prompt: str) -> Iterator[str]:
        """Stream tokens from a prompt. Yields token strings one at a time.

        Parameters
        ----------
        prompt : str
            The input prompt.

        Yields
        ------
        str
            Individual token strings as they are generated.

        Examples
        --------
        >>> for token in m.generate("Hello"):
        ...     print(token, end="", flush=True)
        """
        self._ensure_open()
        lib = get_lib()

        tokens = []
        done = threading.Event()
        error_box = [None]

        def _on_token(text_ptr, _user_data):
            if text_ptr:
                tokens.append(text_ptr.decode("utf-8", errors="replace"))

        # prevent GC of the callback during generation
        cb = ON_TOKEN_CB(_on_token)

        def _run():
            try:
                with self._lock:
                    lib.quant_generate(
                        self._ctx,
                        prompt.encode("utf-8"),
                        cb,
                        None,
                    )
            except Exception as e:
                error_box[0] = e
            finally:
                done.set()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        yielded = 0
        while not done.is_set() or yielded < len(tokens):
            if yielded < len(tokens):
                yield tokens[yielded]
                yielded += 1
            else:
                done.wait(timeout=0.01)

        # Drain remaining tokens
        while yielded < len(tokens):
            yield tokens[yielded]
            yielded += 1

        if error_box[0] is not None:
            raise error_box[0]

    def close(self) -> None:
        """Release model and context resources.

        Safe to call multiple times. Called automatically when used
        as a context manager or when garbage collected.
        """
        if hasattr(self, "_ctx") and self._ctx:
            free_ctx(self._ctx)
            self._ctx = None
        if hasattr(self, "_model") and self._model:
            free_model(self._model)
            self._model = None

    # -- Properties --------------------------------------------------------

    @property
    def path(self) -> str:
        """Path to the loaded model file."""
        return self._path

    # -- Internals ---------------------------------------------------------

    def _ensure_open(self):
        if not getattr(self, "_ctx", None) or not getattr(self, "_model", None):
            raise RuntimeError("Model has been closed")

    def __repr__(self) -> str:
        state = "open" if getattr(self, "_ctx", None) else "closed"
        return f"quantcpp.Model({self._path!r}, state={state})"


def load(path: str, **kwargs) -> Model:
    """Shorthand for Model(path, **kwargs)."""
    return Model(path, **kwargs)


__all__ = ["Model", "load", "__version__"]
