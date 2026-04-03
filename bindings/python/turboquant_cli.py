"""
quant.cpp Python bindings (subprocess wrapper for quant CLI)

Provides a simple, pip-install-friendly interface to the quant.cpp inference
engine without requiring C FFI or shared library loading. Communicates with
the compiled quant binary via subprocess.

For the lower-level ctypes bindings (NumPy arrays, direct quantize/dequantize),
see turboquant/_core.py instead.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional


def _find_quant() -> Optional[str]:
    """Locate the quant binary.

    Search order:
      1. TURBOQUANT_BIN environment variable
      2. ./build/quant  (relative to this file's project root)
      3. PATH lookup via shutil.which
    """
    # 1. Explicit env var
    env_path = os.environ.get("TURBOQUANT_BIN")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Relative to project root (bindings/python/ -> ../../build/)
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        project_root / "build" / "quant",
        project_root / "build" / "Release" / "quant",
        project_root / "build" / "Debug" / "quant",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # 3. PATH
    found = shutil.which("quant")
    return found


class quant.cpp:
    """High-level Python wrapper around the quant CLI.

    Uses subprocess to call the compiled quant binary, parsing its
    stdout/stderr output. No C FFI, no shared library, no NumPy required.

    Args:
        model_path: Path to the model file (.tqm, .safetensors, .gguf).
        kv_type:    KV cache quantization type (default: "turbo_kv_1b").
                    Options: fp32, uniform_4b, uniform_2b, polar_3b, polar_4b,
                    turbo_3b, turbo_4b, turbo_kv_1b, turbo_kv_3b, turbo_kv_4b.
        v_quant:    Value cache quantization: "fp16", "q4", or "q2".
        threads:    Number of threads for matrix multiplication.
        quant_path: Explicit path to quant binary (auto-detected if None).

    Example:
        tq = quant.cpp("models/qwen3.5-0.8b.tqm", kv_type="turbo_kv_1b")
        print(tq.generate("The capital of France is"))
        print(f"PPL: {tq.perplexity('test.txt')}")
        print(tq.memory_stats())
    """

    VALID_KV_TYPES = frozenset({
        "fp32", "uniform_4b", "uniform_2b",
        "polar_3b", "polar_4b",
        "turbo_3b", "turbo_4b",
        "turbo_kv_1b", "turbo_kv_3b", "turbo_kv_4b",
        "qjl_1b", "mixed_4b8",
    })

    VALID_V_QUANTS = frozenset({"fp16", "q4", "q2"})

    def __init__(
        self,
        model_path: str,
        kv_type: str = "turbo_kv_1b",
        v_quant: str = "fp16",
        threads: int = 4,
        quant_path: Optional[str] = None,
    ):
        self._model_path = str(model_path)
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        if kv_type not in self.VALID_KV_TYPES:
            raise ValueError(
                f"Invalid kv_type '{kv_type}'. "
                f"Valid options: {sorted(self.VALID_KV_TYPES)}"
            )
        self._kv_type = kv_type

        if v_quant not in self.VALID_V_QUANTS:
            raise ValueError(
                f"Invalid v_quant '{v_quant}'. "
                f"Valid options: {sorted(self.VALID_V_QUANTS)}"
            )
        self._v_quant = v_quant
        self._threads = int(threads)

        self._quant = quant_path or _find_quant()
        if self._quant is None:
            raise FileNotFoundError(
                "Could not find quant binary. Build with:\n"
                "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
                "  cmake --build build -j$(nproc)\n"
                "Or set TURBOQUANT_BIN=/path/to/quant"
            )

    def _base_cmd(self) -> list:
        """Return the base command with common flags."""
        cmd = [
            self._quant,
            self._model_path,
            "-k", self._kv_type,
            "-v", self._v_quant,
            "-j", str(self._threads),
        ]
        return cmd

    def _run(self, extra_args: list, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run quant with extra arguments and return the result."""
        cmd = self._base_cmd() + extra_args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"quant timed out after {timeout}s. "
                f"Command: {' '.join(cmd)}"
            ) from e
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"quant binary not found at: {self._quant}"
            ) from e
        return result

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = 42,
        timeout: int = 300,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt:      Input text prompt.
            max_tokens:  Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_p:       Top-p nucleus sampling threshold.
            seed:        Random seed for reproducibility.
            timeout:     Maximum seconds to wait.

        Returns:
            Generated text (stdout from quant).
        """
        result = self._run(
            [
                "-p", prompt,
                "-n", str(max_tokens),
                "-T", str(temperature),
                "-P", str(top_p),
                "-s", str(seed),
            ],
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"quant failed (exit {result.returncode}):\n{result.stderr}"
            )
        return result.stdout

    def perplexity(self, text_file: str, timeout: int = 600) -> float:
        """Compute perplexity on a text file (teacher-forced evaluation).

        Args:
            text_file: Path to a plain text file.
            timeout:   Maximum seconds to wait.

        Returns:
            Perplexity value (float).

        The quant binary outputs a machine-parseable line:
            PPL_CSV:<n_eval>,<avg_nll>,<perplexity>
        """
        if not os.path.isfile(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")

        result = self._run(["--ppl", text_file], timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"quant --ppl failed (exit {result.returncode}):\n{result.stderr}"
            )

        # Parse PPL_CSV line from stderr
        combined = result.stdout + "\n" + result.stderr
        match = re.search(r"PPL_CSV:(\d+),([\d.]+),([\d.]+)", combined)
        if match:
            return float(match.group(3))

        raise RuntimeError(
            "Could not parse perplexity from quant output.\n"
            f"stderr: {result.stderr[-500:]}"
        )

    def memory_stats(self, prompt: str = "Hello", timeout: int = 120) -> Dict[str, float]:
        """Get KV cache memory statistics.

        Runs a short generation with the -M flag and parses the output.

        Args:
            prompt:  Short prompt to trigger generation.
            timeout: Maximum seconds to wait.

        Returns:
            Dictionary with keys:
              - tokens:          Number of tokens in cache
              - compressed_bytes: Total compressed KV size (bytes)
              - fp16_bytes:      Equivalent FP16 KV size (bytes)
              - ratio:           Compression ratio
              - compressed_mb:   Compressed size in MB
              - fp16_mb:         FP16 size in MB
              - saved_mb:        Memory saved in MB

        The quant binary outputs a machine-parseable line:
            MEMORY_CSV:<tokens>,<compressed>,<fp16>,<ratio>
        """
        result = self._run(
            ["-p", prompt, "-n", "32", "-M"],
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"quant -M failed (exit {result.returncode}):\n{result.stderr}"
            )

        combined = result.stdout + "\n" + result.stderr
        match = re.search(
            r"MEMORY_CSV:(\d+),(\d+),(\d+),([\d.]+)", combined
        )
        if match:
            tokens = int(match.group(1))
            compressed = int(match.group(2))
            fp16 = int(match.group(3))
            ratio = float(match.group(4))
            return {
                "tokens": tokens,
                "compressed_bytes": compressed,
                "fp16_bytes": fp16,
                "ratio": ratio,
                "compressed_mb": compressed / (1024 * 1024),
                "fp16_mb": fp16 / (1024 * 1024),
                "saved_mb": (fp16 - compressed) / (1024 * 1024),
            }

        raise RuntimeError(
            "Could not parse memory stats from quant output.\n"
            f"stderr: {result.stderr[-500:]}"
        )

    def info(self, timeout: int = 30) -> str:
        """Print model info and return as string.

        Returns:
            Model info text from quant --info.
        """
        result = self._run(["--info"], timeout=timeout)
        return (result.stdout + result.stderr).strip()

    def __repr__(self) -> str:
        return (
            f"quant.cpp(model={self._model_path!r}, "
            f"kv_type={self._kv_type!r}, v_quant={self._v_quant!r})"
        )
