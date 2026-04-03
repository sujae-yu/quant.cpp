"""
quant.cpp Python binding tests.

These tests verify the ctypes-based Python interface. They require
the quant.cpp shared library to be built and accessible.

Run:
    pytest bindings/python/tests/test_python.py -v

If the library is not in a standard path:
    QUANT_LIB_PATH=/path/to/build pytest tests/test_python.py -v
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

# Add the bindings directory to sys.path so we can import turboquant
bindings_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(bindings_dir))


def _library_available() -> bool:
    """Check if the quant.cpp shared library can be found."""
    try:
        from turboquant._core import _find_library
        return _find_library() is not None
    except Exception:
        return False


SKIP_REASON = "quant.cpp shared library not found"


class TestConstants(unittest.TestCase):
    """Test that constants are correctly defined."""

    def test_version(self):
        import turboquant
        self.assertEqual(turboquant.__version__, "0.1.0")

    def test_type_constants(self):
        import turboquant
        self.assertEqual(turboquant.POLAR_3B, 0)
        self.assertEqual(turboquant.POLAR_4B, 1)
        self.assertEqual(turboquant.QJL_1B, 2)
        self.assertEqual(turboquant.TURBO_3B, 3)
        self.assertEqual(turboquant.TURBO_4B, 4)
        self.assertEqual(turboquant.UNIFORM_4B, 5)
        self.assertEqual(turboquant.UNIFORM_2B, 6)

    def test_backend_constants(self):
        import turboquant
        self.assertEqual(turboquant.BACKEND_CPU, 0)
        self.assertEqual(turboquant.BACKEND_CUDA, 1)
        self.assertEqual(turboquant.BACKEND_METAL, 2)
        self.assertEqual(turboquant.BACKEND_AUTO, 99)


@unittest.skipUnless(_library_available(), SKIP_REASON)
class TestContext(unittest.TestCase):
    """Test quant.cppContext lifecycle."""

    def test_create_close(self):
        from turboquant import quant.cppContext
        ctx = quant.cppContext(backend=0)  # CPU
        self.assertIsNotNone(ctx)
        ctx.close()

    def test_context_manager(self):
        from turboquant import quant.cppContext
        with quant.cppContext() as ctx:
            self.assertIsNotNone(ctx)
            backend = ctx.backend
            self.assertIn(backend, [0, 1, 2])

    def test_double_close(self):
        from turboquant import quant.cppContext
        ctx = quant.cppContext()
        ctx.close()
        ctx.close()  # Should not crash


@unittest.skipUnless(_library_available(), SKIP_REASON)
class TestTypeInfo(unittest.TestCase):
    """Test type information functions."""

    def test_type_name(self):
        from turboquant import type_name, POLAR_3B, TURBO_3B, UNIFORM_4B
        self.assertEqual(type_name(POLAR_3B), "polar_3b")
        self.assertEqual(type_name(TURBO_3B), "turbo_3b")
        self.assertEqual(type_name(UNIFORM_4B), "uniform_4b")

    def test_type_bpe(self):
        from turboquant import type_bpe, UNIFORM_4B, UNIFORM_2B
        bpe_4b = type_bpe(UNIFORM_4B)
        bpe_2b = type_bpe(UNIFORM_2B)
        self.assertGreater(bpe_4b, 0.0)
        self.assertGreater(bpe_2b, 0.0)
        self.assertGreater(bpe_4b, bpe_2b)

    def test_invalid_type(self):
        from turboquant import type_name
        name = type_name(99)
        self.assertEqual(name, "unknown")


@unittest.skipUnless(_library_available(), SKIP_REASON)
class TestQuantizeKeys(unittest.TestCase):
    """Test key quantization and dequantization."""

    def _make_keys(self, n: int, head_dim: int) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, head_dim)).astype(np.float32)

    def test_quantize_uniform_4b(self):
        from turboquant import quant.cppContext, UNIFORM_4B
        keys = self._make_keys(10, 128)
        with quant.cppContext() as ctx:
            qdata = ctx.quantize_keys(keys, UNIFORM_4B)
            self.assertIsInstance(qdata, bytes)
            self.assertGreater(len(qdata), 0)

    def test_quantize_polar_4b(self):
        from turboquant import quant.cppContext, POLAR_4B
        keys = self._make_keys(10, 128)
        with quant.cppContext() as ctx:
            qdata = ctx.quantize_keys(keys, POLAR_4B)
            self.assertIsInstance(qdata, bytes)
            self.assertGreater(len(qdata), 0)

    def test_roundtrip_uniform_4b(self):
        """Quantize then dequantize should have low MSE."""
        from turboquant import quant.cppContext, UNIFORM_4B
        n, head_dim = 10, 128
        keys = self._make_keys(n, head_dim)
        with quant.cppContext() as ctx:
            qdata = ctx.quantize_keys(keys, UNIFORM_4B)
            recovered = ctx.dequantize_keys(qdata, n, head_dim, UNIFORM_4B)
            self.assertEqual(recovered.shape, (n, head_dim))
            mse = np.mean((keys - recovered) ** 2)
            # 4-bit uniform should have reasonable roundtrip error
            self.assertLess(mse, 0.1)

    def test_1d_input(self):
        """Single vector (1D) input should work."""
        from turboquant import quant.cppContext, UNIFORM_4B
        key = np.random.randn(128).astype(np.float32)
        with quant.cppContext() as ctx:
            qdata = ctx.quantize_keys(key, UNIFORM_4B)
            self.assertGreater(len(qdata), 0)

    def test_invalid_dimensions(self):
        """3D input should raise ValueError."""
        from turboquant import quant.cppContext, UNIFORM_4B
        keys = np.random.randn(2, 3, 128).astype(np.float32)
        with quant.cppContext() as ctx:
            with self.assertRaises(ValueError):
                ctx.quantize_keys(keys, UNIFORM_4B)


@unittest.skipUnless(_library_available(), SKIP_REASON)
class TestQuantizeValues(unittest.TestCase):
    """Test value quantization."""

    def test_quantize_4bit(self):
        from turboquant import quant.cppContext
        values = np.random.randn(10, 128).astype(np.float32)
        with quant.cppContext() as ctx:
            qdata = ctx.quantize_values(values, bits=4)
            self.assertIsInstance(qdata, bytes)
            self.assertGreater(len(qdata), 0)

    def test_quantize_2bit(self):
        from turboquant import quant.cppContext
        values = np.random.randn(10, 128).astype(np.float32)
        with quant.cppContext() as ctx:
            qdata = ctx.quantize_values(values, bits=2)
            self.assertIsInstance(qdata, bytes)
            # 2-bit should use less memory than 4-bit
            qdata_4 = ctx.quantize_values(values, bits=4)
            self.assertLess(len(qdata), len(qdata_4))


@unittest.skipUnless(_library_available(), SKIP_REASON)
class TestAttention(unittest.TestCase):
    """Test attention score computation."""

    def test_attention_polar(self):
        from turboquant import quant.cppContext, POLAR_4B
        head_dim = 128
        seq_len = 16
        rng = np.random.default_rng(123)
        keys = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
        query = rng.standard_normal(head_dim).astype(np.float32)

        with quant.cppContext() as ctx:
            qdata = ctx.quantize_keys(keys, POLAR_4B)
            scores = ctx.attention(query, qdata, seq_len, POLAR_4B)
            self.assertEqual(scores.shape, (seq_len,))
            self.assertTrue(np.all(np.isfinite(scores)))

    def test_attention_cosine_similarity(self):
        """Quantized attention should correlate with FP32 attention."""
        from turboquant import quant.cppContext, UNIFORM_4B
        head_dim = 128
        seq_len = 32
        rng = np.random.default_rng(456)
        keys = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
        query = rng.standard_normal(head_dim).astype(np.float32)

        # FP32 reference
        fp32_scores = keys @ query

        with quant.cppContext() as ctx:
            qdata = ctx.quantize_keys(keys, UNIFORM_4B)
            deq_keys = ctx.dequantize_keys(qdata, seq_len, head_dim, UNIFORM_4B)
            quant_scores = deq_keys @ query

            # Cosine similarity should be high
            cos_sim = np.dot(fp32_scores, quant_scores) / (
                np.linalg.norm(fp32_scores) * np.linalg.norm(quant_scores)
            )
            self.assertGreater(cos_sim, 0.95)


@unittest.skipUnless(_library_available(), SKIP_REASON)
class TestRecommendStrategy(unittest.TestCase):
    """Test strategy recommendation."""

    def test_recommend_1bit(self):
        from turboquant import quant.cppContext, QJL_1B
        with quant.cppContext() as ctx:
            rec = ctx.recommend_strategy(128, target_bits=1)
            self.assertEqual(rec, QJL_1B)

    def test_recommend_3bit(self):
        from turboquant import quant.cppContext, TURBO_3B
        with quant.cppContext() as ctx:
            rec = ctx.recommend_strategy(128, target_bits=3)
            self.assertEqual(rec, TURBO_3B)


@unittest.skipUnless(_library_available(), SKIP_REASON)
class TestModuleLevelFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    def test_module_quantize_keys(self):
        from turboquant import quantize_keys, UNIFORM_4B
        keys = np.random.randn(5, 128).astype(np.float32)
        qdata = quantize_keys(keys, UNIFORM_4B)
        self.assertIsInstance(qdata, bytes)
        self.assertGreater(len(qdata), 0)

    def test_module_dequantize_keys(self):
        from turboquant import quantize_keys, dequantize_keys, UNIFORM_4B
        keys = np.random.randn(5, 128).astype(np.float32)
        qdata = quantize_keys(keys, UNIFORM_4B)
        recovered = dequantize_keys(qdata, 5, 128, UNIFORM_4B)
        self.assertEqual(recovered.shape, (5, 128))


if __name__ == "__main__":
    unittest.main()
