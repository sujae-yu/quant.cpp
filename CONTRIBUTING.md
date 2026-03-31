# Contributing to TurboQuant.cpp

Thank you for your interest in contributing! Here's how to get started.

## Quick Setup

```bash
git clone https://github.com/quantumaikr/TurboQuant.cpp
cd TurboQuant.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DTQ_BUILD_TESTS=ON
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

## What to Work On

Check [Issues](https://github.com/quantumaikr/TurboQuant.cpp/issues) for tasks labeled `good first issue` or `help wanted`.

**High-impact areas:**
- New model architectures (Llama, Phi, Gemma)
- AVX2/AVX-512 SIMD kernels for x86
- Metal GPU compute shaders
- Long context benchmarks (8K, 32K, 128K tokens)

## Code Standards

- **C11** for core library (`src/`), **C++17** for tests
- No external dependencies in core (libc/libm/pthread only)
- Every public function needs a test
- Run tests before submitting: `ctest --test-dir build`

## Module Ownership

Each module has exclusive files to prevent merge conflicts:

| Module | Files |
|--------|-------|
| `polar` | `src/core/tq_polar.*`, `tests/test_polar.*` |
| `qjl` | `src/core/tq_qjl.*`, `tests/test_qjl.*` |
| `turbo` | `src/core/tq_turbo.*`, `tests/test_turbo.*` |
| `engine` | `src/engine/*` |
| `cache` | `src/cache/*` |
| `simd` | `src/backend/cpu/*` |

## Pull Request Process

1. Fork and create a feature branch
2. Make your changes
3. Ensure all tests pass and no new warnings
4. Submit a PR with a clear description

## Cross-Platform Checklist

Before submitting, verify:
- [ ] NEON intrinsics are inside `#ifdef __ARM_NEON` guards
- [ ] No GCC warnings (`-Wall -Wextra -Wpedantic`)
- [ ] Scalar fallback exists for all SIMD code paths

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
