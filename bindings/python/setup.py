"""
quantcpp build script.

Compiles quant.h into a shared library at install time using the system
C compiler. No build dependencies beyond a working C toolchain.

    pip install .           # build + install
    pip install -e .        # editable / development install
    python setup.py build   # just compile the shared library
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent          # quant.cpp repo root (only exists in dev tree)
PKG_DIR = HERE / "quantcpp"

# Bundled header location (always inside the package — required for sdist).
# Source-of-truth chain: ../../quant.h (dev tree) → ./quant.h (CI staging /
# sdist payload) → quantcpp/_quant.h (final, used by compiler). The middle
# location lives at the package-dir root because it must be NOT excluded by
# any .gitignore so cibuildwheel's isolated source copies pick it up.
BUNDLED_HEADER = PKG_DIR / "_quant.h"
STAGED_HEADER  = HERE / "quant.h"          # CI / sdist payload
QUANT_H_DEV    = PROJECT_ROOT / "quant.h"  # dev tree fallback

def _ensure_bundled_header() -> Path:
    """Ensure quantcpp/_quant.h exists. Try the dev tree, then the staged
    package-dir copy used by sdists and CI."""
    sources = [s for s in (QUANT_H_DEV, STAGED_HEADER) if s.is_file()]
    if sources:
        src = sources[0]
        if (not BUNDLED_HEADER.exists() or
                src.stat().st_mtime > BUNDLED_HEADER.stat().st_mtime):
            shutil.copyfile(src, BUNDLED_HEADER)
            print(f"[quantcpp] Bundled {src} -> {BUNDLED_HEADER}")
    if not BUNDLED_HEADER.is_file():
        raise FileNotFoundError(
            f"Bundled header missing: {BUNDLED_HEADER}. Looked in: "
            f"{QUANT_H_DEV} (dev tree), {STAGED_HEADER} (package-dir staging). "
            "If installing from sdist, the tarball is malformed."
        )
    return BUNDLED_HEADER

# The tiny C file that triggers the implementation. Uses the bundled header
# name so it works identically in dev tree and in installed sdist.
_IMPL_C = """
#define QUANT_IMPLEMENTATION
#include "_quant.h"
"""


def _lib_name() -> str:
    """Return the platform-appropriate shared library filename."""
    if sys.platform == "darwin":
        return "libquant.dylib"
    elif sys.platform == "win32":
        return "quant.dll"
    else:
        return "libquant.so"


def _find_cc() -> str:
    """Find a C compiler."""
    for cc in [os.environ.get("CC"), "cc", "gcc", "clang"]:
        if cc and shutil.which(cc):
            return cc
    raise RuntimeError(
        "No C compiler found. Install gcc or clang, or set the CC "
        "environment variable."
    )


def _compile_shared_lib(output_dir: Path) -> Path:
    """Compile the bundled _quant.h into a shared library."""
    # Make sure the bundled header is in place. In the dev tree this copies
    # from ../../quant.h; in an installed sdist it's already there.
    header = _ensure_bundled_header()

    output_dir.mkdir(parents=True, exist_ok=True)
    lib_name = _lib_name()
    lib_path = output_dir / lib_name

    # Place the impl .c next to the header so the include path is just `.`
    impl_c = header.parent / "_quant_impl.c"
    impl_c.write_text(_IMPL_C)

    cc = _find_cc()
    cmd = [cc]

    # Shared library flags
    if sys.platform == "darwin":
        cmd += ["-dynamiclib", "-install_name", f"@rpath/{lib_name}"]
    elif sys.platform == "win32":
        cmd += ["-shared"]
    else:
        cmd += ["-shared", "-fPIC"]

    cmd += [
        "-O2",
        "-fPIC",
        "-I", str(header.parent),
        str(impl_c),
        "-o", str(lib_path),
        "-lm",
    ]

    # pthreads (not needed on Windows)
    if sys.platform != "win32":
        cmd.append("-lpthread")

    # Suppress common warnings from single-header builds
    cmd += ["-w"]

    print(f"[quantcpp] Compiling {header.name} -> {lib_name}")
    print(f"[quantcpp] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[quantcpp] STDOUT: {result.stdout}", file=sys.stderr)
        print(f"[quantcpp] STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(
            f"Compilation failed (exit code {result.returncode}).\n"
            f"stderr: {result.stderr}"
        )

    print(f"[quantcpp] Built {lib_path} ({lib_path.stat().st_size:,} bytes)")
    return lib_path


# ---------------------------------------------------------------------------
# Custom build command
# ---------------------------------------------------------------------------

class BuildWithCompile(build_py):
    """Extend build_py to compile the shared library into the package."""

    def run(self):
        super().run()

        # Destination inside the built package
        pkg_dir = Path(self.build_lib) / "quantcpp"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        _compile_shared_lib(pkg_dir)


# ---------------------------------------------------------------------------
# Also compile for editable installs (pip install -e .)
# ---------------------------------------------------------------------------

class BuildInPlace(build_py):
    """For editable installs, compile into the source tree."""

    def run(self):
        super().run()

        # Build into the source quantcpp/ directory
        pkg_dir = HERE / "quantcpp"
        lib_path = pkg_dir / _lib_name()
        if not lib_path.exists():
            _compile_shared_lib(pkg_dir)


def _get_build_class():
    """Choose build class based on whether this is an editable install."""
    if "develop" in sys.argv or "editable_wheel" in sys.argv:
        return BuildInPlace
    return BuildWithCompile


# Force platform-specific wheel tag (libquant.{so,dylib,dll} ships in pkg).
# Without this, setuptools generates a py3-none-any wheel that pip happily
# installs on the wrong OS.
from setuptools.dist import Distribution as _Distribution
class _BinaryDistribution(_Distribution):
    def has_ext_modules(self):
        return True
    def is_pure(self):
        return False


# Make sure the bundled header lives in quantcpp/_quant.h *before* setuptools
# walks the package contents. This guarantees sdist includes it without us
# touching MANIFEST.in (sdist is always built in dev tree where ../../quant.h
# exists; downstream sdist installs already contain the bundled file).
try:
    _ensure_bundled_header()
except FileNotFoundError as _e:
    # Don't break metadata-only commands (e.g. pip's get_requires hook running
    # in an isolated copy without the dev tree). The header check will fire
    # again at compile time with a clearer error.
    print(f"[quantcpp] WARNING (continuing): {_e}", file=sys.stderr)


setup(
    cmdclass={"build_py": _get_build_class()},
    distclass=_BinaryDistribution,
)
