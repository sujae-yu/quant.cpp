# TurboQuant.cpp — Standalone Makefile (no CMake needed)
#
# Usage:
#   make              # build tq_run + tq_convert
#   make tq_run       # inference tool only
#   make test         # build and run tests (requires Google Test)
#   make clean        # remove build artifacts
#
# Cross-platform:
#   Linux/gcc:     make CC=gcc
#   macOS/clang:   make                    (auto-detects Apple Silicon)
#   macOS+Metal:   make METAL=1            (enables Metal GPU backend)
#   Windows/mingw: make CC=x86_64-w64-mingw32-gcc TARGET=tq_run.exe
#
# Options:
#   DEBUG=1    — debug build (-g -O0 -fsanitize=address)
#   METAL=1    — enable Metal GPU backend (macOS only)
#   NEON=1     — force NEON (auto-detected on arm64)
#   AVX2=1     — force AVX2 (auto-detected on x86_64)

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter
LDFLAGS ?= -lm -lpthread

# Optimization
ifdef DEBUG
  CFLAGS += -g -O0 -fsanitize=address -fsanitize=undefined
  LDFLAGS += -fsanitize=address -fsanitize=undefined
else
  CFLAGS += -O2
endif

# Include paths
CFLAGS += -Iinclude

# Auto-detect architecture
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_M),arm64)
  CFLAGS += -mcpu=native
  NEON ?= 1
endif
ifeq ($(UNAME_M),aarch64)
  CFLAGS += -mcpu=native
  NEON ?= 1
endif
ifeq ($(UNAME_M),x86_64)
  CFLAGS += -march=native
  AVX2 ?= 1
endif

# Apple Accelerate framework (macOS)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS += -framework Accelerate
  CFLAGS  += -DTQ_HAS_ACCELERATE=1 -DACCELERATE_NEW_LAPACK=1
endif

# ============================================================
# Source files
# ============================================================

SRC_CORE := $(wildcard src/core/*.c)
SRC_CACHE := $(wildcard src/cache/*.c)
SRC_CPU := $(wildcard src/backend/cpu/*.c)
SRC_ENGINE := $(wildcard src/engine/*.c)

SRC_LIB := $(SRC_CORE) $(SRC_CACHE) $(SRC_CPU) $(SRC_ENGINE)
OBJ_LIB := $(SRC_LIB:.c=.o)

# Metal backend (macOS only, optional)
ifdef METAL
  ifeq ($(UNAME_S),Darwin)
    SRC_METAL_OBJ := $(wildcard src/backend/metal/*.m)
    OBJ_METAL := $(SRC_METAL_OBJ:.m=.o)
    OBJ_LIB += $(OBJ_METAL)
    CFLAGS  += -DTQ_HAS_METAL=1
    LDFLAGS += -framework Metal -framework Foundation
    OBJCFLAGS = $(CFLAGS) -fobjc-arc
  endif
endif

# ============================================================
# Targets
# ============================================================

.PHONY: all clean test

all: tq_run tq_convert

# Static library
libturboquant.a: $(OBJ_LIB)
	$(AR) rcs $@ $^

# Main tools
tq_run: tools/tq_run.c libturboquant.a
	$(CC) $(CFLAGS) -o $@ $< -L. -lturboquant $(LDFLAGS)

tq_convert: tools/tq_convert.c libturboquant.a
	$(CC) $(CFLAGS) -o $@ $< -L. -lturboquant $(LDFLAGS)

# ============================================================
# Compile rules
# ============================================================

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Objective-C (Metal dispatch)
%.o: %.m
	$(CC) $(OBJCFLAGS) -c -o $@ $<

# ============================================================
# Test (lightweight — no Google Test dependency)
# ============================================================

test: tq_run
	@echo "=== Quick sanity test ==="
	@echo "Building..."
	@echo "Running tq_run --info on test..."
	@if [ -f model.tqm ]; then \
		./tq_run model.tqm --info && echo "PASS: model loads" || echo "FAIL"; \
	else \
		echo "SKIP: no model.tqm found (download a model first)"; \
	fi
	@echo "=== Done ==="

# ============================================================
# Clean
# ============================================================

clean:
	rm -f $(OBJ_LIB) $(OBJ_METAL) libturboquant.a tq_run tq_convert
	rm -f src/**/*.o

# ============================================================
# Help
# ============================================================

help:
	@echo "TurboQuant.cpp Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make              Build tq_run + tq_convert"
	@echo "  make tq_run       Build inference tool only"
	@echo "  make clean        Remove build artifacts"
	@echo "  make test         Quick sanity test"
	@echo "  make help         Show this help"
	@echo ""
	@echo "Options:"
	@echo "  CC=gcc            Use gcc instead of default cc"
	@echo "  DEBUG=1           Debug build with ASan"
	@echo "  METAL=1           Enable Metal GPU (macOS only)"
	@echo "  AVX2=1            Force AVX2 (x86_64)"
	@echo "  NEON=1            Force NEON (arm64)"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # default build"
	@echo "  make CC=gcc                   # Linux gcc build"
	@echo "  make METAL=1                  # macOS with Metal GPU"
	@echo "  make DEBUG=1                  # debug + sanitizers"
	@echo "  make CC=x86_64-w64-mingw32-gcc  # cross-compile for Windows"
