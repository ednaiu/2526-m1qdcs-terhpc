# =============================================================================
# Makefile  —  TER HPC BLAS Week 1: SGEMM AVX2+FMA
# =============================================================================
#
# Targets:
#   make all        —  build hello, test binary, and bench binary
#   make hello      —  build and run OpenMP Hello World
#   make test       —  build and run correctness tests
#   make bench      —  build and run performance benchmark (single-thread)
#   make clean      —  remove build artefacts
#
# Requirements (on the remote Linux workstation):
#   gcc >= 5.0      (AVX2 + FMA3 support)
#   OpenBLAS        (ubuntu: sudo apt install libopenblas-dev)
#
# Usage on the remote machine:
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 make bench
# =============================================================================

CC      := gcc
CFLAGS  := -O3 -march=native -mavx2 -mfma -fopenmp \
            -Wall -Wextra -std=c11
IFLAGS  := -Iinclude
LIBS    := -lopenblas -lm

# Source files
SGEMM_SRC := src/sgemm.c

# Build directory
BINDIR  := bin

.PHONY: all hello test bench clean

all: $(BINDIR) hello test bench

$(BINDIR):
	mkdir -p $(BINDIR)

# -----------------------------------------------------------------------
# Target 1: OpenMP Hello World
# -----------------------------------------------------------------------
hello: $(BINDIR)/hello_omp
	@echo "--- Running OpenMP Hello World ---"
	OMP_NUM_THREADS=8 $(BINDIR)/hello_omp

$(BINDIR)/hello_omp: src/hello_omp.c | $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $<

# -----------------------------------------------------------------------
# Target 2: Correctness tests
# -----------------------------------------------------------------------
test: $(BINDIR)/test_sgemm
	@echo "--- Running SGEMM Correctness Tests ---"
	-$(BINDIR)/test_sgemm

$(BINDIR)/test_sgemm: tests/test_sgemm.c $(SGEMM_SRC) | $(BINDIR)
	$(CC) $(CFLAGS) $(IFLAGS) -o $@ $^ -lm

# -----------------------------------------------------------------------
# Target 3: Performance benchmark vs OpenBLAS
# -----------------------------------------------------------------------
bench: $(BINDIR)/bench_sgemm
	@echo "--- Running SGEMM Benchmark (single-thread) ---"
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(BINDIR)/bench_sgemm

$(BINDIR)/bench_sgemm: bench/bench_sgemm.c $(SGEMM_SRC) | $(BINDIR)
	$(CC) $(CFLAGS) $(IFLAGS) -o $@ $^ $(LIBS)

# -----------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------
clean:
	rm -rf $(BINDIR)
