# Week 5 — BLAS 2 Optimized AVX2 Kernels + BLAS 1 (Week 4) + SGEMM (Week 3)

## Quick Start — BLAS 2

```bash
# Build BLAS 2 test
make test-blas23

# Run correctness tests (vs OpenBLAS)
OMP_NUM_THREADS=8 ./bin/test_blas23
```

## Quick Start — BLAS 1

```bash
# Build BLAS 1 test + bench
make all-blas

# Run correctness tests (all 1287 tests vs OpenBLAS)
OMP_NUM_THREADS=8 ./bin/test_blas1

# Run performance benchmark (all kernels, all sizes)
OMP_NUM_THREADS=8 ./bin/bench_blas1
```

## Quick Start — SGEMM (Week 3)

```bash
# Build and test
make clean && make test

# Run benchmark (8 threads)
OMP_NUM_THREADS=8 ./bin/bench_sgemm 1024

# Run auto-tuning
cd autotuning
python3 bayesian_opt.py --size 1024 --trials 30
```

## Status: ✅ COMPLETE (Week 5 BLAS 2)

All 7 BLAS 2 kernels implemented with Fortran-compatible signatures and OpenMP parallelism:

| Kernel | Operation | AVX2 | OpenMP |
|--------|-----------|------|--------|
| `sgemv` | y = alpha\*A\*x + beta\*y | ✅ Full (NoTrans/Trans) | ✅ |
| `sger`  | A = alpha\*x\*y' + A | ✅ | ✅ |
| `ssymv` | y = alpha\*A\*x + beta\*y (symmetric) | — | ✅ |
| `strmv` | x = A\*x (triangular) | — | — |
| `strsv` | A\*x = b (triangular solve) | — | — |
| `ssyr`  | A = alpha\*x\*x' + A | — | ✅ |
| `ssyr2` | A = alpha\*x\*y' + alpha\*y\*x' + A | — | ✅ |

All functions have Fortran BLAS entry points (`sgemv_`, `sger_`, etc.) with pointer arguments.

## Status: ✅ COMPLETE (Week 4 BLAS 1)

All 1287 BLAS 1 tests pass (vs OpenBLAS reference). Kernels: `sscal`, `scopy`, `sswap`, `saxpy`, `sdot`, `snrm2`, `sasum`, `isamax`, `srot`.

Two implementation layers per kernel:
- **Scalar baseline** — readable reference
- **AVX2** — 8-wide vectorized, software prefetch

Fortran BLAS signatures implemented (trailing `_`, args by pointer).

**Performance vs OpenBLAS:** 3–25× faster at small/medium sizes

## Status: ✅ COMPLETE (Week 3 SGEMM)

All 96 SGEMM tests pass. Core optimizations implemented:
- Loop unrolling ×4
- Software prefetching (next-tile)
- Non-temporal stores
- ASM kernel bug fixes

## Project Structure

```
week 5/
├── src/
│   ├── blas1.c              # 9 BLAS 1 kernels (AVX2)
│   ├── blas2.c              # 7 BLAS 2 kernels (Week 5)
│   ├── blas3.c              # BLAS 3 stubs (ssyrk)
│   ├── sgemm.c              # BLIS framework (Week 3)
│   └── kernels/
│       ├── kernel_6x16.h     # Main 6×16 micro-kernel (C)
│       ├── kernel_6x16_asm.h # ASM version (×4 unroll)
│       ├── kernel_8x8.h      # 8×8 micro-kernel
│       └── kernel_4x24.h     # 4×24 micro-kernel
├── include/
│   ├── sgemm.h              # SGEMM API and config
│   ├── blas1.h              # BLAS 1 API + Fortran stubs
│   ├── blas2.h              # BLAS 2 API + Fortran stubs
│   └── blas3.h              # BLAS 3 API + Fortran stubs
├── tests/
│   ├── test_blas1.c         # 1287 BLAS 1 test cases
│   ├── test_blas23.c        # BLAS 2 & 3 correctness tests
│   ├── test_sgemm.c         # 96 SGEMM test cases
│   └── simple_test.c        # Lightweight SGEMM tests
├── bench/
│   ├── bench_blas1.c        # BLAS 1 benchmarking harness
│   ├── bench_sgemm.c        # SGEMM benchmark + OpenBLAS comparison
│   └── compare_direct_openblas.c
├── autotuning/
│   ├── bayesian_opt.py
│   ├── gradient_descent.py
│   ├── simulated_annealing.py
│   └── ...
└── results/
    └── *.csv / *.txt        # Benchmark outputs
```

## Build System

```bash
# Targets
make all           # Build benchmark + simple test
make test          # Compile SGEMM test suite
make all-blas      # Build BLAS 1 + 2 + 3 tests + static library
make test-blas23   # Build BLAS 2 & 3 test binary
make bench         # Compile SGEMM benchmark only
make lib-blas      # Create libblas.a static library
make clean         # Remove all build artifacts

# Options
make KERNEL=KERNEL_6X16 test    # Test specific kernel
make TASK=TASK2 R=4 test        # Test with K-replication
```

## Key Results

### SGEMM vs OpenBLAS (Week 3 optimizations)

| Size  | Threads | Week 2  | Week 3  | Gain     |
|-------|---------|---------|---------|----------|
| 256³  | 8       | 152.5   | 216.2   | +41.8%   |
| 512³  | 8       | 176.5   | 373.3   | +111.4%  |
| 1024³ | 16      | 280.8   | 568.2   | +102.3%  |

*(All values in GFLOP/s, median of 5 runs)*

### BLAS 1 vs OpenBLAS

Performance: **3–25× faster** at small/medium sizes (n < 4096).  
Correctness: all 1287 test cases pass.

## Week 5 BLAS 2 — Optimization Notes

### sgemv (NoTrans)
Row-major dot product per output row. AVX2 inner loop with `_mm256_fmadd_ps`, horizontal sum via `hsum256`. Parallelized with `#pragma omp parallel for`.

### sgemv (Trans)
Recast as row-wise saxpy: each row of A is added into the output slice `y[j0..j1]` owned by the thread. This converts strided column reads into contiguous row reads, enabling full AVX2 FMA vectorization and improving cache locality. Parallelized by partitioning the output range `n` across threads.

### sger
Outer product rank-1 update. AVX2 vectorized inner loop over `n` (j-dimension), parallelized over rows with OpenMP.

### ssymv
Respects `uplo` (U/L) by reflecting matrix access. Scalar inner loop + OpenMP parallelism. No AVX2 due to conditional access pattern.

### strmv
Copies `x` to a temporary buffer, then performs triangular matrix-vector multiply (with or without transpose, unit/non-unit diagonal). Sequential.

### strsv
Forward or backward triangular substitution depending on `uplo` and `trans`. Sequential by nature (data dependency chain).

### ssyr / ssyr2
Symmetric rank-1 / rank-2 update respecting `uplo`. Inner loop is scalar, parallelized with OpenMP over rows.

## Known Limitations

1. `ssymv`: reflected triangle access is currently scalar.
2. `ssymv`/`ssyr`/`ssyr2`: no AVX2 intrinsics — only OpenMP parallelism.
3. `isamax`: AVX2 variant falls back to scalar implementation.
4. No task-based (OpenMP `task`) parallelism in BLAS 2 — uses `parallel for` instead.

## Auto-Tuning

Bayesian Optimization finds optimal MC/KC/NC parameters for SGEMM:

```bash
cd autotuning
python3 bayesian_opt.py \
    --size 1024 \
    --trials 50 \
    --mc-range 60 240 \
    --kc-range 256 1024 \
    --nc-range 2040 8160
```

**Optimal configs found:**
- 256³: `MC=120, KC=256, NC=4080`
- 512³: `MC=120, KC=512, NC=4080`
- 1024³: `MC=120, KC=256, NC=4080`

## Benchmarking

```bash
# SGEMM — single size
OMP_NUM_THREADS=8 ./bin/bench_sgemm 1024

# SGEMM — multiple sizes
for sz in 256 512 1024 2048; do
    OMP_NUM_THREADS=16 ./bin/bench_sgemm $sz
done

# BLAS 1 — all kernels
OMP_NUM_THREADS=8 ./bin/bench_blas1
```

## References

- BLIS Framework: [flame/blis](https://github.com/flame/blis)
- Intel Intrinsics Guide: [software.intel.com](https://software.intel.com/intrinsics-guide)
- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn)
- BLAS reference: [netlib.org/blas](https://www.netlib.org/blas/)

## License

Educational project for TER HPC course — Université Paris-Saclay M1 QDCS.
