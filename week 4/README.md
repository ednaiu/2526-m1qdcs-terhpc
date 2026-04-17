# Week 4 — BLAS 1 Optimized AVX2 Kernels + SGEMM (Week 3)

## Quick Start — BLAS 1

```bash
# Build BLAS 1 test + bench
make all-blas1

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
python3 bayesian_optimizer.py --size 1024 --trials 30
```

## Status: ✅ COMPLETE (Week 3 SGEMM)

All 96 SGEMM tests pass. Core optimizations implemented:
- Loop unrolling ×4
- Software prefetching (next-tile)
- Non-temporal stores
- ASM kernel bug fixes

## Status: ✅ COMPLETE (Week 4 BLAS 1)

All 1287 BLAS 1 tests pass (vs OpenBLAS reference). Kernels: `sscal`, `scopy`, `sswap`, `saxpy`, `sdot`, `snrm2`, `sasum`, `isamax`, `srot`.

Three implementation layers per kernel:
- **Scalar baseline** — readable reference
- **AVX2** — 8-wide vectorized, software prefetch
- **AVX2 + OpenMP** — parallel (skipped for small n < 4096)

Fortran BLAS signatures implemented (trailing `_`, args by pointer).

**Performance vs Week 2:** +30–111% improvement  
**Performance vs OpenBLAS:** 3–25× faster at small/medium sizes

## Documentation

- **COMPLETION_SUMMARY.md** — Quick overview (start here)
- **WEEK3_REPORT.md** — Complete technical report
- **WEEK3_PLAN.md** — Original implementation plan
- **INTRODUCTION.md** — Beginner-friendly explanation

## Key Results

### Performance Comparison (Median of 5 runs)

| Size  | Threads | OpenBLAS | Week 2  | Ratio  |
|-------|---------|----------|---------|--------|
| 256³  | 8       | 46.9     | 151.5   | 3.2×   |
| 256³  | 16      | 7.4      | 183.8   | 24.9×  |
| 512³  | 8       | 35.2     | 162.2   | 4.6×   |
| 512³  | 16      | 29.3     | 239.4   | 8.2×   |
| 1024³ | 8       | 210.0    | 192.6   | 0.92×  |
| 1024³ | 16      | 216.3    | 277.7   | 1.28×  |

*(All values in GFLOP/s)*

### Week 3 Optimizations Impact

Week 3 optimizations added on top of Week 2:

| Size  | Threads | Week 2 | Week 3 | Gain    |
|-------|---------|--------|--------|---------|
| 256³  | 8       | 152.5  | 216.2  | +41.8%  |
| 512³  | 8       | 176.5  | 373.3  | +111.4% |
| 1024³ | 16      | 280.8  | 568.2  | +102.3% |

## Project Structure

```
week 3/
├── src/
│   ├── sgemm.c              # BLIS framework
│   └── kernels/
│       ├── kernel_6x16.h     # Main 6×16 micro-kernel (C)
│       ├── kernel_6x16_asm.h # ASM version (×4 unroll)
│       ├── kernel_8x8.h      # 8×8 micro-kernel
│       └── kernel_4x24.h     # 4×24 micro-kernel
├── include/
│   └── sgemm.h              # API and config
├── tests/
│   └── test_sgemm.c         # 96 test cases
├── bench/
│   └── bench_sgemm.c        # Benchmarking harness
├── autotuning/
│   ├── bayesian_optimizer.py
│   └── gradient_descent.py
└── results/
    ├── WEEK3_REPORT.md
    └── *.csv                # Benchmark data

```

## Build System

```bash
# Targets
make all        # Build library + tests + benchmark
make test       # Compile and run test suite
make bench      # Compile benchmark only
make clean      # Remove build artifacts

# Options
make KERNEL=KERNEL_6X16 test    # Test specific kernel
make TASK=TASK2 R=4 test        # Test with K-replication
```

## Testing

96 test cases covering:
- 3 micro-kernels (8×8, 6×16, 4×24)
- 2 parallelism modes (TASK1 2D, TASK2 3D K-replication)
- Square, prime, tall, wide, thin-K matrices
- Alpha/beta variations

```bash
$ make test
===== ALL TESTS PASSED =====
```

## Optimizations Implemented

### 1. Loop Unrolling ×4
- Main k-loop processes 4 k-steps per iteration (was 2)
- Reduces branch overhead by 50%
- Files: `kernel_6x16.h`, `kernel_8x8.h`, `kernel_4x24.h`, `kernel_6x16_asm.h`

### 2. Software Prefetching
- Prefetch next tile's A and B panels during current tile computation
- 6 cache lines prefetched per ×4 iteration (2 for A, 4 for B)
- Uses `_mm_prefetch(..., _MM_HINT_T0)` to bring data into L1

### 3. Non-Temporal Stores
- Bypass cache on C matrix writes using `_mm256_stream_ps`
- Frees L1/L2 cache for A and B panels
- Especially effective for large matrices (2048³)

### 4. ASM Kernel Bug Fix
- Resolved GCC 30-operand limit by computing k4 in assembly
- Uses `movl` + `shrl` + `%%eax` for loop counter

## Benchmarking

```bash
# Single size
OMP_NUM_THREADS=8 ./bin/bench_sgemm 1024

# Multiple sizes
for sz in 256 512 1024 2048; do
    OMP_NUM_THREADS=16 ./bin/bench_sgemm $sz
done
```

## Auto-Tuning

Bayesian Optimization finds optimal MC/KC/NC parameters:

```bash
cd autotuning
python3 bayesian_optimizer.py \
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

## Known Limitations

From Week 2 (not fixed in Week 3):
1. TASK2 reduction runs sequentially (measurable at 8+ threads)
2. Small matrices (64×64) dominated by packing overhead
3. Thread scaling poor at 256×256 with 16 threads (only 2 m-strips)

See `WEEK3_REPORT.md` Section 5 for details and proposed fixes.

## Future Work

1. Create `IMPLEMENTATION_GUIDE.md` for reproducibility
2. Parallelize TASK2 reduction with `#pragma omp for`
3. Implement small-matrix bypass (on-stack buffers)
4. Adaptive MC selection based on thread count
5. Profile 1024³ to understand OpenBLAS advantage at 8T

## References

- BLIS Framework: [flame/blis](https://github.com/flame/blis)
- Intel Intrinsics Guide: [software.intel.com](https://software.intel.com/intrinsics-guide)
- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn)

## License

Educational project for TER HPC course.

---

**For detailed technical analysis, see WEEK3_REPORT.md**
