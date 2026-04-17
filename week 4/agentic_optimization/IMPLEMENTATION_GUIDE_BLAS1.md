# BLAS 1 AI Implementation Guide

**SYSTEM INSTRUCTION FOR AI AGENT:**
You are an expert AI software engineer specialized in high-performance computing (HPC). 
**Your task:** Implement a highly optimized single-precision BLAS Level 1 library from scratch. You MUST follow this blueprint step-by-step.

## 1. Project Structure
```
project/
  include/
    blas1.h              ← Public API (Fortran compatible)
  src/
    blas1.c              ← Implementations (Scalar, AVX2, OpenMP)
  tests/
    test_blas1.c         ← Correctness suite
  bench/
    bench_blas1.c        ← Performance measurement
  Makefile
```

## 2. Kernels to Implement
- `sscal`: $x = \alpha x$
- `scopy`: $y = x$
- `sswap`: $x \leftrightarrow y$
- `saxpy`: $y = \alpha x + y$
- `sdot`: $\sum x_i y_i$
- `snrm2`: $\sqrt{\sum x_i^2}$
- `sasum`: $\sum |x_i|$
- `isamax`: $\text{argmax} |x_i|$
- `srot`: Givens rotation

## 3. Optimization Strategy

### 3.1 AVX2 Vectorization
Use `immintrin.h` for SIMD. 
- Load/Store: `_mm256_loadu_ps`, `_mm256_storeu_ps`.
- Arithmetic: `_mm256_mul_ps`, `_mm256_fmadd_ps`, `_mm256_add_ps`.
- Unrolling: Unroll loops by at least 4x (32 floats) to hide latency and saturate memory bandwidth.

### 3.2 Memory Bandwidth & Prefetching
BLAS 1 is memory-bound.
- **Software Prefetching**: Use `_mm_prefetch((const char*)(ptr + offset), _MM_HINT_T0)`. Experiment with distance (e.g., 32-64 floats).
- **Non-Temporal Stores**: For large vectors that don't fit in cache, use `_mm256_stream_ps` to bypass cache and save bandwidth (requires 32-byte alignment or masking).

### 3.3 Multi-threading
Use OpenMP `#pragma omp parallel for`.
- Use `schedule(static)` for uniform work distribution.
- Set a threshold (e.g., $N > 4096$) below which parallelization is skipped to avoid overhead.
- For reductions (`sdot`, `snrm2`, `sasum`), use `reduction(+:variable)`.

## 4. Automated Agent Loop (Autotuning)
To find the absolute best performance, the implementation should be tunable via a script:
1. Define a template kernel with placeholders for: `UNROLL_FACTOR`, `PREFETCH_DIST`, `USE_NT_STORES`.
2. A master agent (script) loops through parameter combinations.
3. For each combination: compiles, benchmarks, and records GFLOPS.
4. Selects the parameters that maximize GFLOPS for large $N$.

## 5. Implementation Details

### Example: `saxpy` AVX2 (Unroll x4)
```c
void saxpy_avx2(int n, float alpha, const float *x, float *y) {
    __m256 va = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm_prefetch((const char *)(x + i + 64), _MM_HINT_T0);
        _mm_prefetch((const char *)(y + i + 64), _MM_HINT_T0);
        __m256 vx0 = _mm256_loadu_ps(x + i);
        __m256 vy0 = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx0, vy0));
        // ... repeat for i+8, i+16, i+24 ...
    }
    // scalar tail...
}
```

## 6. Verification
Compare results against OpenBLAS.
- Tolerance for `sdot`, `snrm2`: $10^{-4}$ relative error (due to reduction order differences).
- Exact match for `sscal`, `scopy`, `sswap`.
