# SGEMM Optimization Report: Obstacles & Solutions

## Executive Summary

This report documents the optimization of a BLIS-style tiled SGEMM from **0.62–0.70x** (single-threaded baseline) to **0.93–1.04x of OpenBLAS** at 16 threads. The journey involved five major obstacles, each requiring a different class of HPC engineering insight to overcome.

**Target machine**: Intel Xeon E5-2670 v3 (Haswell), 20 cores @ 2.3 GHz, AVX2+FMA, 32 KiB L1d, 256 KiB L2, 25 MiB L3 (shared).

---

## Obstacle 1: Scatter/Gather C Write-Back in the Micro-Kernel

### The Problem

The original 8×8 micro-kernel used **column accumulators**: `acc[j]` held column `j` of the C tile — i.e., 8 values from 8 *different rows*. Since C is row-major (stride = `ldc`), these 8 values are **not contiguous in memory**. The store path required:

```c
// Original: scalar gather → vector process → scalar scatter (per column!)
for (int j = 0; j < NR; j++) {
    float tmp[MR];
    for (int i = 0; i < MR; i++) tmp[i] = C[i*ldc + j];  // GATHER
    __m256 existing = _mm256_loadu_ps(tmp);
    // ... FMA with alpha/beta ...
    _mm256_storeu_ps(tmp, result);
    for (int i = 0; i < MR; i++) C[i*ldc + j] = tmp[i];  // SCATTER
}
```

This **8 scalar reads + 8 scalar writes per column × 8 columns = 128 scalar memory ops** completely negated the SIMD benefit of the compute loop.

### Impact

**~30–40% performance loss.** The micro-kernel spent almost as much time on the store path as on the actual FMA computation.

### Solution: Row-Accumulator Layout

Switched to **row accumulators**: `acc[i]` holds row `i` of the C tile — 8 (or 16) contiguous values in the same row of C. Since C is row-major, `C[i, 0..7]` is contiguous → direct `_mm256_storeu_ps`:

```c
// New: direct vector store (per row)
float *Crow = C + row_idx * ldc;
__m256 existing = _mm256_loadu_ps(Crow);
_mm256_storeu_ps(Crow, _mm256_fmadd_ps(vbeta, existing, row_acc));
```

**Result**: Store path went from 128 scalar ops to 8 vector loads + 8 vector stores.

---

## Obstacle 2: Register-Limited 8×8 Kernel (Insufficient Compute Density)

### The Problem

After fixing the store path, isolated micro-kernel benchmarks showed both row- and column-accumulator 8×8 kernels peaked at **~62 GFLOPS**, while OpenBLAS achieved **~74 GFLOPS** on the same hardware. The 8×8 tile generates only **8 FMAs per k-step**:

| Resource | 8×8 Kernel | Utilization |
|----------|-----------|-------------|
| Accumulators | 8 YMM | 8/16 = 50% |
| B loads | 1 YMM | |
| A broadcast | 1 YMM | |
| **Total registers** | **10/16** | **62.5%** |
| **FMAs per k-step** | **8** | |

Six registers sat idle — wasted compute potential.

### The Diagnosis

A standalone kernel benchmark ([/tmp/bench_kernel.c](file:///tmp/bench_kernel.c)) confirmed this was a fundamental throughput ceiling, not a code-quality issue. Both accumulator layouts hit the same wall at ~62 GFLOPS.

### Solution: 6×16 Micro-Kernel (Matching OpenBLAS Design)

Switched to MR=6, NR=16 — the same tile shape OpenBLAS uses on Haswell:

| Resource | 6×16 Kernel | Utilization |
|----------|-----------|-------------|
| Accumulators | 12 YMM (6 rows × 2 halves) | 12/16 = 75% |
| B loads | 2 YMM (low + high 8 cols) | |
| A broadcast | 1 YMM | |
| Spare | 1 YMM | |
| **Total registers** | **16/16** | **100%** |
| **FMAs per k-step** | **12** | **+50% vs 8×8** |

**Result**: Single-threaded jumped from ~53 to ~65–68 GFLOPS (0.88–0.95x OpenBLAS).

---

## Obstacle 3: Suboptimal Cache-Blocking Tile Sizes

### The Problem

The original tile sizes were `MC=512, KC=256, NC=4096` (later experimented with `MC=64, NC=128` in the previous code). These caused:

1. **MC=512**: A panel = 512×256×4B = **512 KiB** — exceeds L2 (256 KiB), causing L2 thrashing
2. **MC=64** (dynamic): Too few ic-strips for parallelism, excessive per-strip overhead

### The Tuning Process

Tried several configurations:

| MC | KC | NC | Single-thread ratio |
|----|----|----|-------------------|
| 256 | 256 | 2048 | 0.69–0.72x |
| 128 | 512 | 4096 | 0.71–0.74x |
| **120** | **512** | **4096** | **0.88–0.95x** |

### Solution: MC=120, KC=512, NC=4096

The key insight: **MC must be a multiple of MR=6** and the A panel must fit in half of L2:
- A panel: 120 × 512 × 4B = **240 KiB ≤ 256 KiB** (L2) ✓
- Micro-panel of A: 6 × 512 × 4B = **12 KiB ≤ 32 KiB** (L1d) ✓
- B panel: 512 × 4096 × 4B = **8 MiB ≤ 25 MiB** (L3) ✓
- Larger KC=512 **amortizes packing cost** better than KC=256

---

## Obstacle 4: Thread Pool Churn and Serial B-Packing

### The Problem

The initial multi-threaded implementation created and destroyed the OpenMP parallel region **per (jc, pc) iteration**:

```c
for (int jc ...) {
    for (int pc ...) {
        pack_B_panel(...);  // SERIAL — one thread does all B packing
        #pragma omp parallel  // thread pool created HERE
        {
            #pragma omp for
            for (int ic ...) { ... }
        }  // threads destroyed HERE
    }
}
```

Two problems:
1. **Thread pool overhead**: Creating/joining threads per (jc, pc) — for N=2048, K=2048, KC=512, NC=4096: only 1×4 = 4 iterations, but still measurable
2. **Serial B-packing**: One thread packs the entire B panel (up to 8 MiB) while 15 threads idle

### Impact

At 16 threads: **0.25–0.64x OpenBLAS** — all the single-threaded gains were lost.

### Solution: Single Parallel Region + Parallel B-Packing

```c
#pragma omp parallel  // created ONCE
{
    for (int jc ...) {
        for (int pc ...) {
            pack_B_panel_parallel(...);  // ALL threads pack B
            #pragma omp barrier          // sync before compute
            
            #pragma omp for schedule(dynamic) nowait
            for (int ic ...) { ... }
            
            #pragma omp barrier          // sync before next B pack
        }
    }
}
```

**Result at 16 threads**: 2048 went from 0.64x to **1.04x OpenBLAS**.

---

## Obstacle 5: Insufficient Parallelism for Small Matrices

### The Problem

Even with the single parallel region, small matrices had very few parallel tasks:

| N | ic-strips (M/MC) | Available tasks | Threads | Utilization |
|---|------------------|-----------------|---------|-------------|
| 256 | ceil(256/120) = **3** | 3 | 16 | 19% |
| 512 | ceil(512/120) = **5** | 5 | 16 | 31% |
| 1024 | ceil(1024/120) = **9** | 9 | 16 | 56% |
| 2048 | ceil(2048/120) = **18** | 18 | 16 | 100% |

For N=256, 13 out of 16 threads sat idle during the compute phase.

### Solution: 2D Parallel Decomposition

Instead of parallelizing only the ic loop, we flatten the (ic, jr) space into a single parallel loop:

```c
int total_tasks = nr_i_tiles * nr_j_strips;
#pragma omp for schedule(dynamic) nowait
for (int task = 0; task < total_tasks; task++) {
    int ic_idx = task / nr_j_strips;
    int jt     = task % nr_j_strips;
    // ...
}
```

| N | ic-tiles | jr-strips (N/NR) | **Total tasks** | Utilization |
|---|---------|------------------|-----------------|-------------|
| 256 | 3 | 16 | **48** | 100% |
| 512 | 5 | 32 | **160** | 100% |
| 1024 | 9 | 64 | **576** | 100% |
| 2048 | 18 | 128 | **2304** | 100% |

**Result**: N=256 went from 0.29x to **0.93x OpenBLAS**. N=512 went from 0.54x to **0.98x**.

---

## Summary: Performance Progression

| Optimization | N=256 | N=512 | N=1024 | N=2048 |
|---|---|---|---|---|
| **Baseline** (8×8, scatter) | 0.70x* | 0.70x* | 0.65x* | 0.62x* |
| + Row-accumulator 8×8 | 0.74x* | 0.74x* | 0.73x* | 0.71x* |
| + 6×16 kernel + tile tuning | 0.95x* | 0.94x* | 0.92x* | 0.88x* |
| + Single parallel region | 0.30x | 0.45x | 0.82x | **1.04x** |
| **+ 2D decomposition** | **0.93x** | **0.98x** | **0.94x** | **1.04x** |

*\*Single-threaded measurements. All others at 16 threads.*

## Key Lessons

1. **Memory access patterns dominate SIMD performance** — The scatter/gather store destroyed 30-40% of throughput, far more than any compute optimization could recover.

2. **Register utilization = compute density** — The jump from 8×8 (50% register utilization) to 6×16 (100%) directly translated to 50% more FMAs per cycle.

3. **Cache hierarchy must be respected precisely** — MC=512 overflowed L2 by 2×; MC=120 fit with 6% headroom. The difference was ~20% throughput.

4. **Parallelization architecture matters more than micro-optimization at scale** — Single-threaded gains of 0.95x meant nothing when the parallel structure yielded 0.25x at 16 threads.

5. **Work decomposition must match thread count** — 1D parallelism (3 tasks for 16 threads) vs 2D (48 tasks) was the difference between 0.29x and 0.93x.
