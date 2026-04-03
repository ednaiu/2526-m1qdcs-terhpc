# Week 3 — Implementation Plan

**Project:** TER HPC — SGEMM Optimization  
**Baseline:** Week 2 codebase (copy into `week 3/`)  
**Goal:** Apply targeted micro-kernel and macro-level optimizations, fix Week 2 known limitations, and produce a complete from-scratch implementation plan document.

---

## 0. What the README asks for (Week 3)

1. **Performance review at each step** — benchmark after every optimization, include Week 1 & 2 numbers as baseline
2. **Fix Week 2 known limitations** (exclude NUMA)
3. **Loop unrolling** for micro-kernel (extend ×2 → ×4)
4. **Non-temporal stores for C** — bypass cache on write since C is not reused
5. **Prefetching** — hint CPU to load next tile while current tile is computing
6. **Transposition integration** — try integrating B-packing transposition into GEMM itself
7. **From-scratch implementation plan** — document allowing AI to regenerate the full optimized kernel

---

## 1. Baseline Setup

**Task:** Copy week 2 into week 3, verify it compiles and all 96 tests pass.

```
week 3/
  include/    ← copy from week 2
  src/        ← copy from week 2
  bench/      ← copy from week 2
  tests/      ← copy from week 2
  autotuning/ ← copy from week 2
  Makefile    ← copy from week 2 (adjust paths if needed)
```

**Baseline benchmark** (8T and 16T): record Week 2 numbers as reference column.  
All subsequent benchmarks show delta vs. this baseline.

---

## 2. Optimization 1 — Loop Unrolling ×4

### Background
All three C kernels currently unroll the k-loop ×2 (process 2 k-steps per iteration).
Unrolling ×4 reduces loop overhead (branch + counter decrement = 2 instructions per 2 k-steps → 2 instructions per 4 k-steps) and gives the out-of-order engine a larger instruction window to schedule FMAs and loads together.

### Files to change
- `src/kernels/kernel_8x8.h`
- `src/kernels/kernel_6x16.h`
- `src/kernels/kernel_4x24.h`

### What to do
Replace the unrolled-×2 main loop with an unrolled-×4 loop. Keep a ×2 peel loop for `k_len % 4 == 2` and a scalar tail for `k_len % 2 == 1`.

**Pseudo-structure (6×16 kernel):**
```c
int k = 0;
for (; k <= k_len - 4; k += 4) {
    /* k+0 */   load blo0, bhi0; fmadd rows 0..5 with blo0/bhi0;
    /* k+1 */   load blo1, bhi1; fmadd rows 0..5 with blo1/bhi1;
    /* k+2 */   load blo2, bhi2; fmadd rows 0..5 with blo2/bhi2;
    /* k+3 */   load blo3, bhi3; fmadd rows 0..5 with blo3/bhi3;
    pA += 4 * MR_6x16;
    pB += 4 * NR_6x16;
}
/* x2 peel */
for (; k <= k_len - 2; k += 2) { ... }
/* x1 tail */
for (; k < k_len; k++) { ... }
```

**Register budget (6×16):**
- 12 accumulators (ymm0–ymm11) — unchanged
- 2 B registers per k-step (ymm12, ymm13) — loaded sequentially, not held simultaneously
- 1 A broadcast (ymm14) — reused
- ymm15 free → no register spill added

Since B vectors are loaded then consumed immediately, there is no additional register pressure from ×4 unrolling.

### Expected benefit
2–8% throughput gain at medium and large sizes. Negligible effect at small sizes (loop overhead is already amortized). The ASM kernel (kernel_6x16_asm.h) can be extended similarly.

### Validation
Run `make test` after each kernel modification. Benchmark 8T and 16T before continuing.

---

## 3. Optimization 2 — Software Prefetching of Next Tile

### Background
When the CPU finishes a micro-kernel call and moves to the next tile, it must load the new `pA` panel (next m-strip) from L2 into L1. This causes a stall of ~10–15 cycles. Software prefetch lets us overlap this load with the current tile's FMA work.

The existing intra-tile prefetch (`_mm_prefetch(pA + 8*MR, T0)`) is already handled by the hardware prefetcher for sequential access. The important gap is the **next tile** whose address cannot be predicted by hardware.

**Invariant the README states:** "the cache must be able to hold both existing and prefetched tiles simultaneously." This constrains tile sizing — MR×KC×sizeof(float) × 2 must fit in L1 for `_MM_HINT_T0`, or MC×KC × 2 must fit in L2 for `_MM_HINT_T1`.

### Kernel signature change

Add two pointer parameters to every C kernel and the `kernel_fn_t` typedef:

```c
// Before:
typedef void (*kernel_fn_t)(const float *pA, const float *pB,
                             float *C, int k_len,
                             float alpha, float beta, int ldc);

// After:
typedef void (*kernel_fn_t)(const float *pA, const float *pB,
                             float *C, int k_len,
                             float alpha, float beta, int ldc,
                             const float *next_A,   // next tile's A start
                             const float *next_B);  // next tile's B start
```

`next_A` and `next_B` are always valid pointers (never NULL). For the last tile, `macro_kernel` passes the current `pA`/`pB` as dummy values — prefetching already-cached data is harmless.

### Inside the kernel

Remove the existing `_mm_prefetch(pA + 8*MR, T0)` calls (intra-tile, handled by HW).  
Add next-tile prefetch spread across the k-loop:

```c
const float *pfA = next_A;
const float *pfB = next_B;

for (; k <= k_len - 4; k += 4) {
    /* Spread next-tile prefetch across iterations.
     * Each unrolled step advances pfA by 4×MR floats = 1 cache line (for 6×16:
     * 4×6×4 = 96 bytes → 2 cache lines per iter). Use HINT_T0 to bring
     * next tile into L1 before the next kernel call starts.              */
    _mm_prefetch((const char *)pfA, _MM_HINT_T0);
    pfA += 4 * MR;
    _mm_prefetch((const char *)pfB, _MM_HINT_T0);
    _mm_prefetch((const char *)(pfB + 8), _MM_HINT_T0);  // 2nd CL for B (NR≥16)
    pfB += 4 * NR;

    /* FMA compute for k, k+1, k+2, k+3 */
    ...
}
```

**Cache line budget per kernel per ×4 iteration:**

| Kernel | A advance (bytes) | A CLs | B advance (bytes) | B CLs |
|--------|------------------|-------|------------------|-------|
| 8×8    | 4×8×4 = 128      | 2     | 4×8×4 = 128      | 2     |
| 6×16   | 4×6×4 = 96       | 2     | 4×16×4 = 256     | 4     |
| 4×24   | 4×4×4 = 64       | 1     | 4×24×4 = 384     | 6     |

Issue that many `_mm_prefetch` calls per iteration (1 per cache line).

### `macro_kernel` change

```c
/* Compute next_A pointer for this m-strip iteration */
const float *next_A = (it + 1 < nr_m)
                    ? pA_panel + (size_t)(it + 1) * K_blk * MR
                    : pA;   /* last strip: dummy = current (harmless re-prefetch) */
const float *next_B = pB_strip;   /* B doesn't change within macro_kernel */

kfn(pA, pB_strip, Ctile, K_blk, alpha, beta_k, ldc, next_A, next_B);
```

Note: `sgemm_task2` calls `macro_kernel` inside a task — the same change propagates automatically.

### Hint level choice
Use `_MM_HINT_T0` (prefetch into L1 cache). The next tile will be used immediately after the current tile, so we want it as close as possible.

For larger KC (>512), the A panel is large enough that HINT_T1 (into L2 only) avoids L1 pollution. We can make the hint level depend on `KC` at compile time.

### Expected benefit
5–15% improvement at medium sizes (512³–1024³) where tile transitions are frequent. Larger benefit when K is large (many k-steps = more time to prefetch before the next tile is needed).

---

## 4. Optimization 3 — Non-Temporal Stores for C

### Background
C matrix rows are written at the end of each micro-kernel call and not read again until the next K-block accumulation. Using `_mm256_stream_ps` (non-temporal store) writes directly to memory, bypassing L1/L2, keeping those cache lines free for A and B panels.

**When it applies:**
- `beta_k == 0.0f` (first K-block per tile): C is write-only; NT stores are always correct
- `beta_k == 1.0f` (subsequent K-blocks): we read C, accumulate, then NT-store back — the load will miss if previous write was NT, but that is acceptable since we immediately overwrite

**Alignment requirement:** `_mm256_stream_ps` requires 32-byte aligned addresses.

### Implementation strategy

1. **Add `use_nt_store` boolean flag** to `sgemm_config_t`:
   ```c
   int use_nt_store;   /* 0 = regular stores, 1 = non-temporal stores */
   ```
   Default: 0 (off; turned on by benchmark to measure effect).

2. **In `macro_kernel`**: pass `use_nt_store` to the kernel via an extended typedef, OR use two separate code paths (hot-loop conditional outside k-loop).

3. **In kernel STORE phase**: replace `_mm256_storeu_ps` with `_mm256_stream_ps` when enabled. Alignment check: if `(uintptr_t)Cr % 32 != 0`, fall back to `_mm256_storeu_ps`.

4. **After each `kfn()` call in macro_kernel** (or task1/task2), call `_mm_sfence()` to ensure NT stores are visible to other threads. This is a single instruction with ~10 cycle cost.

**Simplified version for testing** (no flag, always NT, requires aligned C):
```c
/* In kernel store phase, replace: */
_mm256_storeu_ps(Cr, result);
/* With: */
if ((uintptr_t)Cr % 32 == 0)
    _mm256_stream_ps(Cr, result);
else
    _mm256_storeu_ps(Cr, result);
```

### Expected benefit
- **Large matrices (2048³):** C doesn't fit in cache anyway; NT stores reduce write-back traffic. Expected +5–20%.
- **Medium matrices (512–1024):** C fits in L3; NT stores free L3 for A panel. Expected 0–10%.
- **Small matrices:** NT stores add sfence overhead → may hurt slightly.

### Test plan
Run benchmarks with NT stores enabled vs disabled. Report per-size delta. Keep NT stores only if median improvement is positive.

---

## 5. Optimization 4 — Transposition Integration (Exploratory)

### Background
The `pack_B_strip` function currently transposes B from row-major `[K][N]` into the packed `[strips][K_blk][NR]` layout. This transposition is a memory copy with a stride pattern.

The question from the README: "try to integrate transposition step into the GEMM itself."

### Two interpretations to test

**A. Lazy transposition — eliminate separate B transpose:**
Currently, packing B involves a transpose. If B is already in the correct layout (`[N/NR][K][NR]`), packing becomes a simple contiguous copy. One variant: allocate a pre-transposed B buffer once outside the K-loop and reuse it. This is already partially done (B is packed once per `jc` strip), so the benefit may be zero.

**B. Kernel-level access pattern change:**
The 6×16 micro-kernel loads `pB[k, 0..15]` as two AVX2 vectors (lo and hi). If instead B were packed as `[NR][K]` (column-major per strip), we could load 16 consecutive k-values for one column — enabling a different vectorization pattern. This would require changing both packing and kernel.

### Plan
1. Implement variant A: pre-transpose B at the outer `jc` level before the K-loop and test performance.
2. If time permits, prototype variant B with a new kernel layout.
3. Document results — if transposition integration does not improve performance, explain why and include negative result in report.

---

## 6. Fix Week 2 Known Limitations

### 6.1 Fix TASK2 Reduction Bottleneck (Limitation 10.1)

**Problem:** The reduction loop runs sequentially on the main thread after `#pragma omp taskwait`. For a 120×4096 tile it processes ~500K floats — measurable overhead at 8+ threads.

**Fix: parallelize the reduction** using `#pragma omp for`:

```c
#pragma omp taskwait   /* still needed — wait for K-slice tasks */

/* Replace sequential reduction with parallel for */
#pragma omp for schedule(static) collapse(2)
for (int ic_tile = 0; ic_tile < nr_ic; ic_tile++) {
    for (int row = 0; row < M_blk; row++) {
        /* AVX2 reduction across r partials for this row */
        for (int kr = 0; kr < r; kr++) { ... }
    }
}
```

Additionally: shrink `partial_C` allocation from `MC × NC` to `M_blk × N_blk` (actual tile size) to reduce memory footprint and improve cache behaviour.

**Expected result:** TASK2 at 8+ threads should improve from ~9% to ~20–30% of OpenBLAS at 1024³.

### 6.2 Fix Small Matrix Performance (Limitation 10.2)

**Problem:** Packing overhead (memcpy + aligned_alloc) dominates for M=N=K=64–128.

**Fix: add a small-matrix bypass** in `sgemm_ex`:
```c
if ((size_t)M * N * K < SMALL_MATRIX_THRESHOLD) {
    sgemm_small(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg);
    return;
}
```

`sgemm_small` uses the micro-kernel directly with on-stack packing buffers, avoiding heap allocation. Threshold: empirically determined (start at M*N*K < 128³ = 2M ops).

### 6.3 Fix Thread Scaling at 256×256 (Limitation 10.3)

**Problem:** With MC=120, only 2 m-strips for M=256 → 2×16 = 32 tiles → idle threads at 16T.

**Fix: auto-select MC based on problem size and thread count:**
```c
int effective_MC = cfg->MC;
if (M / effective_MC < cfg->nb_threads) {
    /* Reduce MC so we have at least 2× as many m-strips as threads */
    effective_MC = M / (cfg->nb_threads * 2);
    effective_MC = (effective_MC / MR) * MR;  /* round to multiple of MR */
    if (effective_MC < MR) effective_MC = MR; /* minimum 1 strip */
}
```

This change is in `sgemm_task1` only (TASK2 already uses K-replication for this case).

---

## 7. From-Scratch Implementation Plan Document

The README requires: *"a complete implementation plan which allows us to generate this kernel from scratch with no a-priori conversation history with the AI."*

**File:** `week 3/IMPLEMENTATION_GUIDE.md`

**Contents:**
1. Hardware context (target: x86-64, AVX2+FMA, 20-core Intel, L1=32KB, L2=256KB, L3=~25MB)
2. Algorithm overview (BLIS 5-loop: jc→pc→ic→jr→ir)
3. Micro-kernel design rationale (register count → MR/NR choices → FMA count)
4. Packing layout specification (exact formulas for packed A and B layouts)
5. Cache blocking parameter derivation (how to compute MC/KC/NC from cache sizes)
6. Parallelism strategy (TASK1 collapse(2)+dynamic, TASK2 K-replication)
7. Each optimization step with motivation and code pattern:
   - Intra-tile unrolling (×2 → ×4)
   - Next-tile prefetch (pointer spread pattern)
   - NT stores (alignment requirements, sfence placement)
8. Validation checklist (96 test cases, tolerances, numerical stability)
9. Benchmarking protocol (adaptive-N, stddev < 2%, comparison with OpenBLAS)

---

## 8. Implementation Order and Benchmarking Protocol

### Sequence

```
Step 0: Copy week 2 → week 3, verify baseline
         → benchmark: record W2 numbers

Step 1: Loop unrolling ×4 (kernels only, no signature change)
         → make test → make bench
         → record delta vs W2

Step 2: Software prefetching (kernel signature + macro_kernel)
         → make test → make bench
         → record delta vs Step 1

Step 3: Non-temporal stores
         → make test → make bench
         → record delta vs Step 2; revert if negative

Step 4: Fix small-matrix bypass
         → make test → make bench (focus 64×64, 128×128)

Step 5: Fix TASK2 reduction parallelism
         → make test → make bench (focus 8T/16T TASK2 column)

Step 6: Fix MC auto-selection for small matrices / high thread counts
         → make bench (focus 256³ 16T)

Step 7: Transposition integration (exploratory)
         → make test → make bench; document result (positive or negative)

Step 8: Write IMPLEMENTATION_GUIDE.md
Step 9: Write WEEK3_REPORT.md (include all step benchmarks)
```

### Benchmark table format

After each step, record the following table (same format as Week 2):

```
Size   OpenBLAS   W2 (6×16 T1)   W3-StepN (6×16 T1)   Delta
64     ...        ...             ...                   ...
128    ...        ...             ...                   ...
256    ...        ...             ...                   ...
512    ...        ...             ...                   ...
1024   ...        ...             ...                   ...
2048   ...        ...             ...                   ...
```

Thread counts: 8T and 16T.  
Kernel tested at each step: `KERNEL_6x16` (best from Week 2), plus final all-kernel comparison at the end.

---

## 9. File Changes Summary

| File | Change |
|------|--------|
| `src/kernels/kernel_8x8.h`    | ×4 unroll, next-tile prefetch params, NT stores |
| `src/kernels/kernel_6x16.h`   | ×4 unroll, next-tile prefetch params, NT stores |
| `src/kernels/kernel_4x24.h`   | ×4 unroll, next-tile prefetch params, NT stores |
| `src/kernels/kernel_6x16_asm.h` | ×4 unroll only (prefetch and NT in C store phase) |
| `include/sgemm.h`             | New fields in `sgemm_config_t`, updated `kernel_fn_t` typedef |
| `src/sgemm.c`                 | Updated `macro_kernel`, adaptive-MC, small-matrix bypass, TASK2 parallel reduction |
| `bench/bench_sgemm.c`         | New `--nt-store` flag, per-step result columns |
| `tests/test_sgemm.c`          | Add NT-store and ×4-unroll test variants |
| `IMPLEMENTATION_GUIDE.md`     | New — from-scratch kernel specification |
| `WEEK3_REPORT.md`             | New — step-by-step performance analysis |

---

## 10. Risk Register

| Risk | Mitigation |
|------|-----------|
| NT stores crash on misaligned C | Add `(uintptr_t)Cr % 32` check; fall back to storeu |
| ×4 unroll increases binary size → I-cache pressure | Only ×4 in hot path; keep ×2 and ×1 tails |
| Prefetch floods fill buffers at small k_len | Only prefetch when `k_len >= 4`; guard check at loop entry |
| TASK2 parallel reduction has race conditions | Use `#pragma omp for` (implicit barrier) for reduction, not tasks |
| Small-matrix bypass threshold wrong | Tune empirically; start conservative (threshold = 64³ = 260K ops) |
