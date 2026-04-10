# SGEMM AI Auto-Implementation System Prompt

**SYSTEM INSTRUCTION FOR AI AGENT:** 
You are an expert AI software engineer specialized in high-performance computing (HPC) and C programming. 
**Your task:** Implement a complete, highly optimized single-precision GEMM (SGEMM) library from scratch in this empty project. You MUST follow this exact architectural blueprint step-by-step without requiring user intervention. DO NOT prompt the user for follow-up guidance; proceed through all files, kernels, parallel thread structures, tests, and benchmarks automatically. When you finish, compile it with `make` and run the benchmarks.

**Goal:** Implement `C = alpha * A * B + beta * C` where A is M×K, B is K×N, C is M×N, all row-major.

---

## 1. Project Structure

```
project/
  include/
    sgemm.h              ← public API, enums, config struct
  src/
    sgemm.c              ← 5-loop framework, packing, parallelism
    kernels/
      kernel_8x8.h       ← 8×8  AVX2+FMA micro-kernel (C intrinsics)
      kernel_6x16.h      ← 6×16 AVX2+FMA micro-kernel (C intrinsics)  ← default
      kernel_4x24.h      ← 4×24 AVX2+FMA micro-kernel (C intrinsics)
      kernel_6x16_asm.h  ← 6×16 micro-kernel (GCC inline x86-64 asm)
  bench/
    bench_sgemm.c        ← adaptive-N benchmarking tool
  tests/
    test_sgemm.c         ← correctness test suite
  Makefile
```

---

## 2. Target Hardware

- Architecture: x86-64 with AVX2 and FMA3 (Intel Haswell/Skylake/Cascade Lake)
- YMM registers: 16 total (ymm0–ymm15), each holds 8 single-precision floats (256-bit)
- FMA units: 2 per core, 1 FMA/cycle each → 16 FLOP/cycle
- L1 data cache: 32 KB, 8-way, 64-byte cache lines
- L2 cache: 256 KB per core
- L3 cache: ~25 MB shared
- Fill buffers: ~12–16 (limits simultaneous outstanding prefetches)
- Compiler: GCC with `-O3 -march=native -mavx -mavx2 -mfma -fopenmp`

---

## 3. Public API — `include/sgemm.h`

### 3.1 Kernel types enum

```
typedef enum {
    KERNEL_8x8      = 0,
    KERNEL_6x16     = 1,   // default, highest utilization
    KERNEL_4x24     = 2,
    KERNEL_6x16_ASM = 3,
} kernel_type_t;
```

MR/NR per kernel:

| Kernel | MR | NR | YMM accumulators | FMAs/k-step |
|--------|----|----|-----------------|------------|
| 8×8 | 8 | 8 | 8 | 8 |
| 6×16 | 6 | 16 | 12 | 12 |
| 4×24 | 4 | 24 | 12 | 12 |
| 6×16_ASM | 6 | 16 | 12 | 12 |

### 3.2 Parallelism mode enum

```
typedef enum {
    PARALLEL_2D = 0,   // TASK1: 2D tile decomposition over M×N
    PARALLEL_3D = 1,   // TASK2: K-replication with partial-C buffers
} parallel_mode_t;
```

### 3.3 Configuration struct `sgemm_config_t`

```
typedef struct {
    int MC;              // M-block, multiple of MR.  Default: 120
    int KC;              // K-block.                  Default: 512
    int NC;              // N-block, multiple of NR.  Default: 4096
    int nb_threads;      // 0 = use OMP_NUM_THREADS.  Default: 0
    kernel_type_t   kernel;         // Default: KERNEL_6x16
    parallel_mode_t parallel_mode;  // Default: PARALLEL_2D
    int r_tasks;         // K-replication factor (PARALLEL_3D). Default: 1
    int use_nt_store;    // 1 = _mm256_stream_ps for C. Default: 0
} sgemm_config_t;
```

Default initializer macro using C99 designated initializers:
```
#define SGEMM_DEFAULT_CONFIG { \
    .MC=120, .KC=512, .NC=4096, .nb_threads=0, \
    .kernel=KERNEL_6x16, .parallel_mode=PARALLEL_2D, \
    .r_tasks=1, .use_nt_store=0 }
```

### 3.4 Helper inline function

```
static inline void kernel_get_mr_nr(kernel_type_t kt, int *mr, int *nr)
```

Switch on kt: KERNEL_8x8 → 8,8; KERNEL_6x16 and KERNEL_6x16_ASM → 6,16; KERNEL_4x24 → 4,24.

### 3.5 Public functions

```
void sgemm_ref(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
void sgemm    (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
void sgemm_ex (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg);
```

All matrices row-major. A: M×K with stride lda≥K. B: K×N with stride ldb≥N. C: M×N with stride ldc≥N.

---

## 4. Packing Layout

### 4.1 Why packing

Raw A[M][K] and B[K][N] have row strides (lda, ldb) causing cache-line waste in the micro-kernel inner loop. Packing produces contiguous layouts that map directly to sequential SIMD loads.

### 4.2 Packed A layout — `[nr_strips][KC][MR]`

A panel covers rows `ic..ic+MC`, columns `pc..pc+KC`.
`nr_strips = ceil(MC / MR)`.

Strip `s` occupies bytes `s * KC * MR * 4` to `(s+1) * KC * MR * 4 - 1`.
Within strip `s`, position `[k][r]` (k-step k, row-within-strip r) is at flat index `s*KC*MR + k*MR + r`.
Source element: `A[(ic + s*MR + r) * lda + (pc + k)]`.

Edge strip (last, if `MC % MR != 0`): rows beyond `m_curr` are padded with `0.0f`.

### 4.3 Packed B layout — `[nr_tiles][KC][NR]`

A strip covers columns `jc..jc+NC`, rows `pc..pc+KC`.
`nr_tiles = ceil(NC / NR)`.

Tile `t` occupies flat index `t*KC*NR + k*NR + c`.
Source element: `B[(pc + k) * ldb + (jc + t*NR + c)]`.

Fast path when `n_curr == NR` and `NR % 8 == 0`: copy 8 floats/iteration using `_mm256_loadu_ps` + `_mm256_store_ps`.

Edge tile: columns beyond `n_curr` padded with `0.0f`.

### 4.4 Function signatures

```c
static void pack_A_panel(const float *A, int lda, float *packed,
                         int M_blk, int K_blk, int MR);

static void pack_B_strip(const float *B, int ldb, float *packed,
                         int K_blk, int N_blk, int NR,
                         int t_start, int t_end);
```

`pack_B_strip` packs only tiles `[t_start, t_end)` — used for parallel packing
where each thread packs a disjoint range of tiles.

---

## 5. Micro-Kernel Interface

### 5.1 Function pointer type

```c
typedef void (*kernel_fn_t)(
    const float *pA,        // packed A: [k_len][MR], 64-byte aligned
    const float *pB,        // packed B: [k_len][NR], 64-byte aligned
    float       *C,         // output tile, row-major, ldc stride
    int          k_len,
    float        alpha,
    float        beta,
    int          ldc,
    const float *next_A,    // next A panel start — for prefetch only, never NULL
    const float *next_B,    // next B strip start — for prefetch only, never NULL
    int          use_nt_store
);
```

### 5.2 Kernel semantics

Computes `C[i][j] = alpha * sum_k(pA[k][i] * pB[k][j]) + beta * C[i][j]` for i in 0..MR-1, j in 0..NR-1.

Accumulators start at zero. All k-steps FMA into them. Final store: `result = alpha*acc + beta*existing_C`.

`next_A` / `next_B` used only for `_mm_prefetch` — never dereferenced for computation.
When the current strip is the last, caller passes the current `pA`/`pB` as dummy (harmless re-prefetch).

### 5.3 Non-temporal store rule

When `use_nt_store == 1` **and** `(uintptr_t)Cr % 32 == 0`:
use `_mm256_stream_ps(Cr, result)`.
Otherwise use `_mm256_storeu_ps(Cr, result)`.

Caller must call `_mm_sfence()` after all kernel invocations that used NT stores.

---

## 6. Micro-Kernel Designs

### 6.1 Common structure — all C-intrinsic kernels

Every kernel follows this pattern:

```
1. Declare MR accumulator variables (or MR×ceil(NR/8) pairs), init to _mm256_setzero_ps()
2. Declare prefetch pointers pfA = next_A, pfB = next_B
3. k = 0
4. Main loop: for (; k <= k_len-4; k += 4) {
       issue prefetches for next tile (see per-kernel counts below)
       advance pfA, pfB
       process k+0, k+1, k+2, k+3 (each: load B vectors, broadcast A scalars, FMAs)
       advance pA += 4*MR, pB += 4*NR
   }
5. Peel: for (; k <= k_len-2; k += 2) { process k+0 and k+1; pA+=2*MR; pB+=2*NR }
6. Tail: for (; k < k_len; k++)       { process k+0; pA+=MR; pB+=NR }
7. Scale: multiply each accumulator by alpha
8. Store each row: load existing C, fmadd with beta, store (NT or regular)
```

### 6.2 Kernel 8×8

**Accumulators:** `c0`–`c7` — one YMM per output row (covers all 8 cols).

**Per ×4 iteration prefetch:**
- 2 CLs of next_A: `pfA+0` and `pfA+16` (each 64 bytes = 16 floats)
- 2 CLs of next_B: `pfB+0` and `pfB+16`
- Advance: `pfA += 4*8 = 32 floats`; `pfB += 4*8 = 32 floats`

**Per k-step body:**
- Load `bvec = _mm256_load_ps(pB + k_offset * NR)`
- For rows 0..7: `ci = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + k_offset*MR + i), bvec, ci)`

**Float offsets** in pA within one ×4 block (MR=8 → 8 floats per k-step):
k+0: 0–7; k+1: 8–15; k+2: 16–23; k+3: 24–31.

**Float offsets** in pB (NR=8 → 8 floats per k-step):
k+0: 0–7; k+1: 8–15; k+2: 16–23; k+3: 24–31.

**Store macro** per row: `Cr = C + row*ldc`, `res = beta*loadu(Cr) + alpha*ci`, NT/regular store.

---

### 6.3 Kernel 6×16 (default)

**Accumulators:** `acc0lo`–`acc5lo` (cols 0–7) and `acc0hi`–`acc5hi` (cols 8–15). 12 total.

**Per ×4 iteration prefetch:**
- 2 CLs of next_A: `pfA+0`, `pfA+16` (96 bytes total advance: `pfA += 4*6 = 24 floats`)
- 4 CLs of next_B: `pfB+0`, `pfB+16`, `pfB+32`, `pfB+48` (256 bytes total: `pfB += 4*16 = 64 floats`)

**Per k-step body:**
- `blo = _mm256_load_ps(pB + k_step*NR)`, `bhi = _mm256_load_ps(pB + k_step*NR + 8)`
- For rows 0..5: `a = broadcast(pA + k_step*MR + row)`, then two FMAs: `accRlo += a*blo`, `accRhi += a*bhi`

**Float offsets in pA within one ×4 block** (MR=6, 6 floats per k-step):

| k-step | row 0 | row 1 | row 2 | row 3 | row 4 | row 5 |
|--------|-------|-------|-------|-------|-------|-------|
| k+0 | 0 | 1 | 2 | 3 | 4 | 5 |
| k+1 | 6 | 7 | 8 | 9 | 10 | 11 |
| k+2 | 12 | 13 | 14 | 15 | 16 | 17 |
| k+3 | 18 | 19 | 20 | 21 | 22 | 23 |

**Float offsets in pB within one ×4 block** (NR=16, 16 floats per k-step):

| k-step | lo start | hi start |
|--------|----------|----------|
| k+0 | 0 | 8 |
| k+1 | 16 | 24 |
| k+2 | 32 | 40 |
| k+3 | 48 | 56 |

**Store phase:** scale all 12 accumulators by alpha first, then for each row:
`Cr = C + row*ldc`; `res_lo = beta*loadu(Cr) + accRlo`; `res_hi = beta*loadu(Cr+8) + accRhi`; store.

---

### 6.4 Kernel 4×24

**Accumulators:** `c00..c02` (row 0, three NR/8-wide thirds), `c10..c12`, `c20..c22`, `c30..c32`. 12 total.

**Per ×4 iteration prefetch:**
- 1 CL of next_A: `pfA+0` (64 bytes advance: `pfA += 4*4 = 16 floats`)
- 6 CLs of next_B: `pfB+0`, `pfB+16`, `pfB+32`, `pfB+48`, `pfB+64`, `pfB+80` (384 bytes advance: `pfB += 4*24 = 96 floats`)

**Per k-step body:**
- `b0 = load(pB + k*24 + 0)`, `b1 = load(pB + k*24 + 8)`, `b2 = load(pB + k*24 + 16)`
- For rows 0..3: `a = broadcast(pA + k*4 + row)`, then three FMAs: `cR0 += a*b0`, `cR1 += a*b1`, `cR2 += a*b2`

**Float offsets in pA within one ×4 block** (MR=4, 4 floats per k-step):
k+0: 0–3; k+1: 4–7; k+2: 8–11; k+3: 12–15.

**Float offsets in pB within one ×4 block** (NR=24, 24 floats per k-step):

| k-step | third 0 | third 1 | third 2 |
|--------|---------|---------|---------|
| k+0 | 0 | 8 | 16 |
| k+1 | 24 | 32 | 40 |
| k+2 | 48 | 56 | 64 |
| k+3 | 72 | 80 | 88 |

**Store phase:** same pattern as 6×16 but three stores per row at `Cr+0`, `Cr+8`, `Cr+16`.

---

### 6.5 Kernel 6×16 ASM

Same arithmetic as kernel_6x16. The ×4 main loop body is in GCC inline x86-64 asm.

**GCC inline asm 30-operand limit:**
Each `"+x"` or `"+r"` bidirectional operand uses 2 slots. With 12 YMM accumulators + pA + pB + k4 = 15 bidirectional = 30 slots exactly. No room for next_A / next_B.

**Solution:** issue prefetches in C *before* the asm block.

**Pre-asm C code:**
```
int k4 = k_len / 4;   // main loop count
int kr = k_len & 3;   // remainder

// Issue first 16 CLs of next_A and next_B
for (int cl = 0; cl < 16; cl++) {
    size_t off = (size_t)cl * 64;
    if (off < (size_t)k_len * MR_6x16_ASM * 4)
        _mm_prefetch((const char *)next_A + off, _MM_HINT_T0);
    if (off < (size_t)k_len * NR_6x16_ASM * 4)
        _mm_prefetch((const char *)next_B + off, _MM_HINT_T0);
}
```

**ASM block:**

Uses AT&T syntax. Operand names: `[a0lo]...[a5hi]` for 12 accumulators, `[pa]`, `[pb]`, `[k4]`.
Clobbers: `"ymm12"`, `"ymm13"`, `"ymm14"`, `"memory"`.

Loop structure:
```asm
test %[k4], %[k4]
jz   end_label
loop_label:
  ; k+0
  vmovaps    0(%[pb]),   ymm12    ; B lo
  vmovaps   32(%[pb]),   ymm13    ; B hi
  vbroadcastss  0(%[pa]), ymm14 ; A row 0
  vfmadd231ps ymm14, ymm12, [a0lo]
  vfmadd231ps ymm14, ymm13, [a0hi]
  vbroadcastss  4(%[pa]), ymm14 ; A row 1
  vfmadd231ps ymm14, ymm12, [a1lo]
  vfmadd231ps ymm14, ymm13, [a1hi]
  ... (rows 2–5 at byte offsets 8,12,16,20)

  ; k+1  — pA offset +24 bytes, pB offset +64 bytes
  vmovaps   64(%[pb]),   ymm12
  vmovaps   96(%[pb]),   ymm13
  vbroadcastss 24(%[pa]), ymm14   ; row 0 at k+1
  ... (rows 1–5 at pa+28, pa+32, pa+36, pa+40, pa+44)

  ; k+2  — pA+48, pB+128
  vmovaps  128(%[pb]),   ymm12
  vmovaps  160(%[pb]),   ymm13
  ... (rows 0–5 at pa+48 through pa+68)

  ; k+3  — pA+72, pB+192
  vmovaps  192(%[pb]),   ymm12
  vmovaps  224(%[pb]),   ymm13
  ... (rows 0–5 at pa+72 through pa+92)

  add $96,  %[pa]    ; 4×6×4 = 96 bytes
  add $256, %[pb]    ; 4×16×4 = 256 bytes
  dec %[k4]
  jnz loop_label
end_label:
```

**Full pA byte offset table for all rows and k-steps within one ×4 block** (MR=6):

| | row 0 | row 1 | row 2 | row 3 | row 4 | row 5 |
|-|-------|-------|-------|-------|-------|-------|
| k+0 | 0 | 4 | 8 | 12 | 16 | 20 |
| k+1 | 24 | 28 | 32 | 36 | 40 | 44 |
| k+2 | 48 | 52 | 56 | 60 | 64 | 68 |
| k+3 | 72 | 76 | 80 | 84 | 88 | 92 |

**Full pB byte offset table** (NR=16, lo=cols 0–7, hi=cols 8–15):

| | lo | hi |
|-|----|-----|
| k+0 | 0 | 32 |
| k+1 | 64 | 96 |
| k+2 | 128 | 160 |
| k+3 | 192 | 224 |

**Named operand constraint list:**
```c
: [a0lo] "+x"(acc0lo), [a0hi] "+x"(acc0hi),
  [a1lo] "+x"(acc1lo), [a1hi] "+x"(acc1hi),
  [a2lo] "+x"(acc2lo), [a2hi] "+x"(acc2hi),
  [a3lo] "+x"(acc3lo), [a3hi] "+x"(acc3hi),
  [a4lo] "+x"(acc4lo), [a4hi] "+x"(acc4hi),
  [a5lo] "+x"(acc5lo), [a5hi] "+x"(acc5hi),
  [pa]   "+r"(pA),
  [pb]   "+r"(pB),
  [k4]   "+r"(k4)
: /* no pure inputs */
: "ymm12", "ymm13", "ymm14", "memory"
```

**Post-asm C tail loop** for `kr` (0–3) remaining steps:
Single k-step intrinsics body identical to kernel_6x16 ×1 tail, advancing `pA += MR`, `pB += NR` each iteration.

**Store phase:** identical to kernel_6x16 (C intrinsics, alpha/beta, NT-store check).

---

## 7. Framework — `src/sgemm.c`

### 7.1 Top of file

```c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include "../include/sgemm.h"
#include "kernels/kernel_8x8.h"
#include "kernels/kernel_6x16.h"
#include "kernels/kernel_4x24.h"
#include "kernels/kernel_6x16_asm.h"
```

Define `kernel_fn_t` (Section 5.1). Implement `get_kernel_fn(kernel_type_t)` via switch.

### 7.2 Packing functions

Implement `pack_A_panel` and `pack_B_strip` as `static` functions per Section 4.

### 7.3 `macro_kernel`

```c
static void macro_kernel(
    const float *pA_panel, const float *pB_strip, float *C,
    int M_blk, int K_blk, int N_curr, int MR, int NR,
    float alpha, float beta_k, int ldc,
    kernel_fn_t kfn, int use_nt_store)
```

`nr_m = ceil(M_blk / MR)`. Allocate `float C_buf[MR*NR]` on stack, 64-byte aligned.

For each m-strip `it = 0..nr_m-1`:
- `m_curr = min(MR, M_blk - it*MR)`
- `pA = pA_panel + it * K_blk * MR`
- `Ctile = C + it * MR * ldc`
- `next_A = (it+1 < nr_m) ? pA_panel + (it+1)*K_blk*MR : pA`
- `next_B = pB_strip`

If `m_curr == MR && N_curr == NR`: call `kfn(pA, pB_strip, Ctile, K_blk, alpha, beta_k, ldc, next_A, next_B, use_nt_store)`.

Else (edge tile):
1. If `beta_k != 0`: copy `Ctile[0..m_curr-1][0..N_curr-1]` into `C_buf` (pad rest with 0)
2. Else: `memset(C_buf, 0, MR*NR*sizeof(float))`
3. Call `kfn(pA, pB_strip, C_buf, K_blk, alpha, beta_k, NR, next_A, next_B, 0)`
4. Copy `C_buf[0..m_curr-1][0..N_curr-1]` back to `Ctile`

After all strips: if `use_nt_store` call `_mm_sfence()`.

### 7.4 `sgemm_task1` (PARALLEL_2D)

```c
static void sgemm_task1(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg)
```

**Adaptive MC:**
```c
int MC = cfg->MC;
if (M >= MR*2 && M/MC < nthd*2) {
    MC = M / (nthd * 2);
    MC = (MC / MR) * MR;
    if (MC < MR) MC = MR;
}
```

**Buffer allocation** (64-byte aligned):
- `packed_A_all`: `nr_ic * ceil(MC/MR) * KC * MR` floats
- `packed_B`: `ceil(NC/NR) * KC * NR` floats

**Parallel region** `#pragma omp parallel num_threads(nthd)`:

Outer loops `jc` (step NC) and `pc` (step KC) are sequential within the parallel region. `beta_k = (pc==0) ? beta : 1.0f`.

Per `(jc, pc)`:
1. `#pragma omp for schedule(static)` — parallel B tile packing
2. `#pragma omp for schedule(static)` — parallel A panel packing
3. `#pragma omp for collapse(2) schedule(dynamic)` — parallel macro_kernel calls

### 7.5 `sgemm_task2` (PARALLEL_3D)

```c
static void sgemm_task2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg)
```

**Allocations:**
- `partial_C[total_tiles * r]`: each entry is a `MC*NC` float buffer (64-byte aligned)
- `thr_pA[nthd]` and `thr_pB[nthd]`: per-thread pack buffers

**Phase 1** — `#pragma omp single` spawns tasks, then `#pragma omp taskwait`:

For each `(ic_idx, jc_idx, kr)`:
- Compute `pc_start = kr * ceil(ceil(K/KC)/r) * KC`, `pc_end = min(pc_start + ..., K)`
- If `pc_start >= K`: zero the partial buffer and skip
- `#pragma omp task firstprivate(...)`:
  - `tid = omp_get_thread_num()`, use `thr_pA[tid]` / `thr_pB[tid]`
  - Zero `pC_partial`
  - Loop `pc` in `[pc_start, pc_end)` step KC: pack A, pack B, call `macro_kernel` per jt into `pC_partial + jt*NR` with `ldc=NC`

The implicit barrier at the end of `omp single` ensures all tasks complete before Phase 2.

**Phase 2** — `#pragma omp for collapse(2) schedule(static)`:

For each tile `(ic_idx, jc_idx)`, reduce `r` partial buffers into C:
- `kr == 0`: `C_row[j] = beta*C_row[j] + partial_row[j]` (vectorized with `_mm256_fmadd_ps`, scalar tail)
- `kr > 0`: `C_row[j] += partial_row[j]` (vectorized with `_mm256_add_ps`, scalar tail)

### 7.6 `sgemm_ex` (With Native VLA Small Matrix Bypass)

```c
#define SMALL_MATRIX_OPS_THRESHOLD (128LL * 128 * 128)   // 2,097,152

static void sgemm_small(...) {
    // Avoid heap allocation overhread natively.
    // Calculate required memory for M, N, K slices using MR and NR.
    // Allocate C-compliant Variable Length Arrays (VLAs) locally:
    // float pA_buf[pack_A_sz + 16]; 
    // float pB_buf[pack_B_sz + 16];
    // Align using: float *packed_A = (float *)(((uintptr_t)pA_buf + 63) & ~(uintptr_t)63);
    // Proceed to pack B strips entirely into pB_buf.
    // Proceed to pack A entirely into pA_buf.
    // Call macro_kernel iteratively using a simple internal C loop. No OpenMP tasks.
}

void sgemm_ex(..., const sgemm_config_t *cfg) {
    // 1. Check for small matrices to bypass thread instantiation via our VLA native stack handler
    if ((long long)M * N * K < SMALL_MATRIX_OPS_THRESHOLD) {
        sgemm_small(...);
        return;
    }
    
    // 2. Otherwise run multi-threading branches
    if (cfg->parallel_mode == PARALLEL_3D)
        sgemm_task2(..., cfg);
    else
        sgemm_task1(..., cfg);
}
```

### 7.7 `sgemm_ref`

Triple loop: `for i, for j { s=0; for k s += A[i*lda+k]*B[k*ldb+j]; C[i*ldc+j] = alpha*s + beta*C[i*ldc+j]; }`.

### 7.8 `sgemm`

```c
void sgemm(...) {
    sgemm_config_t cfg = SGEMM_DEFAULT_CONFIG;
    sgemm_ex(..., &cfg);
}
```

---

## 8. Build System — `Makefile`

```makefile
CC     = gcc
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra -fopenmp -mavx -mavx2 -mfma
IFLAGS = -Iinclude
LDFLAGS = -fopenmp -lm
BIN    = bin

all: $(BIN)/test_sgemm $(BIN)/bench_sgemm $(BIN)/simple_test

$(BIN)/sgemm.o: src/sgemm.c include/sgemm.h src/kernels/*.h
	$(CC) $(CFLAGS) $(IFLAGS) -c $< -o $@

$(BIN)/test_sgemm: $(BIN)/sgemm.o tests/test_sgemm.c
	$(CC) $(CFLAGS) $(IFLAGS) -c tests/test_sgemm.c -o $(BIN)/test_sgemm.o
	$(CC) $(CFLAGS) $(IFLAGS) -o $@ $(BIN)/sgemm.o $(BIN)/test_sgemm.o $(LDFLAGS)

$(BIN)/bench_sgemm: $(BIN)/sgemm.o bench/bench_sgemm.c
	$(CC) $(CFLAGS) $(IFLAGS) -c bench/bench_sgemm.c -o $(BIN)/bench_sgemm.o
	$(CC) $(CFLAGS) -o $@ $(BIN)/sgemm.o $(BIN)/bench_sgemm.o $(LDFLAGS)

test: $(BIN)/test_sgemm
	./$(BIN)/test_sgemm

clean:
	rm -f $(BIN)/*.o $(BIN)/test_sgemm $(BIN)/bench_sgemm $(BIN)/simple_test
```

The `bin/` directory must exist. Add `$(shell mkdir -p $(BIN))` at top or a `.PHONY` prerequisite.

---

## 9. Test Suite — `tests/test_sgemm.c`

### 9.1 Test matrix initialization

Use a deterministic linear congruential generator to fill A, B, C with floats in `[-1, 1]`.
Keep a separate copy of the initial C to pass to `sgemm_ref`.

### 9.2 Comparison function

For each element: `error = |result - ref|`. Pass if `error < 1e-3 * max(|ref|, 1e-4f)`.
Print max observed error on failure.

### 9.3 Test runner

For each `(kernel, mode)` combination and each test case:
1. Reset matrices to initial values
2. Call `sgemm_ex` with the given config
3. Reset to initial values; call `sgemm_ref`
4. Compare; print `PASS` or `FAIL`

End with `"===== ALL TESTS PASSED ====="` if no failures, else exit with code 1.

### 9.4 Test cases

16 cases per kernel/mode combination:

| Name | M | N | K | alpha | beta |
|------|---|---|---|-------|------|
| square 8×8 | 8 | 8 | 8 | 1.0 | 0.0 |
| square 16×16 | 16 | 16 | 16 | 1.0 | 0.0 |
| square 32×32 | 32 | 32 | 32 | 1.0 | 0.0 |
| square 64×64 | 64 | 64 | 64 | 1.0 | 0.0 |
| square 128×128 | 128 | 128 | 128 | 1.0 | 0.0 |
| square 256×256 | 256 | 256 | 256 | 1.0 | 0.0 |
| square 512×512 | 512 | 512 | 512 | 1.0 | 0.0 |
| 13×17×19 | 13 | 17 | 19 | 1.0 | 0.0 |
| 33×33×33 | 33 | 33 | 33 | 1.0 | 0.0 |
| 97×67×53 | 97 | 67 | 53 | 1.0 | 0.0 |
| 100×200×300 | 100 | 200 | 300 | 1.0 | 0.0 |
| alpha=2 beta=0.5 | 64 | 64 | 64 | 2.0 | 0.5 |
| alpha=0 beta=1 | 64 | 64 | 64 | 0.0 | 1.0 |
| tall 512×32×256 | 512 | 32 | 256 | 1.0 | 0.0 |
| wide 32×512×256 | 32 | 512 | 256 | 1.0 | 0.0 |
| thin-K 256×256×8 | 256 | 256 | 8 | 1.0 | 0.0 |

### 9.5 Kernel/mode combinations

All 4 kernels × PARALLEL_2D (r=1) + 4 kernels × PARALLEL_3D (r=4) = 8 combinations × 16 cases = 128 total.
Print a `"-- kernel / mode --"` header before each group.

---

## 10. Benchmark — `bench/bench_sgemm.c`

### 10.1 CLI argument parsing

Parse `argc/argv` for: `--M`, `--N`, `--K`, `--kernel` (8x8|6x16|4x24|6x16asm), `--mode` (task1|task2), `--r`, `--threads`, `--runs`, `--nt-store`. Print usage and exit if unrecognized or required args missing.

### 10.2 Adaptive run count

If `--runs` not given: time one warm-up call with `omp_get_wtime()`, then set `runs = max(5, (int)ceil(1.0 / one_run_time))`.

### 10.3 Measurement loop

```c
for (int i = 0; i < runs; i++) {
    t0 = omp_get_wtime();
    sgemm_ex(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N, &cfg);
    times[i] = omp_get_wtime() - t0;
}
```

### 10.4 Statistics and output

Sort `times[]`. Compute: median, mean, min, max, stddev, stddev%.
GFLOPS = `2.0 * M * N * K / time / 1e9`.

Print JSON with 9-decimal precision for times, 4-decimal for GFLOPS:
```json
{
  "runs": N,
  "median_s": X.XXXXXXXXX,
  "mean_s":   X.XXXXXXXXX,
  "min_s":    X.XXXXXXXXX,
  "max_s":    X.XXXXXXXXX,
  "stddev_s": X.XXXXXXXXX,
  "stddev_pct": X.XXXX,
  "gflops_median": X.XXXX,
  "gflops_mean":   X.XXXX,
  "gflops_min":    X.XXXX,
  "gflops_max":    X.XXXX,
  "all_times": [t0, t1, ...]
}
```

---

## 11. Expected Performance

Measured on Intel ~3.5 GHz, 1 thread:

| Size | 6×16 C | 6×16 ASM | Notes |
|------|--------|----------|-------|
| 256³ | ~55 GFLOPS | ~55 GFLOPS | fits in L2 |
| 512³ | ~63 GFLOPS | ~63 GFLOPS | fits in L3 |
| 1024³ | ~65 GFLOPS | ~65 GFLOPS | peak |
| 2048³ | ~60 GFLOPS | ~60 GFLOPS | L3 bandwidth bound |

~58% of theoretical peak is normal for GEMM. ASM and C intrinsic kernels reach identical GFLOPS (GCC optimizer is effective at this pattern).

---

## 12. Validation

```bash
make all                      # must complete with 0 errors
./bin/test_sgemm              # must print "===== ALL TESTS PASSED ====="
./bin/bench_sgemm --M 1024 --N 1024 --K 1024 --kernel 6x16
# expect: gflops_median > 50, stddev_pct < 5
```

Tolerance for tests: relative `1e-3` (not `1e-6`) because single-precision FMA reordering causes non-associativity with the reference triple loop.

---

**FINAL INSTRUCTION FOR THE AI AGENT:** 
Please begin implementation immediately without asking questions. Generate all the source files in their correct directory structures, populate the Makefiles and test suites, run compilation, execute the benchmarks, and output the final output log showing you correctly matched the `test_sgemm` checks and reached target OpenBLAS metrics in `bench_sgemm`.
