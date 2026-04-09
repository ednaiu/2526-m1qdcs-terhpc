# SGEMM Optimization — From-Scratch Implementation Guide

**Purpose:** A complete specification allowing an AI (or developer) to regenerate the full
optimized SGEMM kernel from zero, without any prior conversation history.

**Target hardware:** x86-64 with AVX2 + FMA3 (Intel Haswell and later)  
**Target FLOPS:** ~65 GFLOPS single-threaded at 1024³  
**Language:** C99 + GCC inline assembly (x86-64 AT&T syntax)

---

## 1. Hardware Context

### CPU characteristics (Intel Skylake/Cascade Lake, 20-core)

| Parameter         | Value                              |
|-------------------|------------------------------------|
| SIMD width        | 256-bit (AVX2), 8 floats/register  |
| YMM registers     | 16 (ymm0–ymm15)                    |
| FMA throughput    | 2 FMA units, 1 FMA/cycle each = 16 FLOP/cycle |
| L1 data cache     | 32 KB, 8-way, 64-byte cache lines  |
| L2 cache          | 256 KB per core, 4-way             |
| L3 cache          | ~25 MB shared                      |
| Cache line size   | 64 bytes = 16 floats               |
| Fill buffers      | ~12–16 per core                    |

**Theoretical peak (single core):** `clock × 2 FMA × 8 floats × 2 ops/FMA`

### Key constraints for GEMM

- L1 holds ~8K floats. A micro-kernel's A panel + B panel must fit.
- L2 holds the working set for an entire K-block column (KC × NR floats).
- NT stores (`_mm256_stream_ps`) need 32-byte aligned addresses and require `_mm_sfence()` before cross-thread visibility.

---

## 2. Algorithm: BLIS 5-Loop GEMM (GotoBLAS)

The standard high-performance GEMM decomposition uses 5 nested loops to maximize cache reuse.
From outermost to innermost:

```
for jc in 0..N step NC:          ← Loop 5 (L5): N-dimension strips
  pack_B(B[0..K, jc..jc+NC])     ← pack B strip into Bp[NC/NR][K/KC][KC][NR]
  for pc in 0..K step KC:         ← Loop 4 (L4): K-dimension blocks
    for ic in 0..M step MC:       ← Loop 3 (L3): M-dimension panels
      pack_A(A[ic..ic+MC, pc..pc+KC]) ← pack A panel into Ap[MC/MR][KC][MR]
      for jr in 0..NC step NR:    ← Loop 2 (L2): micro-panel columns
        for ir in 0..MC step MR:  ← Loop 1 (L1): micro-kernel rows
          micro_kernel(pA, pB, C+ir*ldc+jr, KC, alpha, beta)
```

### Cache assignment

| Data | Target cache | Layout after packing |
|------|-------------|----------------------|
| Micro-kernel A panel (MR × KC floats) | L1 | Row-major: `[KC][MR]` |
| Micro-kernel B panel (KC × NR floats) | L1 | Row-major: `[KC][NR]` |
| Full A macro-panel (MC × KC floats)   | L2 | Tiled: `[MC/MR][KC][MR]` |
| Full B strip (KC × NC floats)         | L3 | Tiled: `[NC/NR][KC][NR]` |

### Why packing is necessary

Raw A and B have stride-lda row access patterns, causing cache misses in the micro-kernel inner loop. Packing creates contiguous memory layouts matched to SIMD access patterns, enabling sequential loads with hardware prefetch.

---

## 3. Blocking Parameter Selection

### Formula-based derivation

**MR and NR** (micro-kernel tile dimensions):
- Chosen so that MR + 2 YMM registers per k-step fits in 16 YMM total.
- For 6×16: 12 accumulators (6 rows × 2 halves) + 2 B regs + 1 A broadcast = 15 YMM.
- For 8×8: 8 accumulators + 1 B reg + 1 A broadcast = 10 YMM.
- For 4×24: 12 accumulators (4 rows × 3 thirds) + 3 B regs + 1 A broadcast = 16 YMM.

**KC** (K-block size): Sized so that A + B micro-panels fit in L1:
```
MR*KC*4 + NR*KC*4 <= L1_size * reuse_factor
KC = L1_size * 0.5 / ((MR + NR) * 4)
```
For 6×16: KC ≈ 32768*0.5 / ((6+16)*4) = 186 → round to 256 (power of 2 friendly).

**MC** (M-block size): Sized so that A macro-panel fits in L2:
```
MC*KC*4 <= L2_size * 0.8
MC = 204800 / (KC * 4) = 200 (for KC=256) → round to 120 (multiple of MR=6)
```

**NC** (N-block size): Sized so that B strip fits in L3:
```
KC*NC*4 <= L3_size * 0.3
NC = L3_size*0.3 / (KC*4) = ~9600 for KC=256, L3=25MB → use 3072 (multiple of NR=16)
```

### Default configuration
```c
MC = 120,  NC = 3072,  KC = 256,  MR = 6,  NR = 16   /* 6×16 kernel */
MC = 64,   NC = 3072,  KC = 256,  MR = 8,  NR = 8    /* 8×8  kernel */
MC = 64,   NC = 3072,  KC = 256,  MR = 4,  NR = 24   /* 4×24 kernel */
```

---

## 4. Packing Layout Specification

### pack_A: A[M][K] → Ap[MC/MR][KC][MR]

For each m-strip `(ic..ic+MC)`, for each k-block `(pc..pc+KC)`:
```
Ap[mi/MR * KC*MR + ki*MR + row_offset]
  where mi = 0..MC, ki = 0..KC
  row_offset = mi % MR
```
Access pattern: read A at `A[ic + mi][pc + ki]` with stride `lda`, write sequentially.

**Edge handling:** If `MC_actual < MC` or `KC_actual < KC`, pad with zeros to maintain full panel size.

### pack_B: B[K][N] → Bp[NC/NR][KC][NR]

For each n-strip `(jc..jc+NC)`, for each k-block `(pc..pc+KC)`:
```
Bp[ni/NR * KC*NR + ki*NR + col_offset]
  where ni = 0..NC, ki = 0..KC
  col_offset = ni % NR
```
Access pattern: read B at `B[pc + ki][jc + ni]` with stride `ldb`, write sequentially.

---

## 5. Micro-Kernel Design

### 5.1 Register Plan (6×16 kernel)

```
ymm0  : accumulator row 0, cols 0–7    (acc0lo)
ymm1  : accumulator row 0, cols 8–15   (acc0hi)
ymm2  : accumulator row 1, cols 0–7    (acc1lo)
ymm3  : accumulator row 1, cols 8–15   (acc1hi)
...
ymm10 : accumulator row 5, cols 0–7    (acc5lo)
ymm11 : accumulator row 5, cols 8–15   (acc5hi)
ymm12 : B vector, cols 0–7             (loaded each k-step)
ymm13 : B vector, cols 8–15            (loaded each k-step)
ymm14 : A broadcast scalar             (one float → all 8 lanes)
ymm15 : unused / scratch
```

**FMA count per k-step:** 12 FMAs (6 rows × 2 B halves)  
**FLOP per k-step:** 12 × 2 = 24  
**FLOP per k-step (full 6×16 tile):** 6 × 16 × 2 = 192

### 5.2 Single k-step pseudocode

```c
__m256 blo = _mm256_load_ps(pB);           // cols 0–7
__m256 bhi = _mm256_load_ps(pB + 8);       // cols 8–15
for (int row = 0; row < MR; row++) {
    __m256 a = _mm256_broadcast_ss(pA + row);
    acc[row][0] = _mm256_fmadd_ps(a, blo, acc[row][0]);
    acc[row][1] = _mm256_fmadd_ps(a, bhi, acc[row][1]);
}
pA += MR;    // advance to next row of packed A
pB += NR;    // advance to next row of packed B
```

### 5.3 K-loop structure (×4 unrolled)

```c
int k = 0;
// Main loop: unrolled ×4 for reduced branch overhead
for (; k <= k_len - 4; k += 4) {
    // k+0: load blo0/bhi0, fmadd all rows
    // k+1: load blo1/bhi1, fmadd all rows
    // k+2: load blo2/bhi2, fmadd all rows
    // k+3: load blo3/bhi3, fmadd all rows
    pA += 4 * MR;
    pB += 4 * NR;
}
// ×2 peel for k_len % 4 == 2 or 3
for (; k <= k_len - 2; k += 2) { ... pA += 2*MR; pB += 2*NR; }
// ×1 tail for k_len % 2 == 1
for (; k < k_len; k++) { ... pA += MR; pB += NR; }
```

**Why ×4:** Gives the out-of-order engine a 4× larger instruction window to overlap
FMA latency (4 cycles) with loads and pointer arithmetic. Empirically: 2–8% gain vs ×2.

### 5.4 Store phase

```c
__m256 va = _mm256_set1_ps(alpha);
__m256 vb = _mm256_set1_ps(beta);

for (int row = 0; row < MR; row++) {
    float *Cr = C + row * ldc;
    __m256 existing_lo = _mm256_loadu_ps(Cr);
    __m256 existing_hi = _mm256_loadu_ps(Cr + 8);
    __m256 res_lo = _mm256_fmadd_ps(vb, existing_lo, _mm256_mul_ps(va, acc[row][0]));
    __m256 res_hi = _mm256_fmadd_ps(vb, existing_hi, _mm256_mul_ps(va, acc[row][1]));

    if (use_nt_store && ((uintptr_t)Cr % 32 == 0)) {
        _mm256_stream_ps(Cr,     res_lo);   // NT store: bypass cache
        _mm256_stream_ps(Cr + 8, res_hi);
    } else {
        _mm256_storeu_ps(Cr,     res_lo);
        _mm256_storeu_ps(Cr + 8, res_hi);
    }
}
```

**NT stores:** Use `_mm256_stream_ps` when `use_nt_store=1` and address is 32-byte aligned.
Bypasses L1/L2 on write, keeping cache free for A and B panels. Requires `_mm_sfence()` after
all kernel calls in the caller to ensure memory ordering.

---

## 6. Software Prefetching

### 6.1 Motivation

When the CPU finishes micro-kernel `(ir, jr)` and begins `(ir+1, jr)`, it must load a new
A panel from L2 → L1. This takes ~10–15 cycles on a cache miss. Software prefetch during the
previous kernel's k-loop hides this latency.

### 6.2 Kernel signature extension

```c
typedef void (*kernel_fn_t)(
    const float *pA,        // current A panel (packed)
    const float *pB,        // current B panel (packed)
    float       *C,         // output tile (unpacked, row-major)
    int          k_len,     // number of k-steps
    float        alpha,
    float        beta,
    int          ldc,       // leading dimension of C
    const float *next_A,    // start of NEXT A panel (for prefetch)
    const float *next_B,    // start of NEXT B strip (for prefetch)
    int          use_nt_store
);
```

### 6.3 Prefetch spread pattern

Issue one `_mm_prefetch` per cache line of the next tile, spread across k-iterations:

```c
const float *pfA = next_A;
const float *pfB = next_B;

for (; k <= k_len - 4; k += 4) {
    // Each ×4 iter consumes 4×MR floats of A = 4×6×4=96 bytes = 2 CLs (6×16)
    _mm_prefetch((const char *)pfA,      _MM_HINT_T0);
    _mm_prefetch((const char *)(pfA+16), _MM_HINT_T0);
    pfA += 4 * MR;

    // B: 4×NR floats = 4×16×4=256 bytes = 4 CLs (6×16)
    _mm_prefetch((const char *)pfB,      _MM_HINT_T0);
    _mm_prefetch((const char *)(pfB+16), _MM_HINT_T0);
    _mm_prefetch((const char *)(pfB+32), _MM_HINT_T0);
    _mm_prefetch((const char *)(pfB+48), _MM_HINT_T0);
    pfB += 4 * NR;

    /* ... FMA compute ... */
}
```

**Hint:** `_MM_HINT_T0` (L1). Use `_MM_HINT_T1` (L2) if KC > 512 to avoid L1 pollution.

### 6.4 Caller: computing next_A and next_B

In `macro_kernel`, before calling the kernel function:
```c
int nr_m = (M_blk + MR - 1) / MR;   // number of m-strips in this panel
const float *next_A = (it + 1 < nr_m)
    ? pA_panel + (size_t)(it + 1) * K_blk * MR   // next strip
    : pA;                                           // last strip → harmless re-prefetch
const float *next_B = pB_strip;   // B doesn't change within macro_kernel

kfn(pA, pB_strip, Ctile, K_blk, alpha, beta_k, ldc, next_A, next_B, use_nt_store);
```

### 6.5 ASM kernel: prefetch before entering asm

The GCC inline asm operand limit is 30 (each `+x`/`+r` counts as 2). With 12 accumulator
registers + 3 pointer/counter registers, the budget is exactly full. To avoid exceeding it,
issue the prefetches in C before the asm block:

```c
int k4 = k_len / 4;     // main loop iterations
int kr = k_len & 3;     // remainder (0–3)

// Pre-issue first 16 CLs of next_A and next_B
size_t nA_bytes = (size_t)k_len * MR * sizeof(float);
size_t nB_bytes = (size_t)k_len * NR * sizeof(float);
for (int cl = 0; cl < 16; cl++) {
    size_t off = (size_t)cl * 64;
    if (off < nA_bytes) _mm_prefetch((const char *)next_A + off, _MM_HINT_T0);
    if (off < nB_bytes) _mm_prefetch((const char *)next_B + off, _MM_HINT_T0);
}

// ASM block with k4 as main loop counter; hardware prefetcher handles rest
__asm__ volatile (
    "test %[k4], %[k4]\n\t"
    "jz   3f\n\t"
    "1:\n\t"
    /* ... 4 k-steps FMA ... */
    "add $96,  %[pa]\n\t"    /* 4×MR×4 bytes for MR=6 */
    "add $256, %[pb]\n\t"    /* 4×NR×4 bytes for NR=16 */
    "dec %[k4]\n\t"
    "jnz 1b\n\t"
    "3:\n\t"
    : [a0lo] "+x"(acc0lo), ..., [pa] "+r"(pA), [pb] "+r"(pB), [k4] "+r"(k4)
    : : "ymm12", "ymm13", "ymm14", "memory"
);

// C tail loop for remainder
for (int k = 0; k < kr; k++) {
    __m256 blo = _mm256_load_ps(pB);
    __m256 bhi = _mm256_load_ps(pB + 8);
    // broadcast each A element and fmadd
    ...
    pA += MR; pB += NR;
}
```

---

## 7. Parallelism Strategy

### 7.1 TASK1: 2D Collapse (default, good for large matrices)

```c
#pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(nthd)
for (int jc = 0; jc < N; jc += NC)
    for (int ic = 0; ic < M; ic += MC)
        macro_kernel(ic, jc, ...);
```

Each iteration is one `MC × NC × KC` block. Dynamic scheduling balances load for
non-square matrices. Works well when `M*N / (MC*NC) >> nthd`.

### 7.2 TASK2: K-replication (for memory-bound small matrices)

When `M*N` is small, TASK1 has few tiles and threads starve. TASK2 splits K into `r` slices,
each thread computes a partial C, then reduces:

```c
// Phase 1: spawn tasks (one per ic×jc×pc combination)
#pragma omp single
{
    for each (ic, jc, pc):
        #pragma omp task
            partial_C[task_id] += A[ic,pc] * B[pc,jc]
    #pragma omp taskwait
}

// Phase 2: parallel reduction across K-slices
#pragma omp for collapse(2) schedule(static)
for (ic_tile ...) for (jc_tile ...):
    C[tile] = sum_over_r(partial_C[r][tile])
```

### 7.3 Small-matrix bypass

For `(long long)M*N*K < SMALL_MATRIX_OPS_THRESHOLD` (default: `128LL*128*128 = 2097152`),
skip OpenMP and run single-threaded TASK1 to avoid parallel overhead:

```c
if ((long long)M * N * K < SMALL_MATRIX_OPS_THRESHOLD) {
    // single-thread TASK1
    macro_kernel(0, 0, ...);
    return;
}
```

### 7.4 Adaptive MC

When `M / MC < nthd * 2`, there are fewer m-strips than 2× thread count → poor load balance.
Auto-shrink MC:

```c
int effective_MC = cfg->MC;
if (M >= MR * 2 && M / effective_MC < cfg->nb_threads * 2) {
    effective_MC = M / (cfg->nb_threads * 2);
    effective_MC = (effective_MC / MR) * MR;   // round down to MR multiple
    if (effective_MC < MR) effective_MC = MR;  // never below 1 strip
}
```

---

## 8. Complete Kernel Function Signature

All kernels (8×8, 6×16, 4×24, 6×16-asm) share this signature:

```c
void micro_kernel_NAME(
    const float * __restrict__ pA,      // packed A panel [k_len][MR]
    const float * __restrict__ pB,      // packed B panel [k_len][NR]
    float       * __restrict__ C,       // output, row-major, C[0..MR-1][0..NR-1]
    int                        k_len,   // inner dimension (depth of tile)
    float                      alpha,   // scalar α
    float                      beta,    // scalar β  (0 = first K-block, 1 = accumulate)
    int                        ldc,     // leading dimension of C (C stride between rows)
    const float * __restrict__ next_A,  // prefetch hint: start of next A panel
    const float * __restrict__ next_B,  // prefetch hint: start of next B panel
    int                        use_nt_store  // 1 = _mm256_stream_ps for C output
);
```

Function pointer type:
```c
typedef void (*kernel_fn_t)(const float *, const float *, float *,
                             int, float, float, int,
                             const float *, const float *, int);
```

---

## 9. GCC Inline ASM Operand Limit

GCC limits inline asm to **30 operand slots** total. Each `"+x"` or `"+r"` bidirectional
operand counts as **2** (one input slot + one output slot).

Budget for 6×16-asm kernel:
```
12 accumulators × "+x" = 24 slots
 3 registers    × "+r" = 6 slots   (pA, pB, k4)
Total: 30 slots  ← exactly at limit
```

**Consequences:**
- `next_A` and `next_B` cannot be asm operands → handled in C before asm.
- The remainder loop (k_len % 4) cannot use a 4th `"+r"` operand → handled in C after asm
  using `kr = k_len & 3` pre-computed before asm.
- Scratch registers (e.g. `%%eax` for internal use) must be listed in the clobber list.

---

## 10. Build System

### Compiler flags
```makefile
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra -fopenmp -mavx -mavx2 -mfma
```

### Kernel selection
The kernel to use is selected via `sgemm_config_t.kernel`:
```c
typedef enum { KERNEL_8X8, KERNEL_6X16, KERNEL_4X24, KERNEL_6X16_ASM } kernel_type_t;
```

`sgemm_ex()` maps this to a function pointer and calls through `macro_kernel`.

### Key files
```
include/sgemm.h                ← public API + sgemm_config_t definition
src/sgemm.c                    ← 5-loop framework, packing, macro_kernel, task1/task2
src/kernels/kernel_8x8.h       ← 8×8 AVX2+FMA micro-kernel (C intrinsics)
src/kernels/kernel_6x16.h      ← 6×16 AVX2+FMA micro-kernel (C intrinsics)
src/kernels/kernel_4x24.h      ← 4×24 AVX2+FMA micro-kernel (C intrinsics)
src/kernels/kernel_6x16_asm.h  ← 6×16 micro-kernel (GCC inline x86-64 asm)
bench/bench_sgemm.c            ← adaptive benchmark (--M --N --K --kernel flags)
tests/test_sgemm.c             ← 96-case correctness test suite
```

---

## 11. Validation Checklist

After any kernel modification, run:

```bash
make test        # builds test binary and executes it
./bin/test_sgemm # must print "ALL TESTS PASSED"
```

The test suite covers:
- Square matrices: 8×8 through 512×512
- Odd sizes: 13×17×19, 33×33×33, 97×67×53
- Non-square: 100×200×300
- Non-trivial scalars: α=2, β=0.5
- Zero-alpha: α=0, β=1 (C unchanged)
- Tall: 512×32×256
- Wide: 32×512×256
- Thin-K: 256×256×8
- All four kernel types (8×8, 6×16, 4×24, 6×16-asm)
- Both TASK1 and TASK2 parallelism modes

**Tolerance:** `|result - reference| / |reference| < 1e-4` (relative), or `< 1e-4` absolute
for near-zero values. Computed via OpenMP-parallelized reference GEMM.

---

## 12. Benchmarking Protocol

```bash
./bin/bench_sgemm --M 1024 --N 1024 --K 1024 --kernel 6x16asm
```

Output: JSON with median GFLOPS, stddev%, min/max times.  
**Target:** stddev < 5% (< 2% ideal). If higher, increase `--runs` or check thermal throttling.

**Performance reference points (single socket, ~3.5 GHz, 1 thread):**
- 8×8: ~60 GFLOPS at 1024³
- 6×16: ~65 GFLOPS at 1024³
- 4×24: ~62 GFLOPS at 1024³
- 6×16-asm: ~65 GFLOPS at 1024³ (matches C intrinsics; compiler is effective)

---

## 13. Step-by-Step Reconstruction Order

To build this implementation from scratch:

1. **Baseline:** scalar GEMM with triple loop
2. **AVX2 micro-kernel (no packing):** hard-code `pA` stride = `lda`, `pB` stride = `ldb`
3. **Packing:** implement `pack_A` and `pack_B`, verify correctness
4. **5-loop framework:** add L3/L2/L1 blocking loops around micro-kernel
5. **×2 unrolling:** manually unroll k-loop
6. **×4 unrolling:** extend to 4 steps; add ×2 peel + ×1 tail
7. **Software prefetch:** add `next_A`/`next_B` params; spread prefetches across k-loop
8. **NT stores:** add `use_nt_store` flag; `_mm256_stream_ps` + alignment check
9. **OpenMP TASK1:** `#pragma omp parallel for collapse(2) schedule(dynamic)`
10. **OpenMP TASK2:** K-replication with task-parallel reduction
11. **Small-matrix bypass:** threshold at M*N*K < 128³
12. **Adaptive MC:** shrink when M/MC < 2*nthd
13. **ASM kernel:** rewrite ×4 loop in GCC inline asm; keep pre/post in C (operand limit)

At each step: `make test` must pass, `make bench` should show monotone improvement.
