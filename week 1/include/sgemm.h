/*
 * sgemm.h  —  Public header for the TER-HPC SGEMM library
 *
 * Implements:  C = alpha * A * B + beta * C
 *   A : M x K  (row-major)
 *   B : K x N  (row-major)
 *   C : M x N  (row-major)
 */

#ifndef SGEMM_H
#define SGEMM_H

#include <stddef.h>

/* --------------------------------------------------------------------------
 * Micro-kernel tile sizes — 6×16 for Haswell (16 YMM registers)
 *
 *   Register plan:
 *     12 accumulators: 6 rows × 2 YMM (low 8 + high 8 cols)
 *      2 B vectors:    B_lo[k,0..7], B_hi[k,8..15]
 *      1 A broadcast:  one A[i,k] scalar
 *      1 spare
 *     = 16 total YMM registers
 *
 *   This gives 12 FMAs per k-step (vs 8 for 8×8).
 * -------------------------------------------------------------------------- */
#define MR  6       /* rows processed by micro-kernel               */
#define NR  16      /* cols processed by micro-kernel (2 YMM widths)*/

/* --------------------------------------------------------------------------
 * Cache-blocking tile sizes  (tuned for Haswell Xeon E5-2670 v3)
 *   L1d = 32 KiB, L2 = 256 KiB, L3 = 25 MiB (shared)
 *
 *   MC × KC panel of A → fits in L2:  120 × 512 × 4B = 240 KiB ≤ 256 KiB
 *   KC × NC panel of B → fits in L3:  512 × 4096 × 4B = 8 MiB ≤ 25 MiB
 *   MR × KC micro-panel of A → fits in L1: 6 × 512 × 4B = 12 KiB ≤ 32 KiB
 * -------------------------------------------------------------------------- */
#define MC  120     /* rows of A packed into L2 (multiple of MR=6)  */
#define KC  512     /* depth dimension packed into L2 slab           */
#define NC  4096    /* cols of B packed into L3 slab                 */

/* --------------------------------------------------------------------------
 * Public SGEMM interface
 * -------------------------------------------------------------------------- */

/*
 * sgemm_ref  —  naive triple-loop reference (for correctness tests only)
 */
void sgemm_ref(int M, int N, int K,
               float alpha,
               const float *A, int lda,
               const float *B, int ldb,
               float beta,
               float       *C, int ldc);

/*
 * sgemm  —  optimised AVX2+FMA blocked SGEMM
 */
void sgemm(int M, int N, int K,
           float alpha,
           const float *A, int lda,
           const float *B, int ldb,
           float beta,
           float       *C, int ldc);

#endif /* SGEMM_H */
