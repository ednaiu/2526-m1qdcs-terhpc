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
 * Micro-kernel tile sizes
 *   MR x NR  must fit in 16 YMM registers:
 *     NR accumulators  +  1 A-vector  +  1 B-broadcast  = NR + 2
 *   With MR=8 (1 YMM holds 8 floats) and NR=8 we need 10 regs → OK.
 * -------------------------------------------------------------------------- */
#define MR  8   /* rows processed by micro-kernel  (= 1 YMM width) */
#define NR  8   /* cols processed by micro-kernel                   */

/* --------------------------------------------------------------------------
 * Cache-blocking tile sizes  (tune after inspecting lscpu on target machine)
 *   KC x MC  panel of A stays in L1 during micro-kernel sweep
 *   KC x NC  panel of B stays in L2
 *   MC, NC govern L3 reuse
 * -------------------------------------------------------------------------- */
#define MC  512     /* rows of A packed into L1/L2 slab   */
#define KC  256     /* depth dimension packed into L2 slab */
#define NC  4096    /* cols of B packed into L3 slab       */

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
