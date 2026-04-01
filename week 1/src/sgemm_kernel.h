/*
 * sgemm_kernel.h  —  6×16 AVX2+FMA SGEMM micro-kernel (inline)
 *
 * Computes:  C[MR x NR] = alpha * A[MR x k_len] * B[k_len x NR] + beta * C
 *   where A and B are pre-packed.
 *
 * Register allocation (16 YMM registers):
 *   ymm0 – ymm5   : accumulators for rows 0-5, low  8 cols (C[i, 0..7])
 *   ymm6 – ymm11  : accumulators for rows 0-5, high 8 cols (C[i, 8..15])
 *   ymm12          : B[k, 0..7]   (low half of B row)
 *   ymm13          : B[k, 8..15]  (high half of B row)
 *   ymm14          : broadcast of A[i, k]
 *   ymm15          : scratch / alpha / beta
 *
 * Strategy:
 *   For each k step:
 *     Load B_lo = B[k, 0..7], B_hi = B[k, 8..15]
 *     For each row i = 0..5:
 *       broadcast a_ik = A[i + k*MR]
 *       acc_lo[i] += a_ik * B_lo   (FMA)
 *       acc_hi[i] += a_ik * B_hi   (FMA)
 *
 *   12 FMAs per k step → 192 flops per step.
 *
 * Store:
 *   Each accumulator row maps to 2 contiguous YMM stores per C row.
 *   For row-major C: C[i, 0..7] and C[i, 8..15] are contiguous → no scatter.
 */

#ifndef SGEMM_KERNEL_H
#define SGEMM_KERNEL_H

#include <immintrin.h>   /* AVX2 + FMA intrinsics */
#include "sgemm.h"       /* MR=6, NR=16 constants */

static inline __attribute__((always_inline)) void micro_kernel(
        const float * __restrict__ packed_A,  /* [k_len × MR] */
        const float * __restrict__ packed_B,  /* [k_len × NR] */
        float       * __restrict__ C,
        int k_len, float alpha, float beta, int ldc)
{
    /* 6 rows × 2 halves = 12 accumulators */
    __m256 acc0lo = _mm256_setzero_ps();   __m256 acc0hi = _mm256_setzero_ps();
    __m256 acc1lo = _mm256_setzero_ps();   __m256 acc1hi = _mm256_setzero_ps();
    __m256 acc2lo = _mm256_setzero_ps();   __m256 acc2hi = _mm256_setzero_ps();
    __m256 acc3lo = _mm256_setzero_ps();   __m256 acc3hi = _mm256_setzero_ps();
    __m256 acc4lo = _mm256_setzero_ps();   __m256 acc4hi = _mm256_setzero_ps();
    __m256 acc5lo = _mm256_setzero_ps();   __m256 acc5hi = _mm256_setzero_ps();

    const float *pA = packed_A;
    const float *pB = packed_B;

    /* Main loop, unrolled ×2 */
    int k = 0;
    for (; k <= k_len - 2; k += 2) {
        /* Prefetch */
        _mm_prefetch((const char *)(pA + 8 * MR), _MM_HINT_T0);
        _mm_prefetch((const char *)(pB + 8 * NR), _MM_HINT_T0);

        /* --- k+0 --- */
        __m256 blo0 = _mm256_load_ps(pB);
        __m256 bhi0 = _mm256_load_ps(pB + 8);

        __m256 a;
        a = _mm256_broadcast_ss(pA + 0);
        acc0lo = _mm256_fmadd_ps(a, blo0, acc0lo);
        acc0hi = _mm256_fmadd_ps(a, bhi0, acc0hi);
        a = _mm256_broadcast_ss(pA + 1);
        acc1lo = _mm256_fmadd_ps(a, blo0, acc1lo);
        acc1hi = _mm256_fmadd_ps(a, bhi0, acc1hi);
        a = _mm256_broadcast_ss(pA + 2);
        acc2lo = _mm256_fmadd_ps(a, blo0, acc2lo);
        acc2hi = _mm256_fmadd_ps(a, bhi0, acc2hi);
        a = _mm256_broadcast_ss(pA + 3);
        acc3lo = _mm256_fmadd_ps(a, blo0, acc3lo);
        acc3hi = _mm256_fmadd_ps(a, bhi0, acc3hi);
        a = _mm256_broadcast_ss(pA + 4);
        acc4lo = _mm256_fmadd_ps(a, blo0, acc4lo);
        acc4hi = _mm256_fmadd_ps(a, bhi0, acc4hi);
        a = _mm256_broadcast_ss(pA + 5);
        acc5lo = _mm256_fmadd_ps(a, blo0, acc5lo);
        acc5hi = _mm256_fmadd_ps(a, bhi0, acc5hi);

        /* --- k+1 --- */
        __m256 blo1 = _mm256_load_ps(pB + NR);
        __m256 bhi1 = _mm256_load_ps(pB + NR + 8);

        a = _mm256_broadcast_ss(pA + MR + 0);
        acc0lo = _mm256_fmadd_ps(a, blo1, acc0lo);
        acc0hi = _mm256_fmadd_ps(a, bhi1, acc0hi);
        a = _mm256_broadcast_ss(pA + MR + 1);
        acc1lo = _mm256_fmadd_ps(a, blo1, acc1lo);
        acc1hi = _mm256_fmadd_ps(a, bhi1, acc1hi);
        a = _mm256_broadcast_ss(pA + MR + 2);
        acc2lo = _mm256_fmadd_ps(a, blo1, acc2lo);
        acc2hi = _mm256_fmadd_ps(a, bhi1, acc2hi);
        a = _mm256_broadcast_ss(pA + MR + 3);
        acc3lo = _mm256_fmadd_ps(a, blo1, acc3lo);
        acc3hi = _mm256_fmadd_ps(a, bhi1, acc3hi);
        a = _mm256_broadcast_ss(pA + MR + 4);
        acc4lo = _mm256_fmadd_ps(a, blo1, acc4lo);
        acc4hi = _mm256_fmadd_ps(a, bhi1, acc4hi);
        a = _mm256_broadcast_ss(pA + MR + 5);
        acc5lo = _mm256_fmadd_ps(a, blo1, acc5lo);
        acc5hi = _mm256_fmadd_ps(a, bhi1, acc5hi);

        pA += 2 * MR;
        pB += 2 * NR;
    }

    /* Tail */
    for (; k < k_len; k++) {
        __m256 blo = _mm256_load_ps(pB);
        __m256 bhi = _mm256_load_ps(pB + 8);
        __m256 a;

        a = _mm256_broadcast_ss(pA + 0);
        acc0lo = _mm256_fmadd_ps(a, blo, acc0lo);  acc0hi = _mm256_fmadd_ps(a, bhi, acc0hi);
        a = _mm256_broadcast_ss(pA + 1);
        acc1lo = _mm256_fmadd_ps(a, blo, acc1lo);  acc1hi = _mm256_fmadd_ps(a, bhi, acc1hi);
        a = _mm256_broadcast_ss(pA + 2);
        acc2lo = _mm256_fmadd_ps(a, blo, acc2lo);  acc2hi = _mm256_fmadd_ps(a, bhi, acc2hi);
        a = _mm256_broadcast_ss(pA + 3);
        acc3lo = _mm256_fmadd_ps(a, blo, acc3lo);  acc3hi = _mm256_fmadd_ps(a, bhi, acc3hi);
        a = _mm256_broadcast_ss(pA + 4);
        acc4lo = _mm256_fmadd_ps(a, blo, acc4lo);  acc4hi = _mm256_fmadd_ps(a, bhi, acc4hi);
        a = _mm256_broadcast_ss(pA + 5);
        acc5lo = _mm256_fmadd_ps(a, blo, acc5lo);  acc5hi = _mm256_fmadd_ps(a, bhi, acc5hi);

        pA += MR;
        pB += NR;
    }

    /* Scale by alpha and store with beta */
    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbeta  = _mm256_set1_ps(beta);

    /* Scale accumulators by alpha */
    acc0lo = _mm256_mul_ps(valpha, acc0lo);  acc0hi = _mm256_mul_ps(valpha, acc0hi);
    acc1lo = _mm256_mul_ps(valpha, acc1lo);  acc1hi = _mm256_mul_ps(valpha, acc1hi);
    acc2lo = _mm256_mul_ps(valpha, acc2lo);  acc2hi = _mm256_mul_ps(valpha, acc2hi);
    acc3lo = _mm256_mul_ps(valpha, acc3lo);  acc3hi = _mm256_mul_ps(valpha, acc3hi);
    acc4lo = _mm256_mul_ps(valpha, acc4lo);  acc4hi = _mm256_mul_ps(valpha, acc4hi);
    acc5lo = _mm256_mul_ps(valpha, acc5lo);  acc5hi = _mm256_mul_ps(valpha, acc5hi);

    /* C[i,:] = alpha*acc[i] + beta*C[i,:] — 2 stores per row, direct to memory */
#define STORE_ROW(row_lo, row_hi, row_idx) do {                              \
        float *Crow = C + (row_idx) * ldc;                                   \
        __m256 elo = _mm256_loadu_ps(Crow);                                  \
        __m256 ehi = _mm256_loadu_ps(Crow + 8);                              \
        _mm256_storeu_ps(Crow,     _mm256_fmadd_ps(vbeta, elo, (row_lo)));   \
        _mm256_storeu_ps(Crow + 8, _mm256_fmadd_ps(vbeta, ehi, (row_hi)));  \
    } while(0)

    STORE_ROW(acc0lo, acc0hi, 0);
    STORE_ROW(acc1lo, acc1hi, 1);
    STORE_ROW(acc2lo, acc2hi, 2);
    STORE_ROW(acc3lo, acc3hi, 3);
    STORE_ROW(acc4lo, acc4hi, 4);
    STORE_ROW(acc5lo, acc5hi, 5);

#undef STORE_ROW
}

#endif /* SGEMM_KERNEL_H */
