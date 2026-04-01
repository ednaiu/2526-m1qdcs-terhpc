/*
 * sgemm_kernel.h  —  8×8 AVX2+FMA SGEMM micro-kernel (inline)
 *
 * Computes:  C[MR x NR] = alpha * A[MR x k_len] * B[k_len x NR] + beta * C
 *   where A and B are pre-packed into contiguous panels.
 *
 * Register allocation (16 YMM registers on x86-64):
 *   ymm0  – ymm7  : 8 row accumulators  acc[i] = C[i, 0..7]
 *   ymm8           : B row vector  B[k, 0..7]
 *   ymm9           : broadcast of A[i, k] scalar
 *   ymm14, ymm15   : alpha, beta vectors for scaling
 *
 * Strategy:
 *   Row-accumulator layout: acc[i] holds one full row C[i, 0..NR-1].
 *   Since C is row-major, each acc maps directly to a contiguous
 *   memory region → enables direct _mm256_storeu_ps, no scatter.
 *
 *   Inner loop over k:
 *     load b_row = B[k, 0..7]   (1 YMM load, contiguous in packed_B)
 *     for i = 0..7:
 *       broadcast a_ik = A[i + k*MR]
 *       acc[i] += a_ik * b_row   (VFMADD231PS)
 *
 *   8 FMAs per k step = 128 flops per iteration.
 *
 * Peak throughput on Haswell:
 *   2 FMA units × 8 floats × 2 flops = 32 SP-FLOPS/cycle
 */

#ifndef SGEMM_KERNEL_H
#define SGEMM_KERNEL_H

#include <immintrin.h>   /* AVX2 + FMA intrinsics */
#include "sgemm.h"       /* MR, NR constants      */

/*
 * micro_kernel  —  8×8 AVX2+FMA, row-accumulator layout
 *
 *   packed_A : [k_len][MR]  — for each k, 8 A values contiguous
 *   packed_B : [k_len][NR]  — for each k, 8 B values contiguous
 *   C        : row-major, stride ldc
 *   k_len    : depth (≤ KC)
 *   alpha    : scalar for A*B product
 *   beta     : scalar for existing C
 *   ldc      : leading dimension of C
 */
static inline __attribute__((always_inline)) void micro_kernel(
        const float * __restrict__ packed_A,  /* [k_len × MR] */
        const float * __restrict__ packed_B,  /* [k_len × NR] */
        float       * __restrict__ C,
        int k_len, float alpha, float beta, int ldc)
{
    /* ------------------------------------------------------------------ *
     * 8 row accumulators: acc[i] holds C[i, 0..7]                        *
     * ------------------------------------------------------------------ */
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    /* ------------------------------------------------------------------ *
     * Main k loop                                                         *
     * ------------------------------------------------------------------ */
    const float *pA = packed_A;
    const float *pB = packed_B;

    int k = 0;
    for (; k <= k_len - 4; k += 4) {
        /* Prefetch ahead */
        _mm_prefetch((const char *)(pA + 16 * MR), _MM_HINT_T0);
        _mm_prefetch((const char *)(pB + 16 * NR), _MM_HINT_T0);

        /* --- k+0 --- */
        __m256 b0 = _mm256_load_ps(pB);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 0), b0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 1), b0, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2), b0, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3), b0, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 4), b0, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 5), b0, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 6), b0, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 7), b0, acc7);

        /* --- k+1 --- */
        __m256 b1 = _mm256_load_ps(pB + NR);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + MR + 0), b1, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + MR + 1), b1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + MR + 2), b1, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + MR + 3), b1, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + MR + 4), b1, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + MR + 5), b1, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + MR + 6), b1, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + MR + 7), b1, acc7);

        /* --- k+2 --- */
        __m256 b2 = _mm256_load_ps(pB + 2*NR);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2*MR + 0), b2, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2*MR + 1), b2, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2*MR + 2), b2, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2*MR + 3), b2, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2*MR + 4), b2, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2*MR + 5), b2, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2*MR + 6), b2, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2*MR + 7), b2, acc7);

        /* --- k+3 --- */
        __m256 b3 = _mm256_load_ps(pB + 3*NR);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3*MR + 0), b3, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3*MR + 1), b3, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3*MR + 2), b3, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3*MR + 3), b3, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3*MR + 4), b3, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3*MR + 5), b3, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3*MR + 6), b3, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3*MR + 7), b3, acc7);

        pA += 4 * MR;
        pB += 4 * NR;
    }

    /* --- tail (k_len % 4) --- */
    for (; k < k_len; k++) {
        __m256 bk = _mm256_load_ps(pB);
        acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 0), bk, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 1), bk, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 2), bk, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 3), bk, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 4), bk, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 5), bk, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 6), bk, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_broadcast_ss(pA + 7), bk, acc7);
        pA += MR;
        pB += NR;
    }

    /* ------------------------------------------------------------------ *
     * Scale by alpha/beta and store directly to C rows (no scatter!)      *
     * ------------------------------------------------------------------ */
    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbeta  = _mm256_set1_ps(beta);

    /* Multiply accumulators by alpha */
    acc0 = _mm256_mul_ps(valpha, acc0);
    acc1 = _mm256_mul_ps(valpha, acc1);
    acc2 = _mm256_mul_ps(valpha, acc2);
    acc3 = _mm256_mul_ps(valpha, acc3);
    acc4 = _mm256_mul_ps(valpha, acc4);
    acc5 = _mm256_mul_ps(valpha, acc5);
    acc6 = _mm256_mul_ps(valpha, acc6);
    acc7 = _mm256_mul_ps(valpha, acc7);

    /* Store: C[i,:] = alpha * acc[i] + beta * C[i,:] */
#define STORE_ROW(row_acc, row_idx) do {                              \
        float *Crow = C + (row_idx) * ldc;                            \
        __m256 existing = _mm256_loadu_ps(Crow);                      \
        __m256 result = _mm256_fmadd_ps(vbeta, existing, (row_acc));  \
        _mm256_storeu_ps(Crow, result);                               \
    } while(0)

    STORE_ROW(acc0, 0);
    STORE_ROW(acc1, 1);
    STORE_ROW(acc2, 2);
    STORE_ROW(acc3, 3);
    STORE_ROW(acc4, 4);
    STORE_ROW(acc5, 5);
    STORE_ROW(acc6, 6);
    STORE_ROW(acc7, 7);

#undef STORE_ROW
}

#endif /* SGEMM_KERNEL_H */
