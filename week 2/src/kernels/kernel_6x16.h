/*
 * kernel_6x16.h  —  6×16 AVX2+FMA SGEMM micro-kernel
 *
 * Computes:  C[6×16] += alpha * A[6×k] * B[k×16]  (packed inputs)
 *
 * Register plan (AVX2, 16 YMM of 8 floats each):
 *   ymm0 –ymm5   : accumulators, row 0-5, low  8 cols  (C[i, 0..7])
 *   ymm6 –ymm11  : accumulators, row 0-5, high 8 cols  (C[i, 8..15])
 *   ymm12        : B[k, 0..7]    (low half of B row)
 *   ymm13        : B[k, 8..15]   (high half of B row)
 *   ymm14        : A[i,k] broadcast
 *   ymm15        : scratch / alpha / beta
 *
 *   12 FMAs per k-step = 192 FLOP/step
 *   Arithmetic intensity: 2×6×16 / (6+16) ≈ 8.73 FLOP/byte
 *
 * Packing layout expected:
 *   packed_A[k, i]  →  pA[k * 6 + i]   (MR=6 elements per k-step)
 *   packed_B[k, j]  →  pB[k * 16 + j]  (NR=16 elements per k-step)
 */

#ifndef KERNEL_6X16_H
#define KERNEL_6X16_H

#include <immintrin.h>

#define MR_6x16  6
#define NR_6x16  16

static inline __attribute__((always_inline))
void micro_kernel_6x16(const float * __restrict__ pA,
                       const float * __restrict__ pB,
                       float       * __restrict__ C,
                       int k_len, float alpha, float beta, int ldc)
{
    /* 6 rows × 2 halves = 12 accumulators */
    __m256 acc0lo = _mm256_setzero_ps(), acc0hi = _mm256_setzero_ps();
    __m256 acc1lo = _mm256_setzero_ps(), acc1hi = _mm256_setzero_ps();
    __m256 acc2lo = _mm256_setzero_ps(), acc2hi = _mm256_setzero_ps();
    __m256 acc3lo = _mm256_setzero_ps(), acc3hi = _mm256_setzero_ps();
    __m256 acc4lo = _mm256_setzero_ps(), acc4hi = _mm256_setzero_ps();
    __m256 acc5lo = _mm256_setzero_ps(), acc5hi = _mm256_setzero_ps();

    /* Main loop, unrolled ×2 */
    int k = 0;
    for (; k <= k_len - 2; k += 2) {
        _mm_prefetch((const char *)(pA + 8 * MR_6x16), _MM_HINT_T0);
        _mm_prefetch((const char *)(pB + 8 * NR_6x16), _MM_HINT_T0);

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
        __m256 blo1 = _mm256_load_ps(pB + NR_6x16);
        __m256 bhi1 = _mm256_load_ps(pB + NR_6x16 + 8);
        a = _mm256_broadcast_ss(pA + MR_6x16 + 0);
        acc0lo = _mm256_fmadd_ps(a, blo1, acc0lo);
        acc0hi = _mm256_fmadd_ps(a, bhi1, acc0hi);
        a = _mm256_broadcast_ss(pA + MR_6x16 + 1);
        acc1lo = _mm256_fmadd_ps(a, blo1, acc1lo);
        acc1hi = _mm256_fmadd_ps(a, bhi1, acc1hi);
        a = _mm256_broadcast_ss(pA + MR_6x16 + 2);
        acc2lo = _mm256_fmadd_ps(a, blo1, acc2lo);
        acc2hi = _mm256_fmadd_ps(a, bhi1, acc2hi);
        a = _mm256_broadcast_ss(pA + MR_6x16 + 3);
        acc3lo = _mm256_fmadd_ps(a, blo1, acc3lo);
        acc3hi = _mm256_fmadd_ps(a, bhi1, acc3hi);
        a = _mm256_broadcast_ss(pA + MR_6x16 + 4);
        acc4lo = _mm256_fmadd_ps(a, blo1, acc4lo);
        acc4hi = _mm256_fmadd_ps(a, bhi1, acc4hi);
        a = _mm256_broadcast_ss(pA + MR_6x16 + 5);
        acc5lo = _mm256_fmadd_ps(a, blo1, acc5lo);
        acc5hi = _mm256_fmadd_ps(a, bhi1, acc5hi);

        pA += 2 * MR_6x16;
        pB += 2 * NR_6x16;
    }

    /* Tail */
    for (; k < k_len; k++) {
        __m256 blo = _mm256_load_ps(pB);
        __m256 bhi = _mm256_load_ps(pB + 8);
        __m256 a;
        a = _mm256_broadcast_ss(pA + 0);
        acc0lo = _mm256_fmadd_ps(a, blo, acc0lo); acc0hi = _mm256_fmadd_ps(a, bhi, acc0hi);
        a = _mm256_broadcast_ss(pA + 1);
        acc1lo = _mm256_fmadd_ps(a, blo, acc1lo); acc1hi = _mm256_fmadd_ps(a, bhi, acc1hi);
        a = _mm256_broadcast_ss(pA + 2);
        acc2lo = _mm256_fmadd_ps(a, blo, acc2lo); acc2hi = _mm256_fmadd_ps(a, bhi, acc2hi);
        a = _mm256_broadcast_ss(pA + 3);
        acc3lo = _mm256_fmadd_ps(a, blo, acc3lo); acc3hi = _mm256_fmadd_ps(a, bhi, acc3hi);
        a = _mm256_broadcast_ss(pA + 4);
        acc4lo = _mm256_fmadd_ps(a, blo, acc4lo); acc4hi = _mm256_fmadd_ps(a, bhi, acc4hi);
        a = _mm256_broadcast_ss(pA + 5);
        acc5lo = _mm256_fmadd_ps(a, blo, acc5lo); acc5hi = _mm256_fmadd_ps(a, bhi, acc5hi);
        pA += MR_6x16;
        pB += NR_6x16;
    }

    /* Store: C[i,:] = alpha * acc[i] + beta * C[i,:] */
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vb = _mm256_set1_ps(beta);

    /* Scale by alpha */
    acc0lo = _mm256_mul_ps(va, acc0lo); acc0hi = _mm256_mul_ps(va, acc0hi);
    acc1lo = _mm256_mul_ps(va, acc1lo); acc1hi = _mm256_mul_ps(va, acc1hi);
    acc2lo = _mm256_mul_ps(va, acc2lo); acc2hi = _mm256_mul_ps(va, acc2hi);
    acc3lo = _mm256_mul_ps(va, acc3lo); acc3hi = _mm256_mul_ps(va, acc3hi);
    acc4lo = _mm256_mul_ps(va, acc4lo); acc4hi = _mm256_mul_ps(va, acc4hi);
    acc5lo = _mm256_mul_ps(va, acc5lo); acc5hi = _mm256_mul_ps(va, acc5hi);

#define STORE_ROW16(rlo, rhi, row) do {                                       \
    float *Cr = C + (row) * ldc;                                              \
    _mm256_storeu_ps(Cr,     _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr),     rlo)); \
    _mm256_storeu_ps(Cr + 8, _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr + 8), rhi)); \
} while (0)

    STORE_ROW16(acc0lo, acc0hi, 0);
    STORE_ROW16(acc1lo, acc1hi, 1);
    STORE_ROW16(acc2lo, acc2hi, 2);
    STORE_ROW16(acc3lo, acc3hi, 3);
    STORE_ROW16(acc4lo, acc4hi, 4);
    STORE_ROW16(acc5lo, acc5hi, 5);

#undef STORE_ROW16
}

#endif /* KERNEL_6X16_H */
