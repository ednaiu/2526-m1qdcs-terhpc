/*
 * kernel_8x8.h  —  8×8 AVX2+FMA SGEMM micro-kernel
 *
 * Computes:  C[8×8] += alpha * A[8×k] * B[k×8]  (packed inputs)
 *
 * Register plan (AVX2, 16 YMM of 8 floats each):
 *   ymm0–ymm7   : 8 accumulators, one per output row (1 YMM = 8 cols)
 *   ymm8        : B row  B[k, 0..7]
 *   ymm9        : A[i,k] broadcast (scalar → 8 lanes)
 *   ymm10–ymm15 : free (used for alpha/beta scale)
 *
 *   8 FMAs per k-step = 128 FLOP/step
 *   Arithmetic intensity: 2×8×8 / (8+8) = 8.0 FLOP/byte
 *
 * Packing layout expected:
 *   packed_A[k, i]  →  pA[k * 8 + i]   (MR=8 elements per k-step)
 *   packed_B[k, j]  →  pB[k * 8 + j]   (NR=8 elements per k-step)
 */

#ifndef KERNEL_8X8_H
#define KERNEL_8X8_H

#include <immintrin.h>

#define MR_8x8  8
#define NR_8x8  8

static inline __attribute__((always_inline))
void micro_kernel_8x8(const float * __restrict__ pA,
                      const float * __restrict__ pB,
                      float       * __restrict__ C,
                      int k_len, float alpha, float beta, int ldc)
{
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    /* Main loop, unrolled ×2 */
    int k = 0;
    for (; k <= k_len - 2; k += 2) {
        _mm_prefetch((const char *)(pA + 8 * MR_8x8), _MM_HINT_T0);
        _mm_prefetch((const char *)(pB + 8 * NR_8x8), _MM_HINT_T0);

        /* k+0 */
        __m256 b0 = _mm256_load_ps(pB);
        __m256 a;
        a = _mm256_broadcast_ss(pA + 0); c0 = _mm256_fmadd_ps(a, b0, c0);
        a = _mm256_broadcast_ss(pA + 1); c1 = _mm256_fmadd_ps(a, b0, c1);
        a = _mm256_broadcast_ss(pA + 2); c2 = _mm256_fmadd_ps(a, b0, c2);
        a = _mm256_broadcast_ss(pA + 3); c3 = _mm256_fmadd_ps(a, b0, c3);
        a = _mm256_broadcast_ss(pA + 4); c4 = _mm256_fmadd_ps(a, b0, c4);
        a = _mm256_broadcast_ss(pA + 5); c5 = _mm256_fmadd_ps(a, b0, c5);
        a = _mm256_broadcast_ss(pA + 6); c6 = _mm256_fmadd_ps(a, b0, c6);
        a = _mm256_broadcast_ss(pA + 7); c7 = _mm256_fmadd_ps(a, b0, c7);

        /* k+1 */
        __m256 b1 = _mm256_load_ps(pB + NR_8x8);
        a = _mm256_broadcast_ss(pA + MR_8x8 + 0); c0 = _mm256_fmadd_ps(a, b1, c0);
        a = _mm256_broadcast_ss(pA + MR_8x8 + 1); c1 = _mm256_fmadd_ps(a, b1, c1);
        a = _mm256_broadcast_ss(pA + MR_8x8 + 2); c2 = _mm256_fmadd_ps(a, b1, c2);
        a = _mm256_broadcast_ss(pA + MR_8x8 + 3); c3 = _mm256_fmadd_ps(a, b1, c3);
        a = _mm256_broadcast_ss(pA + MR_8x8 + 4); c4 = _mm256_fmadd_ps(a, b1, c4);
        a = _mm256_broadcast_ss(pA + MR_8x8 + 5); c5 = _mm256_fmadd_ps(a, b1, c5);
        a = _mm256_broadcast_ss(pA + MR_8x8 + 6); c6 = _mm256_fmadd_ps(a, b1, c6);
        a = _mm256_broadcast_ss(pA + MR_8x8 + 7); c7 = _mm256_fmadd_ps(a, b1, c7);

        pA += 2 * MR_8x8;
        pB += 2 * NR_8x8;
    }

    /* Tail */
    for (; k < k_len; k++) {
        __m256 b = _mm256_load_ps(pB);
        __m256 a;
        a = _mm256_broadcast_ss(pA + 0); c0 = _mm256_fmadd_ps(a, b, c0);
        a = _mm256_broadcast_ss(pA + 1); c1 = _mm256_fmadd_ps(a, b, c1);
        a = _mm256_broadcast_ss(pA + 2); c2 = _mm256_fmadd_ps(a, b, c2);
        a = _mm256_broadcast_ss(pA + 3); c3 = _mm256_fmadd_ps(a, b, c3);
        a = _mm256_broadcast_ss(pA + 4); c4 = _mm256_fmadd_ps(a, b, c4);
        a = _mm256_broadcast_ss(pA + 5); c5 = _mm256_fmadd_ps(a, b, c5);
        a = _mm256_broadcast_ss(pA + 6); c6 = _mm256_fmadd_ps(a, b, c6);
        a = _mm256_broadcast_ss(pA + 7); c7 = _mm256_fmadd_ps(a, b, c7);
        pA += MR_8x8;
        pB += NR_8x8;
    }

    /* Store: C[i,:] = alpha * acc[i] + beta * C[i,:] */
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vb = _mm256_set1_ps(beta);

#define STORE_ROW8(ci, row) do {                                              \
    float *Cr = C + (row) * ldc;                                              \
    __m256 existing = _mm256_loadu_ps(Cr);                                    \
    _mm256_storeu_ps(Cr, _mm256_fmadd_ps(vb, existing,                       \
                         _mm256_mul_ps(va, ci)));                              \
} while (0)

    STORE_ROW8(c0, 0); STORE_ROW8(c1, 1);
    STORE_ROW8(c2, 2); STORE_ROW8(c3, 3);
    STORE_ROW8(c4, 4); STORE_ROW8(c5, 5);
    STORE_ROW8(c6, 6); STORE_ROW8(c7, 7);

#undef STORE_ROW8
}

#endif /* KERNEL_8X8_H */
