/*
 * kernel_4x24.h  —  4×24 AVX2+FMA SGEMM micro-kernel
 *
 * Computes:  C[4×24] += alpha * A[4×k] * B[k×24]  (packed inputs)
 *
 * Register plan (AVX2, 16 YMM of 8 floats each):
 *   ymm0 –ymm3   : accumulators, row 0-3, cols  0..7   (C[i,  0..7 ])
 *   ymm4 –ymm7   : accumulators, row 0-3, cols  8..15  (C[i,  8..15])
 *   ymm8 –ymm11  : accumulators, row 0-3, cols 16..23  (C[i, 16..23])
 *   ymm12        : B[k,  0..7]
 *   ymm13        : B[k,  8..15]
 *   ymm14        : B[k, 16..23]
 *   ymm15        : A[i,k] broadcast
 *
 *   12 FMAs per k-step = 192 FLOP/step  (same as 6×16!)
 *   Better register pressure balance for wide N workloads.
 *   Arithmetic intensity: 2×4×24 / (4+24) ≈ 6.86 FLOP/byte
 *
 * Packing layout expected:
 *   packed_A[k, i]  →  pA[k * 4 + i]   (MR=4 elements per k-step)
 *   packed_B[k, j]  →  pB[k * 24 + j]  (NR=24 elements per k-step)
 */

#ifndef KERNEL_4X24_H
#define KERNEL_4X24_H

#include <immintrin.h>

#define MR_4x24  4
#define NR_4x24  24

static inline __attribute__((always_inline))
void micro_kernel_4x24(const float * __restrict__ pA,
                       const float * __restrict__ pB,
                       float       * __restrict__ C,
                       int k_len, float alpha, float beta, int ldc)
{
    /* 4 rows × 3 thirds = 12 accumulators */
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();

    /* Main loop, unrolled ×2 */
    int k = 0;
    for (; k <= k_len - 2; k += 2) {
        _mm_prefetch((const char *)(pA + 8 * MR_4x24), _MM_HINT_T0);
        _mm_prefetch((const char *)(pB + 8 * NR_4x24), _MM_HINT_T0);

        /* --- k+0 --- */
        __m256 b00 = _mm256_load_ps(pB);
        __m256 b01 = _mm256_load_ps(pB + 8);
        __m256 b02 = _mm256_load_ps(pB + 16);
        __m256 a;
        a = _mm256_broadcast_ss(pA + 0);
        c00 = _mm256_fmadd_ps(a, b00, c00);
        c01 = _mm256_fmadd_ps(a, b01, c01);
        c02 = _mm256_fmadd_ps(a, b02, c02);
        a = _mm256_broadcast_ss(pA + 1);
        c10 = _mm256_fmadd_ps(a, b00, c10);
        c11 = _mm256_fmadd_ps(a, b01, c11);
        c12 = _mm256_fmadd_ps(a, b02, c12);
        a = _mm256_broadcast_ss(pA + 2);
        c20 = _mm256_fmadd_ps(a, b00, c20);
        c21 = _mm256_fmadd_ps(a, b01, c21);
        c22 = _mm256_fmadd_ps(a, b02, c22);
        a = _mm256_broadcast_ss(pA + 3);
        c30 = _mm256_fmadd_ps(a, b00, c30);
        c31 = _mm256_fmadd_ps(a, b01, c31);
        c32 = _mm256_fmadd_ps(a, b02, c32);

        /* --- k+1 --- */
        __m256 b10 = _mm256_load_ps(pB + NR_4x24);
        __m256 b11 = _mm256_load_ps(pB + NR_4x24 + 8);
        __m256 b12 = _mm256_load_ps(pB + NR_4x24 + 16);
        a = _mm256_broadcast_ss(pA + MR_4x24 + 0);
        c00 = _mm256_fmadd_ps(a, b10, c00);
        c01 = _mm256_fmadd_ps(a, b11, c01);
        c02 = _mm256_fmadd_ps(a, b12, c02);
        a = _mm256_broadcast_ss(pA + MR_4x24 + 1);
        c10 = _mm256_fmadd_ps(a, b10, c10);
        c11 = _mm256_fmadd_ps(a, b11, c11);
        c12 = _mm256_fmadd_ps(a, b12, c12);
        a = _mm256_broadcast_ss(pA + MR_4x24 + 2);
        c20 = _mm256_fmadd_ps(a, b10, c20);
        c21 = _mm256_fmadd_ps(a, b11, c21);
        c22 = _mm256_fmadd_ps(a, b12, c22);
        a = _mm256_broadcast_ss(pA + MR_4x24 + 3);
        c30 = _mm256_fmadd_ps(a, b10, c30);
        c31 = _mm256_fmadd_ps(a, b11, c31);
        c32 = _mm256_fmadd_ps(a, b12, c32);

        pA += 2 * MR_4x24;
        pB += 2 * NR_4x24;
    }

    /* Tail */
    for (; k < k_len; k++) {
        __m256 b0 = _mm256_load_ps(pB);
        __m256 b1 = _mm256_load_ps(pB + 8);
        __m256 b2 = _mm256_load_ps(pB + 16);
        __m256 a;
        a = _mm256_broadcast_ss(pA + 0);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); c02 = _mm256_fmadd_ps(a, b2, c02);
        a = _mm256_broadcast_ss(pA + 1);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); c12 = _mm256_fmadd_ps(a, b2, c12);
        a = _mm256_broadcast_ss(pA + 2);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); c22 = _mm256_fmadd_ps(a, b2, c22);
        a = _mm256_broadcast_ss(pA + 3);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); c32 = _mm256_fmadd_ps(a, b2, c32);
        pA += MR_4x24;
        pB += NR_4x24;
    }

    /* Store: C[i,:] = alpha * acc[i] + beta * C[i,:] */
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vb = _mm256_set1_ps(beta);

#define STORE_ROW24(r0, r1, r2, row) do {                                     \
    float *Cr = C + (row) * ldc;                                              \
    _mm256_storeu_ps(Cr,      _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr),       \
                              _mm256_mul_ps(va, r0)));                        \
    _mm256_storeu_ps(Cr + 8,  _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr + 8),  \
                              _mm256_mul_ps(va, r1)));                        \
    _mm256_storeu_ps(Cr + 16, _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr + 16), \
                              _mm256_mul_ps(va, r2)));                        \
} while (0)

    STORE_ROW24(c00, c01, c02, 0);
    STORE_ROW24(c10, c11, c12, 1);
    STORE_ROW24(c20, c21, c22, 2);
    STORE_ROW24(c30, c31, c32, 3);

#undef STORE_ROW24
}

#endif /* KERNEL_4X24_H */
