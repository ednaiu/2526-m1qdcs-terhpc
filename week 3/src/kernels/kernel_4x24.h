/*
 * kernel_4x24.h  —  4×24 AVX2+FMA SGEMM micro-kernel  (Week 3)
 *
 * Computes:  C[4×24] += alpha * A[4×k] * B[k×24]  (packed inputs)
 *
 * Week 3 changes vs Week 2:
 *   - Main loop unrolled ×4 (was ×2)
 *   - Next-tile software prefetch: 1 CL of A (4×4×4=64B) and
 *     6 CLs of B (4×24×4=384B) per ×4 iteration
 *   - NT store support via use_nt_store flag
 *
 * Register plan (AVX2, 16 YMM of 8 floats each):
 *   ymm0 –ymm3   : accumulators, row 0-3, cols  0..7
 *   ymm4 –ymm7   : accumulators, row 0-3, cols  8..15
 *   ymm8 –ymm11  : accumulators, row 0-3, cols 16..23
 *   ymm12–ymm14  : B thirds (loaded each k-step)
 *   ymm15        : A broadcast
 *
 *   12 FMAs per k-step = 192 FLOP/step
 */

#ifndef KERNEL_4X24_H
#define KERNEL_4X24_H

#include <immintrin.h>
#include <stdint.h>

#define MR_4x24  4
#define NR_4x24  24

static inline __attribute__((always_inline))
void micro_kernel_4x24(const float * __restrict__ pA,
                       const float * __restrict__ pB,
                       float       * __restrict__ C,
                       int k_len, float alpha, float beta, int ldc,
                       const float * __restrict__ next_A,
                       const float * __restrict__ next_B,
                       int use_nt_store)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();

    /* Next-tile prefetch pointers.
     * Per ×4 iteration: A advances 4×4×4=64 bytes (1 CL),
     *                   B advances 4×24×4=384 bytes (6 CLs). */
    const float *pfA = next_A;
    const float *pfB = next_B;

    int k = 0;

    /* ── Main loop: ×4 unrolled ──────────────────────────────── */
    for (; k <= k_len - 4; k += 4) {

        /* Prefetch next tile: 1 CL of A, 6 CLs of B */
        _mm_prefetch((const char *)(pfA),       _MM_HINT_T0);
        pfA += 4 * MR_4x24;   /* 64 bytes */

        _mm_prefetch((const char *)(pfB),       _MM_HINT_T0);
        _mm_prefetch((const char *)(pfB + 16),  _MM_HINT_T0);
        _mm_prefetch((const char *)(pfB + 32),  _MM_HINT_T0);
        _mm_prefetch((const char *)(pfB + 48),  _MM_HINT_T0);
        _mm_prefetch((const char *)(pfB + 64),  _MM_HINT_T0);
        _mm_prefetch((const char *)(pfB + 80),  _MM_HINT_T0);
        pfB += 4 * NR_4x24;   /* 384 bytes */

        /* --- k+0 --- */
        {
            __m256 b0 = _mm256_load_ps(pB),      b1 = _mm256_load_ps(pB + 8),
                   b2 = _mm256_load_ps(pB + 16);
            __m256 a;
            a = _mm256_broadcast_ss(pA + 0);
            c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); c02 = _mm256_fmadd_ps(a, b2, c02);
            a = _mm256_broadcast_ss(pA + 1);
            c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); c12 = _mm256_fmadd_ps(a, b2, c12);
            a = _mm256_broadcast_ss(pA + 2);
            c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); c22 = _mm256_fmadd_ps(a, b2, c22);
            a = _mm256_broadcast_ss(pA + 3);
            c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); c32 = _mm256_fmadd_ps(a, b2, c32);
        }
        /* --- k+1 --- */
        {
            __m256 b0 = _mm256_load_ps(pB + NR_4x24),
                   b1 = _mm256_load_ps(pB + NR_4x24 + 8),
                   b2 = _mm256_load_ps(pB + NR_4x24 + 16);
            __m256 a;
            a = _mm256_broadcast_ss(pA + MR_4x24 + 0);
            c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); c02 = _mm256_fmadd_ps(a, b2, c02);
            a = _mm256_broadcast_ss(pA + MR_4x24 + 1);
            c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); c12 = _mm256_fmadd_ps(a, b2, c12);
            a = _mm256_broadcast_ss(pA + MR_4x24 + 2);
            c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); c22 = _mm256_fmadd_ps(a, b2, c22);
            a = _mm256_broadcast_ss(pA + MR_4x24 + 3);
            c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); c32 = _mm256_fmadd_ps(a, b2, c32);
        }
        /* --- k+2 --- */
        {
            __m256 b0 = _mm256_load_ps(pB + 2 * NR_4x24),
                   b1 = _mm256_load_ps(pB + 2 * NR_4x24 + 8),
                   b2 = _mm256_load_ps(pB + 2 * NR_4x24 + 16);
            __m256 a;
            a = _mm256_broadcast_ss(pA + 2 * MR_4x24 + 0);
            c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); c02 = _mm256_fmadd_ps(a, b2, c02);
            a = _mm256_broadcast_ss(pA + 2 * MR_4x24 + 1);
            c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); c12 = _mm256_fmadd_ps(a, b2, c12);
            a = _mm256_broadcast_ss(pA + 2 * MR_4x24 + 2);
            c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); c22 = _mm256_fmadd_ps(a, b2, c22);
            a = _mm256_broadcast_ss(pA + 2 * MR_4x24 + 3);
            c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); c32 = _mm256_fmadd_ps(a, b2, c32);
        }
        /* --- k+3 --- */
        {
            __m256 b0 = _mm256_load_ps(pB + 3 * NR_4x24),
                   b1 = _mm256_load_ps(pB + 3 * NR_4x24 + 8),
                   b2 = _mm256_load_ps(pB + 3 * NR_4x24 + 16);
            __m256 a;
            a = _mm256_broadcast_ss(pA + 3 * MR_4x24 + 0);
            c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01); c02 = _mm256_fmadd_ps(a, b2, c02);
            a = _mm256_broadcast_ss(pA + 3 * MR_4x24 + 1);
            c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); c12 = _mm256_fmadd_ps(a, b2, c12);
            a = _mm256_broadcast_ss(pA + 3 * MR_4x24 + 2);
            c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); c22 = _mm256_fmadd_ps(a, b2, c22);
            a = _mm256_broadcast_ss(pA + 3 * MR_4x24 + 3);
            c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); c32 = _mm256_fmadd_ps(a, b2, c32);
        }

        pA += 4 * MR_4x24;
        pB += 4 * NR_4x24;
    }

    /* ── ×2 peel ─────────────────────────────────────────────── */
    for (; k <= k_len - 2; k += 2) {
        __m256 b00 = _mm256_load_ps(pB),       b01 = _mm256_load_ps(pB + 8),
               b02 = _mm256_load_ps(pB + 16);
        __m256 a;
        a = _mm256_broadcast_ss(pA + 0);
        c00 = _mm256_fmadd_ps(a, b00, c00); c01 = _mm256_fmadd_ps(a, b01, c01); c02 = _mm256_fmadd_ps(a, b02, c02);
        a = _mm256_broadcast_ss(pA + 1);
        c10 = _mm256_fmadd_ps(a, b00, c10); c11 = _mm256_fmadd_ps(a, b01, c11); c12 = _mm256_fmadd_ps(a, b02, c12);
        a = _mm256_broadcast_ss(pA + 2);
        c20 = _mm256_fmadd_ps(a, b00, c20); c21 = _mm256_fmadd_ps(a, b01, c21); c22 = _mm256_fmadd_ps(a, b02, c22);
        a = _mm256_broadcast_ss(pA + 3);
        c30 = _mm256_fmadd_ps(a, b00, c30); c31 = _mm256_fmadd_ps(a, b01, c31); c32 = _mm256_fmadd_ps(a, b02, c32);

        __m256 b10 = _mm256_load_ps(pB + NR_4x24),
               b11 = _mm256_load_ps(pB + NR_4x24 + 8),
               b12 = _mm256_load_ps(pB + NR_4x24 + 16);
        a = _mm256_broadcast_ss(pA + MR_4x24 + 0);
        c00 = _mm256_fmadd_ps(a, b10, c00); c01 = _mm256_fmadd_ps(a, b11, c01); c02 = _mm256_fmadd_ps(a, b12, c02);
        a = _mm256_broadcast_ss(pA + MR_4x24 + 1);
        c10 = _mm256_fmadd_ps(a, b10, c10); c11 = _mm256_fmadd_ps(a, b11, c11); c12 = _mm256_fmadd_ps(a, b12, c12);
        a = _mm256_broadcast_ss(pA + MR_4x24 + 2);
        c20 = _mm256_fmadd_ps(a, b10, c20); c21 = _mm256_fmadd_ps(a, b11, c21); c22 = _mm256_fmadd_ps(a, b12, c22);
        a = _mm256_broadcast_ss(pA + MR_4x24 + 3);
        c30 = _mm256_fmadd_ps(a, b10, c30); c31 = _mm256_fmadd_ps(a, b11, c31); c32 = _mm256_fmadd_ps(a, b12, c32);
        pA += 2 * MR_4x24;
        pB += 2 * NR_4x24;
    }

    /* ── ×1 tail ─────────────────────────────────────────────── */
    for (; k < k_len; k++) {
        __m256 b0 = _mm256_load_ps(pB), b1 = _mm256_load_ps(pB + 8), b2 = _mm256_load_ps(pB + 16);
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

    /* ── Store: C[i,:] = alpha*acc[i] + beta*C[i,:] ─────────── */
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vb = _mm256_set1_ps(beta);

#define STORE_ROW24(r0, r1, r2, row) do {                                      \
    float *Cr = C + (row) * ldc;                                               \
    __m256 res0 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr),      _mm256_mul_ps(va, r0)); \
    __m256 res1 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr + 8),  _mm256_mul_ps(va, r1)); \
    __m256 res2 = _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr + 16), _mm256_mul_ps(va, r2)); \
    if (use_nt_store && ((uintptr_t)Cr % 32 == 0)) {                           \
        _mm256_stream_ps(Cr,      res0);                                       \
        _mm256_stream_ps(Cr + 8,  res1);                                       \
        _mm256_stream_ps(Cr + 16, res2);                                       \
    } else {                                                                   \
        _mm256_storeu_ps(Cr,      res0);                                       \
        _mm256_storeu_ps(Cr + 8,  res1);                                       \
        _mm256_storeu_ps(Cr + 16, res2);                                       \
    }                                                                          \
} while (0)

    STORE_ROW24(c00, c01, c02, 0);
    STORE_ROW24(c10, c11, c12, 1);
    STORE_ROW24(c20, c21, c22, 2);
    STORE_ROW24(c30, c31, c32, 3);

#undef STORE_ROW24
}

#endif /* KERNEL_4X24_H */
