/*
 * kernel_8x8.h  —  8×8 AVX2+FMA SGEMM micro-kernel  (Week 3)
 *
 * Computes:  C[8×8] += alpha * A[8×k] * B[k×8]  (packed inputs)
 *
 * Week 3 changes vs Week 2:
 *   - Main loop unrolled ×4 (was ×2) for reduced branch overhead
 *   - Next-tile software prefetch: next_A / next_B spread across ×4 loop
 *     (2 CLs of A and 2 CLs of B per ×4 iteration: 4×8×4=128 bytes each)
 *   - NT store support: _mm256_stream_ps when aligned + use_nt_store flag
 *
 * Register plan (AVX2, 16 YMM of 8 floats each):
 *   ymm0–ymm7   : 8 accumulators, one per output row
 *   ymm8–ymm9   : B vectors (loaded each k-step)
 *   ymm10–ymm15 : A broadcasts + alpha/beta
 *
 *   8 FMAs per k-step = 128 FLOP/step
 */

#ifndef KERNEL_8X8_H
#define KERNEL_8X8_H

#include <immintrin.h>
#include <stdint.h>

#define MR_8x8  8
#define NR_8x8  8

static inline __attribute__((always_inline))
void micro_kernel_8x8(const float * __restrict__ pA,
                      const float * __restrict__ pB,
                      float       * __restrict__ C,
                      int k_len, float alpha, float beta, int ldc,
                      const float * __restrict__ next_A,
                      const float * __restrict__ next_B,
                      int use_nt_store)
{
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();

    /* Next-tile prefetch pointers — advanced through next tile each iteration.
     * Each ×4 step consumes 4×8×4 = 128 bytes = 2 cache lines of A and B. */
    const float *pfA = next_A;
    const float *pfB = next_B;

    int k = 0;

    /* ── Main loop: ×4 unrolled ──────────────────────────────────────── */
    for (; k <= k_len - 4; k += 4) {

        /* Prefetch 2 cache lines of next tile A and B */
        _mm_prefetch((const char *)(pfA),      _MM_HINT_T0);
        _mm_prefetch((const char *)(pfA + 16), _MM_HINT_T0);
        pfA += 4 * MR_8x8;   /* advance by 4 k-steps = 128 bytes */

        _mm_prefetch((const char *)(pfB),      _MM_HINT_T0);
        _mm_prefetch((const char *)(pfB + 16), _MM_HINT_T0);
        pfB += 4 * NR_8x8;

        /* k+0 */
        {
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
        }
        /* k+1 */
        {
            __m256 b = _mm256_load_ps(pB + NR_8x8);
            __m256 a;
            a = _mm256_broadcast_ss(pA + MR_8x8 + 0); c0 = _mm256_fmadd_ps(a, b, c0);
            a = _mm256_broadcast_ss(pA + MR_8x8 + 1); c1 = _mm256_fmadd_ps(a, b, c1);
            a = _mm256_broadcast_ss(pA + MR_8x8 + 2); c2 = _mm256_fmadd_ps(a, b, c2);
            a = _mm256_broadcast_ss(pA + MR_8x8 + 3); c3 = _mm256_fmadd_ps(a, b, c3);
            a = _mm256_broadcast_ss(pA + MR_8x8 + 4); c4 = _mm256_fmadd_ps(a, b, c4);
            a = _mm256_broadcast_ss(pA + MR_8x8 + 5); c5 = _mm256_fmadd_ps(a, b, c5);
            a = _mm256_broadcast_ss(pA + MR_8x8 + 6); c6 = _mm256_fmadd_ps(a, b, c6);
            a = _mm256_broadcast_ss(pA + MR_8x8 + 7); c7 = _mm256_fmadd_ps(a, b, c7);
        }
        /* k+2 */
        {
            __m256 b = _mm256_load_ps(pB + 2 * NR_8x8);
            __m256 a;
            a = _mm256_broadcast_ss(pA + 2 * MR_8x8 + 0); c0 = _mm256_fmadd_ps(a, b, c0);
            a = _mm256_broadcast_ss(pA + 2 * MR_8x8 + 1); c1 = _mm256_fmadd_ps(a, b, c1);
            a = _mm256_broadcast_ss(pA + 2 * MR_8x8 + 2); c2 = _mm256_fmadd_ps(a, b, c2);
            a = _mm256_broadcast_ss(pA + 2 * MR_8x8 + 3); c3 = _mm256_fmadd_ps(a, b, c3);
            a = _mm256_broadcast_ss(pA + 2 * MR_8x8 + 4); c4 = _mm256_fmadd_ps(a, b, c4);
            a = _mm256_broadcast_ss(pA + 2 * MR_8x8 + 5); c5 = _mm256_fmadd_ps(a, b, c5);
            a = _mm256_broadcast_ss(pA + 2 * MR_8x8 + 6); c6 = _mm256_fmadd_ps(a, b, c6);
            a = _mm256_broadcast_ss(pA + 2 * MR_8x8 + 7); c7 = _mm256_fmadd_ps(a, b, c7);
        }
        /* k+3 */
        {
            __m256 b = _mm256_load_ps(pB + 3 * NR_8x8);
            __m256 a;
            a = _mm256_broadcast_ss(pA + 3 * MR_8x8 + 0); c0 = _mm256_fmadd_ps(a, b, c0);
            a = _mm256_broadcast_ss(pA + 3 * MR_8x8 + 1); c1 = _mm256_fmadd_ps(a, b, c1);
            a = _mm256_broadcast_ss(pA + 3 * MR_8x8 + 2); c2 = _mm256_fmadd_ps(a, b, c2);
            a = _mm256_broadcast_ss(pA + 3 * MR_8x8 + 3); c3 = _mm256_fmadd_ps(a, b, c3);
            a = _mm256_broadcast_ss(pA + 3 * MR_8x8 + 4); c4 = _mm256_fmadd_ps(a, b, c4);
            a = _mm256_broadcast_ss(pA + 3 * MR_8x8 + 5); c5 = _mm256_fmadd_ps(a, b, c5);
            a = _mm256_broadcast_ss(pA + 3 * MR_8x8 + 6); c6 = _mm256_fmadd_ps(a, b, c6);
            a = _mm256_broadcast_ss(pA + 3 * MR_8x8 + 7); c7 = _mm256_fmadd_ps(a, b, c7);
        }

        pA += 4 * MR_8x8;
        pB += 4 * NR_8x8;
    }

    /* ── ×2 peel ─────────────────────────────────────────────── */
    for (; k <= k_len - 2; k += 2) {
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

    /* ── ×1 tail ─────────────────────────────────────────────── */
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

    /* ── Store: C[i,:] = alpha*acc[i] + beta*C[i,:] ─────────── */
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vb = _mm256_set1_ps(beta);

#define STORE_ROW8(ci, row) do {                                               \
    float *Cr = C + (row) * ldc;                                               \
    __m256 res = _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr),                     \
                                 _mm256_mul_ps(va, ci));                        \
    if (use_nt_store && ((uintptr_t)Cr % 32 == 0))                             \
        _mm256_stream_ps(Cr, res);                                             \
    else                                                                       \
        _mm256_storeu_ps(Cr, res);                                             \
} while (0)

    STORE_ROW8(c0, 0); STORE_ROW8(c1, 1);
    STORE_ROW8(c2, 2); STORE_ROW8(c3, 3);
    STORE_ROW8(c4, 4); STORE_ROW8(c5, 5);
    STORE_ROW8(c6, 6); STORE_ROW8(c7, 7);

#undef STORE_ROW8
}

#endif /* KERNEL_8X8_H */
