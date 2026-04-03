/*
 * kernel_6x16_asm.h  —  6×16 SGEMM micro-kernel in GCC inline x86-64 assembly
 *                        (Week 3)
 *
 * Week 3 changes vs Week 2:
 *   - Main loop unrolled ×4: processes 4 k-steps per iteration
 *   - Next-tile prefetch embedded in the ×4 ASM loop (2 CLs of A, 4 CLs of B)
 *   - ×1 tail loop handles remainder (k_len % 4)
 *   - NT store support in C store phase (use_nt_store flag)
 *
 * Register assignment (unchanged):
 *   ymm0 –ymm5   : row accumulators, low  half  (cols  0..7)
 *   ymm6 –ymm11  : row accumulators, high half  (cols 8..15)
 *   ymm12        : B low  (loaded each k-step)
 *   ymm13        : B high (loaded each k-step)
 *   ymm14        : A broadcast
 *   ymm15        : scratch
 */

#ifndef KERNEL_6X16_ASM_H
#define KERNEL_6X16_ASM_H

#include <immintrin.h>

#define MR_6x16_ASM  6
#define NR_6x16_ASM  16

static inline __attribute__((always_inline))
void micro_kernel_6x16_asm(const float * __restrict__ pA,
                            const float * __restrict__ pB,
                            float       * __restrict__ C,
                            int k_len, float alpha, float beta, int ldc,
                            const float * __restrict__ next_A,
                            const float * __restrict__ next_B,
                            int use_nt_store)
{
    __m256 acc0lo, acc0hi, acc1lo, acc1hi, acc2lo, acc2hi;
    __m256 acc3lo, acc3hi, acc4lo, acc4hi, acc5lo, acc5hi;

    acc0lo = _mm256_setzero_ps(); acc0hi = _mm256_setzero_ps();
    acc1lo = _mm256_setzero_ps(); acc1hi = _mm256_setzero_ps();
    acc2lo = _mm256_setzero_ps(); acc2hi = _mm256_setzero_ps();
    acc3lo = _mm256_setzero_ps(); acc3hi = _mm256_setzero_ps();
    acc4lo = _mm256_setzero_ps(); acc4hi = _mm256_setzero_ps();
    acc5lo = _mm256_setzero_ps(); acc5hi = _mm256_setzero_ps();

    /* Split k into ×4 iterations and ×1 remainder */
    int k4 = k_len / 4;
    int kr = k_len & 3;

    /* Non-const copies for asm "+r" (pointer increment) */
    const float *npa = next_A;
    const float *npb = next_B;

    __asm__ volatile (
        /* ════════════════════════════════════════════════
         * ×4 UNROLLED MAIN LOOP
         * Each iteration processes k+0, k+1, k+2, k+3.
         * Prefetches 2 CLs of next_A, 4 CLs of next_B.
         * pA stride: 4×MR×4 = 96 bytes per iteration
         * pB stride: 4×NR×4 = 256 bytes per iteration
         * ════════════════════════════════════════════════ */
        "test   %[k4], %[k4]\n\t"
        "jz     3f\n\t"
        "1:\n\t"

        /* ── Next-tile prefetch ── */
        "prefetcht0    (%[npa])\n\t"
        "prefetcht0    64(%[npa])\n\t"
        "add    $96,    %[npa]\n\t"     /* 4×MR×4 = 96 bytes */

        "prefetcht0    (%[npb])\n\t"
        "prefetcht0    64(%[npb])\n\t"
        "prefetcht0    128(%[npb])\n\t"
        "prefetcht0    192(%[npb])\n\t"
        "add    $256,   %[npb]\n\t"     /* 4×NR×4 = 256 bytes */

        /* ── k+0: B at [pb+0], A at [pa+0] ── */
        "vmovaps    0(%[pb]),      %%ymm12\n\t"
        "vmovaps    32(%[pb]),     %%ymm13\n\t"
        "vbroadcastss  0(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a0lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a0hi]\n\t"
        "vbroadcastss  4(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a1lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a1hi]\n\t"
        "vbroadcastss  8(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a2lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a2hi]\n\t"
        "vbroadcastss  12(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a3lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a3hi]\n\t"
        "vbroadcastss  16(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a4lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a4hi]\n\t"
        "vbroadcastss  20(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a5lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a5hi]\n\t"

        /* ── k+1: B at [pb+64], A at [pa+24] ── */
        "vmovaps    64(%[pb]),     %%ymm12\n\t"
        "vmovaps    96(%[pb]),     %%ymm13\n\t"
        "vbroadcastss  24(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a0lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a0hi]\n\t"
        "vbroadcastss  28(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a1lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a1hi]\n\t"
        "vbroadcastss  32(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a2lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a2hi]\n\t"
        "vbroadcastss  36(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a3lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a3hi]\n\t"
        "vbroadcastss  40(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a4lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a4hi]\n\t"
        "vbroadcastss  44(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a5lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a5hi]\n\t"

        /* ── k+2: B at [pb+128], A at [pa+48] ── */
        "vmovaps    128(%[pb]),    %%ymm12\n\t"
        "vmovaps    160(%[pb]),    %%ymm13\n\t"
        "vbroadcastss  48(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a0lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a0hi]\n\t"
        "vbroadcastss  52(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a1lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a1hi]\n\t"
        "vbroadcastss  56(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a2lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a2hi]\n\t"
        "vbroadcastss  60(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a3lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a3hi]\n\t"
        "vbroadcastss  64(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a4lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a4hi]\n\t"
        "vbroadcastss  68(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a5lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a5hi]\n\t"

        /* ── k+3: B at [pb+192], A at [pa+72] ── */
        "vmovaps    192(%[pb]),    %%ymm12\n\t"
        "vmovaps    224(%[pb]),    %%ymm13\n\t"
        "vbroadcastss  72(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a0lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a0hi]\n\t"
        "vbroadcastss  76(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a1lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a1hi]\n\t"
        "vbroadcastss  80(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a2lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a2hi]\n\t"
        "vbroadcastss  84(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a3lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a3hi]\n\t"
        "vbroadcastss  88(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a4lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a4hi]\n\t"
        "vbroadcastss  92(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a5lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a5hi]\n\t"

        "add    $96,    %[pa]\n\t"  /* 4×MR×4 bytes */
        "add    $256,   %[pb]\n\t"  /* 4×NR×4 bytes */
        "dec    %[k4]\n\t"
        "jnz    1b\n\t"
        "3:\n\t"

        /* ════════════════════════════════════════════════
         * ×1 TAIL LOOP (handles k_len % 4 remainder)
         * ════════════════════════════════════════════════ */
        "test   %[kr], %[kr]\n\t"
        "jz     2f\n\t"
        "4:\n\t"
        "vmovaps    (%[pb]),       %%ymm12\n\t"
        "vmovaps    32(%[pb]),     %%ymm13\n\t"
        "vbroadcastss  0(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a0lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a0hi]\n\t"
        "vbroadcastss  4(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a1lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a1hi]\n\t"
        "vbroadcastss  8(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a2lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a2hi]\n\t"
        "vbroadcastss  12(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a3lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a3hi]\n\t"
        "vbroadcastss  16(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a4lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a4hi]\n\t"
        "vbroadcastss  20(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a5lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a5hi]\n\t"
        "add    $24,    %[pa]\n\t"
        "add    $64,    %[pb]\n\t"
        "dec    %[kr]\n\t"
        "jnz    4b\n\t"
        "2:\n\t"

        : /* outputs — read-write accumulators + pointer advances */
          [a0lo] "+x" (acc0lo), [a0hi] "+x" (acc0hi),
          [a1lo] "+x" (acc1lo), [a1hi] "+x" (acc1hi),
          [a2lo] "+x" (acc2lo), [a2hi] "+x" (acc2hi),
          [a3lo] "+x" (acc3lo), [a3hi] "+x" (acc3hi),
          [a4lo] "+x" (acc4lo), [a4hi] "+x" (acc4hi),
          [a5lo] "+x" (acc5lo), [a5hi] "+x" (acc5hi),
          [pa]   "+r" (pA),
          [pb]   "+r" (pB),
          [npa]  "+r" (npa),
          [npb]  "+r" (npb),
          [k4]   "+r" (k4),
          [kr]   "+r" (kr)
        : /* inputs (none beyond the above) */
        : /* clobbers */
          "ymm12", "ymm13", "ymm14", "memory"
    );

    /* ── Store phase (C intrinsics, same as week 2 but NT-aware) ── */
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vb = _mm256_set1_ps(beta);

    acc0lo = _mm256_mul_ps(va, acc0lo); acc0hi = _mm256_mul_ps(va, acc0hi);
    acc1lo = _mm256_mul_ps(va, acc1lo); acc1hi = _mm256_mul_ps(va, acc1hi);
    acc2lo = _mm256_mul_ps(va, acc2lo); acc2hi = _mm256_mul_ps(va, acc2hi);
    acc3lo = _mm256_mul_ps(va, acc3lo); acc3hi = _mm256_mul_ps(va, acc3hi);
    acc4lo = _mm256_mul_ps(va, acc4lo); acc4hi = _mm256_mul_ps(va, acc4hi);
    acc5lo = _mm256_mul_ps(va, acc5lo); acc5hi = _mm256_mul_ps(va, acc5hi);

#define STORE_ROW16A(rlo, rhi, row) do {                                       \
    float *Cr = C + (row) * ldc;                                               \
    __m256 rlo_res = _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr),      rlo);      \
    __m256 rhi_res = _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr + 8),  rhi);      \
    if (use_nt_store && ((uintptr_t)Cr % 32 == 0)) {                           \
        _mm256_stream_ps(Cr,     rlo_res);                                     \
        _mm256_stream_ps(Cr + 8, rhi_res);                                     \
    } else {                                                                   \
        _mm256_storeu_ps(Cr,     rlo_res);                                     \
        _mm256_storeu_ps(Cr + 8, rhi_res);                                     \
    }                                                                          \
} while (0)

    STORE_ROW16A(acc0lo, acc0hi, 0);
    STORE_ROW16A(acc1lo, acc1hi, 1);
    STORE_ROW16A(acc2lo, acc2hi, 2);
    STORE_ROW16A(acc3lo, acc3hi, 3);
    STORE_ROW16A(acc4lo, acc4hi, 4);
    STORE_ROW16A(acc5lo, acc5hi, 5);

#undef STORE_ROW16A
}

#endif /* KERNEL_6X16_ASM_H */
