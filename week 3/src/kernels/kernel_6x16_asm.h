/*
 * kernel_6x16_asm.h  —  6×16 SGEMM micro-kernel in GCC inline x86-64 assembly
 *                        (Week 3)
 *
 * Week 3 changes vs Week 2:
 *   - Main loop unrolled ×4 (processes 4 k-steps per iteration)
 *   - Next-tile prefetch: issued from C before the asm block (first 32 CLs
 *     of next_A and next_B; hardware prefetcher handles the rest since
 *     packed tiles are sequential)
 *   - ×1 tail loop handles k_len % 4 remainder inside asm
 *   - NT store support in C store phase (use_nt_store flag)
 *
 * GCC inline-asm operand budget:
 *   GCC counts each "+x"/"+r" operand as BOTH input and output → each uses
 *   2 slots out of the 30-operand limit.  We use 12 "+x" + 3 "+r" = 30 total,
 *   exactly at the limit.  k_len/4 and remainder are computed inside the asm
 *   using rax as a scratch register (listed in the clobber list).
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
#include <stdint.h>

#define MR_6x16_ASM  6
#define NR_6x16_ASM  16

/* Maximum cache lines to pre-issue for next tile.
 * Modern Intel has ~12-16 fill buffers; 16 per pointer is a safe cap. */
#define ASM_PREFETCH_CLS 16

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

    /* ── Pre-issue next-tile prefetches from C ───────────────────────────
     * Issue the first ASM_PREFETCH_CLS cache lines of next_A and next_B.
     * The hardware prefetcher handles the rest (sequential access pattern).
     * This avoids including npa/npb as asm operands (would push past limit). */
    {
        size_t nA_bytes = (size_t)k_len * MR_6x16_ASM * sizeof(float);
        size_t nB_bytes = (size_t)k_len * NR_6x16_ASM * sizeof(float);
        for (int cl = 0; cl < ASM_PREFETCH_CLS; cl++) {
            size_t off = (size_t)cl * 64;
            if (off < nA_bytes)
                _mm_prefetch((const char *)next_A + off, _MM_HINT_T0);
            if (off < nB_bytes)
                _mm_prefetch((const char *)next_B + off, _MM_HINT_T0);
        }
    }

    /* ── Inner k-loop in extended GCC inline assembly ───────────────────
     *
     * Operand budget (GCC counts "+x"/"+r" as 2 each):
     *   12 "+x"  (accumulators)  → 24
     *    3 "+r"  (pA, pB, k4)   → 6
     *   Total: 30  (exactly at limit)
     *
     * k4 = k_len/4 is pre-computed in C and passed as the main loop counter.
     * The 0-3 remainder iterations are handled by a C tail loop after the asm.
     *
     * pA stride per ×4 iteration: 4×MR×4 = 96 bytes
     * pB stride per ×4 iteration: 4×NR×4 = 256 bytes
     * ──────────────────────────────────────────────────────────────────── */

    __asm__ volatile (
        /* ════════ ×4 MAIN LOOP ════════ */
        "movl   %[klen], %%eax\n\t"    /* copy k_len */
        "shrl   $2, %%eax\n\t"         /* eax = k_len / 4 */
        "test   %%eax, %%eax\n\t"
        "jz     3f\n\t"
        "1:\n\t"

        /* k+0 */
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

        /* k+1 */
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

        /* k+2 */
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

        /* k+3 */
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

        "add    $96,    %[pa]\n\t"      /* 4×MR×4 bytes */
        "add    $256,   %[pb]\n\t"      /* 4×NR×4 bytes */
        "dec    %%eax\n\t"
        "jnz    1b\n\t"
        "3:\n\t"

        /* ════════ ×1 TAIL LOOP (k_len % 4 remainder) ════════ */
        "and    $3, %[klen]\n\t"        /* k_len &= 3  */
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
        "dec    %[klen]\n\t"
        "jnz    4b\n\t"
        "2:\n\t"

        : /* outputs — 12 accumulators + pa, pb, klen (= 30 operand slots) */
          [a0lo] "+x" (acc0lo), [a0hi] "+x" (acc0hi),
          [a1lo] "+x" (acc1lo), [a1hi] "+x" (acc1hi),
          [a2lo] "+x" (acc2lo), [a2hi] "+x" (acc2hi),
          [a3lo] "+x" (acc3lo), [a3hi] "+x" (acc3hi),
          [a4lo] "+x" (acc4lo), [a4hi] "+x" (acc4hi),
          [a5lo] "+x" (acc5lo), [a5hi] "+x" (acc5hi),
          [pa]   "+r" (pA),
          [pb]   "+r" (pB),
          [klen] "+r" (k_len)
        : /* inputs — none beyond the above */
        : /* clobbers */
          "eax", "ymm12", "ymm13", "ymm14", "memory"
    );

    /* ── Store phase (C intrinsics, NT-aware) ─────────────────────────── */
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
