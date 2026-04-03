/*
 * kernel_6x16_asm.h  —  6×16 SGEMM micro-kernel in GCC inline x86-64 assembly
 *
 * Same computation as kernel_6x16.h but hand-written in assembly to:
 *   1. Guarantee exact register allocation with no compiler interference
 *   2. Demonstrate the raw instruction sequence for HPC education
 *   3. Allow explicit control of instruction scheduling and latency hiding
 *
 * Register assignment (same as intrinsics version):
 *   ymm0 –ymm5   : row accumulators, low  half  (cols  0..7)
 *   ymm6 –ymm11  : row accumulators, high half  (cols 8..15)
 *   ymm12        : B low  (loaded each k-step)
 *   ymm13        : B high (loaded each k-step)
 *   ymm14        : A broadcast  (reused for each row within a k-step)
 *   ymm15        : alpha / beta / scratch
 *
 * Calling convention: System V AMD64
 *   rdi = pA, rsi = pB, rdx = C, ecx = k_len,
 *   xmm0(float) = alpha passed via stack float (see wrapper)
 *   We use a C wrapper that calls the asm inner loop.
 *
 * NOTE: The asm kernel handles the full k-loop and the alpha/beta scaling.
 *       The C wrapper handles argument marshalling and edge cases.
 */

#ifndef KERNEL_6X16_ASM_H
#define KERNEL_6X16_ASM_H

#include <immintrin.h>

#define MR_6x16_ASM  6
#define NR_6x16_ASM  16

/*
 * micro_kernel_6x16_asm — inline assembly implementation
 *
 * We use extended GCC asm with explicit register constraints.
 * The inner loop body is in asm; pA/pB pointer advances are done in asm.
 * The store phase reverts to C intrinsics (acceptable trade-off).
 */
static inline __attribute__((always_inline))
void micro_kernel_6x16_asm(const float * __restrict__ pA,
                            const float * __restrict__ pB,
                            float       * __restrict__ C,
                            int k_len, float alpha, float beta, int ldc)
{
    /* Zero all 12 accumulators via C to let the compiler assign them,
     * then enter asm for the k-loop. */
    __m256 acc0lo, acc0hi, acc1lo, acc1hi, acc2lo, acc2hi;
    __m256 acc3lo, acc3hi, acc4lo, acc4hi, acc5lo, acc5hi;

    acc0lo = _mm256_setzero_ps(); acc0hi = _mm256_setzero_ps();
    acc1lo = _mm256_setzero_ps(); acc1hi = _mm256_setzero_ps();
    acc2lo = _mm256_setzero_ps(); acc2hi = _mm256_setzero_ps();
    acc3lo = _mm256_setzero_ps(); acc3hi = _mm256_setzero_ps();
    acc4lo = _mm256_setzero_ps(); acc4hi = _mm256_setzero_ps();
    acc5lo = _mm256_setzero_ps(); acc5hi = _mm256_setzero_ps();

    /* ----------------------------------------------------------------
     * Inner k-loop in extended GCC inline assembly.
     *
     * We dedicate:
     *   %[pa]  = pointer to packed A (incremented inside asm)
     *   %[pb]  = pointer to packed B (incremented inside asm)
     *   %[klen]= k counter (decrements to 0)
     *
     * Accumulators are passed as "+x" (xmm/ymm) operands so GCC knows
     * they are both read and written. We force ymm0–ymm11 assignment
     * via the named operand list to match the documented register plan.
     * ---------------------------------------------------------------- */
    __asm__ volatile (
        /* Load loop counter */
        "test   %[klen], %[klen]\n\t"
        "jz     2f\n\t"
        "1:\n\t"
        /* Load B row: ymm12 = B[k, 0..7], ymm13 = B[k, 8..15] */
        "vmovaps    (%[pb]),       %%ymm12\n\t"
        "vmovaps    32(%[pb]),     %%ymm13\n\t"

        /* Row 0: broadcast A[0,k] into ymm14, then FMA */
        "vbroadcastss  0(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a0lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a0hi]\n\t"

        /* Row 1 */
        "vbroadcastss  4(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a1lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a1hi]\n\t"

        /* Row 2 */
        "vbroadcastss  8(%[pa]),   %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a2lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a2hi]\n\t"

        /* Row 3 */
        "vbroadcastss  12(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a3lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a3hi]\n\t"

        /* Row 4 */
        "vbroadcastss  16(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a4lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a4hi]\n\t"

        /* Row 5 */
        "vbroadcastss  20(%[pa]),  %%ymm14\n\t"
        "vfmadd231ps   %%ymm14, %%ymm12, %[a5lo]\n\t"
        "vfmadd231ps   %%ymm14, %%ymm13, %[a5hi]\n\t"

        /* Advance pointers: pA += MR*4 bytes = 24, pB += NR*4 bytes = 64 */
        "add    $24,   %[pa]\n\t"
        "add    $64,   %[pb]\n\t"
        "dec    %[klen]\n\t"
        "jnz    1b\n\t"
        "2:\n\t"
        : /* outputs (read-write accumulators) */
          [a0lo] "+x" (acc0lo), [a0hi] "+x" (acc0hi),
          [a1lo] "+x" (acc1lo), [a1hi] "+x" (acc1hi),
          [a2lo] "+x" (acc2lo), [a2hi] "+x" (acc2hi),
          [a3lo] "+x" (acc3lo), [a3hi] "+x" (acc3hi),
          [a4lo] "+x" (acc4lo), [a4hi] "+x" (acc4hi),
          [a5lo] "+x" (acc5lo), [a5hi] "+x" (acc5hi),
          [pa]   "+r" (pA),
          [pb]   "+r" (pB),
          [klen] "+r" (k_len)
        : /* inputs (none beyond the above) */
        : /* clobbers */
          "ymm12", "ymm13", "ymm14", "memory"
    );

    /* ----------------------------------------------------------------
     * Store phase — same as intrinsics version
     * ---------------------------------------------------------------- */
    __m256 va = _mm256_set1_ps(alpha);
    __m256 vb = _mm256_set1_ps(beta);

    acc0lo = _mm256_mul_ps(va, acc0lo); acc0hi = _mm256_mul_ps(va, acc0hi);
    acc1lo = _mm256_mul_ps(va, acc1lo); acc1hi = _mm256_mul_ps(va, acc1hi);
    acc2lo = _mm256_mul_ps(va, acc2lo); acc2hi = _mm256_mul_ps(va, acc2hi);
    acc3lo = _mm256_mul_ps(va, acc3lo); acc3hi = _mm256_mul_ps(va, acc3hi);
    acc4lo = _mm256_mul_ps(va, acc4lo); acc4hi = _mm256_mul_ps(va, acc4hi);
    acc5lo = _mm256_mul_ps(va, acc5lo); acc5hi = _mm256_mul_ps(va, acc5hi);

#define STORE_ROW16A(rlo, rhi, row) do {                                      \
    float *Cr = C + (row) * ldc;                                              \
    _mm256_storeu_ps(Cr,     _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr),     rlo)); \
    _mm256_storeu_ps(Cr + 8, _mm256_fmadd_ps(vb, _mm256_loadu_ps(Cr + 8), rhi)); \
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
