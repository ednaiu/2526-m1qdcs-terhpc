/*
 * sgemm.c  —  Parallel BLIS-style 5-loop tiled SGEMM with AVX2+FMA micro-kernel
 *
 * Parallelization Strategy:
 *   Parallelize Loop 3 (ic). All threads share the packed B panel for the 
 *   current (jc, pc) tile. Each thread uses its own packed A buffer and C_buf.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "../include/sgemm.h"
#include "sgemm_kernel.h"

/* ============================================================
 * Packing helpers
 * ============================================================ */

/*
 * pack_A_panel
 *   packed_A layout: [ceil(M_blk/MR)][K_blk][MR]
 */
static void pack_A_panel(const float * __restrict__ A, int lda,
                         float * __restrict__ packed,
                         int M_blk, int K_blk)
{
    int nr_strips = (M_blk + MR - 1) / MR;

    for (int s = 0; s < nr_strips; s++) {
        int row_start = s * MR;
        int m_curr    = (row_start + MR <= M_blk) ? MR : (M_blk - row_start);

        for (int k = 0; k < K_blk; k++) {
            float *dst = packed + (s * K_blk + k) * MR;
            int r;
            for (r = 0; r < m_curr; r++)
                dst[r] = A[(row_start + r) * lda + k];
            for (; r < MR; r++)
                dst[r] = 0.0f;
        }
    }
}

/*
 * pack_B_panel
 *   packed_B layout: [ceil(N_blk/NR)][K_blk][NR]
 */
static void pack_B_panel(const float * __restrict__ B, int ldb,
                         float * __restrict__ packed,
                         int K_blk, int N_blk)
{
    int nr_strips = (N_blk + NR - 1) / NR;

    for (int t = 0; t < nr_strips; t++) {
        int col_start = t * NR;
        int n_curr    = (col_start + NR <= N_blk) ? NR : (N_blk - col_start);

        for (int k = 0; k < K_blk; k++) {
            float *dst        = packed + (t * K_blk + k) * NR;
            const float *src  = B + k * ldb + col_start;
            int c;
            for (c = 0; c < n_curr; c++)
                dst[c] = src[c];
            for (; c < NR; c++)
                dst[c] = 0.0f;
        }
    }
}

/* ============================================================
 * Naive reference (for correctness tests)
 * ============================================================ */
void sgemm_ref(int M, int N, int K,
               float alpha,
               const float *A, int lda,
               const float *B, int ldb,
               float beta,
               float *C, int ldc)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i*lda + k] * B[k*ldb + j];
            C[i*ldc + j] = alpha * sum + beta * C[i*ldc + j];
        }
    }
}

/* ============================================================
 * Optimised parallel SGEMM  (C = alpha*A*B + beta*C)
 * ============================================================ */
void sgemm(int M, int N, int K,
           float alpha,
           const float *A, int lda,
           const float *B, int ldb,
           float beta,
           float *C, int ldc)
{
    /* Dynamic blocking sizes */
    // Ensure we have enough tiles to keep all threads busy.
    // If we have up to 20 threads, we want at least 20 tiles.
    int mc_dyn = 64;
    int nc_dyn = 128; // Smaller NC means more independent tasks on N
    int kc_dyn = 256;

    #pragma omp parallel
    {
        /* Thread-local buffers */
        int A_strips_max = (mc_dyn + MR - 1) / MR;
        int B_strips_max = (nc_dyn + NR - 1) / NR;
        
        float *packed_A = (float *)aligned_alloc(32, (size_t)A_strips_max * kc_dyn * MR * sizeof(float));
        float *packed_B = (float *)aligned_alloc(32, (size_t)B_strips_max * kc_dyn * NR * sizeof(float));
        float *C_buf    = (float *)aligned_alloc(32, (size_t)MR * NR * sizeof(float));
        
        if (!packed_A || !packed_B || !C_buf) { exit(1); }

        #pragma omp for collapse(2) schedule(dynamic)
        for (int jc = 0; jc < N; jc += nc_dyn) {
            for (int ic = 0; ic < M; ic += mc_dyn) {
                int M_blk = (ic + mc_dyn <= M) ? mc_dyn : (M - ic);
                int N_blk = (jc + nc_dyn <= N) ? nc_dyn : (N - jc);

                // For a specific tile of C, iterate over K chunks
                for (int pc = 0; pc < K; pc += kc_dyn) {
                    int K_blk = (pc + kc_dyn <= K) ? kc_dyn : (K - pc);

                    /* Pack panels (thread-local) */
                    pack_A_panel(A + ic * lda + pc, lda, packed_A, M_blk, K_blk);
                    pack_B_panel(B + pc * ldb + jc, ldb, packed_B, K_blk, N_blk);

                    float beta_k = (pc == 0) ? beta : 1.0f;
                    int nr_n_strips = (N_blk + NR - 1) / NR;
                    int nr_m_strips = (M_blk + MR - 1) / MR;

                    for (int jt = 0; jt < nr_n_strips; jt++) {
                        int n_curr = (jt * NR + NR <= N_blk) ? NR : (N_blk - jt * NR);
                        const float *pB = packed_B + jt * K_blk * NR;

                        for (int it = 0; it < nr_m_strips; it++) {
                            int m_curr = (it * MR + MR <= M_blk) ? MR : (M_blk - it * MR);
                            const float *pA = packed_A + it * K_blk * MR;
                            int ci = ic + it * MR;
                            int cj = jc + jt * NR;

                            if (m_curr == MR && n_curr == NR) {
                                micro_kernel(pA, pB, C + ci * ldc + cj, K_blk, alpha, beta_k, ldc);
                            } else {
                                memset(C_buf, 0, MR * NR * sizeof(float));
                                for (int ei = 0; ei < m_curr; ei++)
                                    for (int ej = 0; ej < n_curr; ej++)
                                        C_buf[ei * NR + ej] = C[(ci+ei)*ldc + (cj+ej)];

                                micro_kernel(pA, pB, C_buf, K_blk, alpha, beta_k, NR);

                                for (int ei = 0; ei < m_curr; ei++)
                                    for (int ej = 0; ej < n_curr; ej++)
                                        C[(ci+ei)*ldc + (cj+ej)] = C_buf[ei * NR + ej];
                            }
                        }
                    }
                }
            }
        }
        free(packed_A);
        free(packed_B);
        free(C_buf);
    }
}
