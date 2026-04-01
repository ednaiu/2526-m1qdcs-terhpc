/*
 * sgemm.c  —  BLIS-style 5-loop tiled SGEMM with AVX2+FMA 6×16 micro-kernel
 *
 * Parallelization: Single parallel region, 2D decomposition over
 *   (ic, jr) tiles for maximum thread utilization on all sizes.
 *   B packing is parallelized across threads with barrier sync.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

#include "../include/sgemm.h"
#include "sgemm_kernel.h"

/* ============================================================
 * Packing helpers
 * ============================================================ */

static void pack_A_panel(const float * __restrict__ A, int lda,
                         float * __restrict__ packed,
                         int M_blk, int K_blk)
{
    int nr_strips = (M_blk + MR - 1) / MR;

    for (int s = 0; s < nr_strips; s++) {
        int row_start = s * MR;
        int m_curr    = (row_start + MR <= M_blk) ? MR : (M_blk - row_start);
        float *dst_base = packed + (size_t)s * K_blk * MR;

        if (m_curr == MR) {
            for (int k = 0; k < K_blk; k++) {
                float *dst = dst_base + k * MR;
                for (int r = 0; r < MR; r++)
                    dst[r] = A[(row_start + r) * lda + k];
            }
        } else {
            for (int k = 0; k < K_blk; k++) {
                float *dst = dst_base + k * MR;
                int r;
                for (r = 0; r < m_curr; r++)
                    dst[r] = A[(row_start + r) * lda + k];
                for (; r < MR; r++)
                    dst[r] = 0.0f;
            }
        }
    }
}

static void pack_B_panel_parallel(const float * __restrict__ B, int ldb,
                                   float * __restrict__ packed,
                                   int K_blk, int N_blk)
{
    int nr_strips = (N_blk + NR - 1) / NR;

    #pragma omp for schedule(static) nowait
    for (int t = 0; t < nr_strips; t++) {
        int col_start = t * NR;
        int n_curr    = (col_start + NR <= N_blk) ? NR : (N_blk - col_start);
        float *dst_base = packed + (size_t)t * K_blk * NR;

        if (n_curr == NR) {
            for (int k = 0; k < K_blk; k++) {
                float *dst       = dst_base + k * NR;
                const float *src = B + k * ldb + col_start;
                _mm256_store_ps(dst,     _mm256_loadu_ps(src));
                _mm256_store_ps(dst + 8, _mm256_loadu_ps(src + 8));
            }
        } else {
            for (int k = 0; k < K_blk; k++) {
                float *dst       = dst_base + k * NR;
                const float *src = B + k * ldb + col_start;
                int c;
                for (c = 0; c < n_curr; c++)
                    dst[c] = src[c];
                for (; c < NR; c++)
                    dst[c] = 0.0f;
            }
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
 * Optimised SGEMM  (C = alpha*A*B + beta*C)
 *
 * 2D parallelization over (ic, jr) tiles:
 *   Total tasks = ceil(M/MC) * ceil(N_blk/NR)
 *   This provides much better load balance for small matrices.
 * ============================================================ */
void sgemm(int M, int N, int K,
           float alpha,
           const float *A, int lda,
           const float *B, int ldb,
           float beta,
           float *C, int ldc)
{
    int mc = MC;
    int kc = KC;
    int nc = NC;

    /* Shared B packing buffer */
    int B_strips_max = (nc + NR - 1) / NR;
    float *packed_B = (float *)aligned_alloc(32,
                        (size_t)B_strips_max * kc * NR * sizeof(float));
    if (!packed_B) exit(1);

    #pragma omp parallel
    {
        /* Thread-local A packing buffer + edge-tile C buffer */
        int A_strips_max = (mc + MR - 1) / MR;
        float *packed_A = (float *)aligned_alloc(32,
                            (size_t)A_strips_max * kc * MR * sizeof(float));
        if (!packed_A) exit(1);
        float C_buf[MR * NR] __attribute__((aligned(32)));

        for (int jc = 0; jc < N; jc += nc) {
            int N_blk = (jc + nc <= N) ? nc : (N - jc);
            int nr_j_strips = (N_blk + NR - 1) / NR;

            for (int pc = 0; pc < K; pc += kc) {
                int K_blk = (pc + kc <= K) ? kc : (K - pc);
                float beta_k = (pc == 0) ? beta : 1.0f;

                /* Parallel B packing */
                pack_B_panel_parallel(B + pc * ldb + jc, ldb,
                                       packed_B, K_blk, N_blk);
                #pragma omp barrier

                /* 2D parallel decomposition: each task is one (ic, jr) tile */
                int nr_i_tiles = (M + mc - 1) / mc;
                int total_tasks = nr_i_tiles * nr_j_strips;

                #pragma omp for schedule(dynamic) nowait
                for (int task = 0; task < total_tasks; task++) {
                    int ic_idx = task / nr_j_strips;
                    int jt     = task % nr_j_strips;

                    int ic    = ic_idx * mc;
                    int M_blk = (ic + mc <= M) ? mc : (M - ic);
                    int n_curr = (jt * NR + NR <= N_blk) ? NR : (N_blk - jt * NR);
                    int cj     = jt * NR;

                    /* Pack A for this ic block */
                    pack_A_panel(A + ic * lda + pc, lda, packed_A, M_blk, K_blk);

                    /* Compute micro-tiles for this (ic, jr) strip */
                    const float *pB = packed_B + (size_t)jt * K_blk * NR;
                    int nr_m_strips = (M_blk + MR - 1) / MR;

                    for (int it = 0; it < nr_m_strips; it++) {
                        int m_curr = (it * MR + MR <= M_blk) ? MR : (M_blk - it * MR);
                        const float *pA = packed_A + (size_t)it * K_blk * MR;
                        int ci = it * MR;

                        if (m_curr == MR && n_curr == NR) {
                            micro_kernel(pA, pB,
                                         C + (ic + ci) * ldc + (jc + cj), K_blk,
                                         alpha, beta_k, ldc);
                        } else {
                            memset(C_buf, 0, sizeof(C_buf));
                            for (int ei = 0; ei < m_curr; ei++)
                                for (int ej = 0; ej < n_curr; ej++)
                                    C_buf[ei * NR + ej] = C[(ic+ci+ei)*ldc + (jc+cj+ej)];

                            micro_kernel(pA, pB, C_buf, K_blk,
                                         alpha, beta_k, NR);

                            for (int ei = 0; ei < m_curr; ei++)
                                for (int ej = 0; ej < n_curr; ej++)
                                    C[(ic+ci+ei)*ldc + (jc+cj+ej)] = C_buf[ei * NR + ej];
                        }
                    }
                }

                #pragma omp barrier
            }
        }

        free(packed_A);
    }

    free(packed_B);
}
