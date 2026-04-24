/*
 * sgemm.c  —  Week 5/6: Optimized SGEMM with 3D Parallelism and Recursive Reduction
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#include "../include/sgemm.h"
#include "kernels/kernel_8x8.h"
#include "kernels/kernel_6x16.h"
#include "kernels/kernel_4x24.h"
#include "kernels/kernel_6x16_asm.h"

/* ======================================================================
 * Micro-kernel dispatch
 * ====================================================================== */
typedef void (*kernel_fn_t)(const float *, const float *, float *,
                             int, float, float, int,
                             const float *, const float *, int);

static kernel_fn_t get_kernel_fn(kernel_type_t kt)
{
    switch (kt) {
        case KERNEL_8x8:      return micro_kernel_8x8;
        case KERNEL_4x24:     return micro_kernel_4x24;
        case KERNEL_6x16_ASM: return micro_kernel_6x16_asm;
        case KERNEL_6x16:     /* fall through */
        default:              return micro_kernel_6x16;
    }
}

/* ======================================================================
 * Packing helpers
 * ====================================================================== */
static void pack_A_panel(const float * __restrict__ A, int lda,
                         float * __restrict__ packed,
                         int M_blk, int K_blk, int MR)
{
    int nr_strips = (M_blk + MR - 1) / MR;
    for (int s = 0; s < nr_strips; s++) {
        int row0   = s * MR;
        int m_curr = (row0 + MR <= M_blk) ? MR : (M_blk - row0);
        float *dst = packed + (size_t)s * K_blk * MR;

        if (m_curr == MR) {
            for (int k = 0; k < K_blk; k++) {
                float *d = dst + k * MR;
                for (int r = 0; r < MR; r++) d[r] = A[(row0 + r) * lda + k];
            }
        } else {
            for (int k = 0; k < K_blk; k++) {
                float *d = dst + k * MR;
                int r;
                for (r = 0; r < m_curr; r++) d[r] = A[(row0 + r) * lda + k];
                for (; r < MR; r++)           d[r] = 0.0f;
            }
        }
    }
}

static void pack_B_strip(const float * __restrict__ B, int ldb,
                         float * __restrict__ packed,
                         int K_blk, int N_blk, int NR,
                         int t_start, int t_end)
{
    for (int t = t_start; t < t_end; t++) {
        int col0   = t * NR;
        int n_curr = (col0 + NR <= N_blk) ? NR : (N_blk - col0);
        float *dst = packed + (size_t)t * K_blk * NR;

        if (n_curr == NR && (NR % 8) == 0) {
            for (int k = 0; k < K_blk; k++) {
                float *d        = dst + k * NR;
                const float *s  = B + k * ldb + col0;
                for (int c = 0; c < NR; c += 8)
                    _mm256_store_ps(d + c, _mm256_loadu_ps(s + c));
            }
        } else {
            for (int k = 0; k < K_blk; k++) {
                float *d        = dst + k * NR;
                const float *s  = B + k * ldb + col0;
                int c;
                for (c = 0; c < n_curr; c++) d[c] = s[c];
                for (; c < NR; c++)           d[c] = 0.0f;
            }
        }
    }
}

/* ======================================================================
 * Inner macro-kernel
 * ====================================================================== */
static void macro_kernel(const float * __restrict__ pA_panel,
                         const float * __restrict__ pB_strip,
                         float * __restrict__ C,
                         int M_blk, int K_blk, int N_curr,
                         int MR, int NR,
                         float alpha, float beta_k, int ldc,
                         kernel_fn_t kfn, int use_nt_store)
{
    int nr_m = (M_blk + MR - 1) / MR;
    float C_buf[MR * NR] __attribute__((aligned(64)));

    for (int it = 0; it < nr_m; it++) {
        int m_curr   = (it * MR + MR <= M_blk) ? MR : (M_blk - it * MR);
        const float *pA    = pA_panel + (size_t)it * K_blk * MR;
        float       *Ctile = C + it * MR * ldc;
        const float *next_A = (it + 1 < nr_m) ? pA_panel + (size_t)(it + 1) * K_blk * MR : pA;

        if (m_curr == MR && N_curr == NR) {
            kfn(pA, pB_strip, Ctile, K_blk, alpha, beta_k, ldc, next_A, pB_strip, use_nt_store);
        } else {
            if (beta_k != 0.0f) {
                for (int ei = 0; ei < m_curr; ei++) {
                    for (int ej = 0; ej < N_curr; ej++) C_buf[ei * NR + ej] = Ctile[ei * ldc + ej];
                    for (int ej = N_curr; ej < NR; ej++) C_buf[ei * NR + ej] = 0.0f;
                }
                for (int ei = m_curr; ei < MR; ei++) memset(C_buf + ei * NR, 0, NR * sizeof(float));
            } else memset(C_buf, 0, (size_t)MR * NR * sizeof(float));

            kfn(pA, pB_strip, C_buf, K_blk, alpha, beta_k, NR, next_A, pB_strip, 0);

            for (int ei = 0; ei < m_curr; ei++)
                for (int ej = 0; ej < N_curr; ej++) Ctile[ei * ldc + ej] = C_buf[ei * NR + ej];
        }
    }
    if (use_nt_store) _mm_sfence();
}

/* ======================================================================
 * TASK1: 2D task-based parallel SGEMM
 * ====================================================================== */
static void sgemm_task1(int M, int N, int K,
                        float alpha, const float *A, int lda,
                        const float *B, int ldb,
                        float beta,  float *C, int ldc,
                        const sgemm_config_t *cfg)
{
    int KC = cfg->KC, NC = cfg->NC, MC_base = cfg->MC;
    int MR, NR;
    kernel_get_mr_nr(cfg->kernel, &MR, &NR);
    kernel_fn_t kfn = get_kernel_fn(cfg->kernel);
    int nthd = (cfg->nb_threads > 0) ? cfg->nb_threads : omp_get_max_threads();

    int MC = MC_base;
    if (M >= MR * 2 && M / MC < nthd * 2) {
        MC = M / (nthd * 2);
        MC = (MC / MR) * MR;
        if (MC < MR) MC = MR;
    }

    int nr_ic = (M + MC - 1) / MC;
    size_t packed_A_stride = (size_t)((MC+MR-1)/MR) * KC * MR;
    float *packed_A_all = (float *)aligned_alloc(64, (size_t)nr_ic * packed_A_stride * sizeof(float));
    float *packed_B = (float *)aligned_alloc(64, (size_t)((NC+NR-1)/NR) * KC * NR * sizeof(float));

    #pragma omp parallel num_threads(nthd)
    {
        for (int jc = 0; jc < N; jc += NC) {
            int N_blk = (jc + NC <= N) ? NC : (N - jc);
            int nr_jt = (N_blk + NR - 1) / NR;
            for (int pc = 0; pc < K; pc += KC) {
                int K_blk  = (pc + KC <= K) ? KC : (K - pc);
                float beta_k = (pc == 0) ? beta : 1.0f;

                #pragma omp for schedule(static)
                for (int t = 0; t < nr_jt; t++) pack_B_strip(B+pc*ldb+jc, ldb, packed_B, K_blk, N_blk, NR, t, t+1);

                #pragma omp for schedule(static)
                for (int ic_idx = 0; ic_idx < nr_ic; ic_idx++) {
                    int ic = ic_idx * MC;
                    int M_blk = (ic + MC <= M) ? MC : (M - ic);
                    pack_A_panel(A+ic*lda+pc, lda, packed_A_all + (size_t)ic_idx*packed_A_stride, M_blk, K_blk, MR);
                }

                #pragma omp single
                {
                    #pragma omp taskloop collapse(2) grainsize(1)
                    for (int ic_idx = 0; ic_idx < nr_ic; ic_idx++) {
                        for (int jt = 0; jt < nr_jt; jt++) {
                            int ic = ic_idx * MC;
                            int M_blk = (ic + MC <= M) ? MC : (M - ic);
                            int N_curr = (jt * NR + NR <= N_blk) ? NR : (N_blk - jt * NR);
                            macro_kernel(packed_A_all + (size_t)ic_idx*packed_A_stride, packed_B + (size_t)jt*K_blk*NR,
                                         C + ic*ldc + jc + jt*NR, M_blk, K_blk, N_curr, MR, NR, alpha, beta_k, ldc, kfn, cfg->use_nt_store);
                        }
                    }
                }
            }
        }
    }
    free(packed_B); free(packed_A_all);
}

/* ======================================================================
 * Recursive Reduction Helper for 3D Parallelism
 * ====================================================================== */
static void sgemm_reduce_recursive(float *C, int ldc, float **partial_C, 
                                   int tile_idx, int r_start, int r_end, 
                                   int r_total, int ic, int jc, int M_blk, int N_blk, 
                                   int MC, int NC, float beta)
{
    int count = r_end - r_start;
    if (count == 1) {
        #pragma omp task depend(in: partial_C[tile_idx * r_total + r_start]) \
                         firstprivate(ic, jc, M_blk, N_blk, beta, tile_idx, r_start, r_total, MC, NC)
        {
            float *Prow = partial_C[tile_idx * r_total + r_start];
            for (int i = 0; i < M_blk; i++) {
                float *Crow = C + (ic + i) * ldc + jc;
                float *Pr = Prow + i * NC;
                int j = 0;
                if (beta == 0.0f) {
                    for (; j <= N_blk - 8; j += 8) _mm256_storeu_ps(Crow + j, _mm256_loadu_ps(Pr + j));
                    for (; j < N_blk; j++) Crow[j] = Pr[j];
                } else if (beta == 1.0f) {
                    for (; j <= N_blk - 8; j += 8) 
                         _mm256_storeu_ps(Crow + j, _mm256_add_ps(_mm256_loadu_ps(Crow + j), _mm256_loadu_ps(Pr + j)));
                    for (; j < N_blk; j++) Crow[j] += Pr[j];
                } else {
                    __m256 vb = _mm256_set1_ps(beta);
                    for (; j <= N_blk - 8; j += 8)
                         _mm256_storeu_ps(Crow + j, _mm256_fmadd_ps(vb, _mm256_loadu_ps(Crow + j), _mm256_loadu_ps(Pr + j)));
                    for (; j < N_blk; j++) Crow[j] = beta * Crow[j] + Pr[j];
                }
            }
        }
        return;
    }

    int mid = r_start + count / 2;
    sgemm_reduce_recursive(C, ldc, partial_C, tile_idx, r_start, mid, r_total, ic, jc, M_blk, N_blk, MC, NC, beta);
    sgemm_reduce_recursive(C, ldc, partial_C, tile_idx, mid, r_end, r_total, ic, jc, M_blk, N_blk, MC, NC, 1.0f);

    #pragma omp task depend(in: partial_C[tile_idx * r_total + r_start], partial_C[tile_idx * r_total + mid]) \
                     depend(out: partial_C[tile_idx * r_total + r_start]) \
                     firstprivate(tile_idx, r_start, mid, r_total, MC, NC)
    {
        float *pL = partial_C[tile_idx * r_total + r_start];
        float *pR = partial_C[tile_idx * r_total + mid];
        for (size_t next = 0; next < (size_t)MC * NC; next += 8)
            _mm256_storeu_ps(pL + next, _mm256_add_ps(_mm256_loadu_ps(pL + next), _mm256_loadu_ps(pR + next)));
    }
}

/* ======================================================================
 * TASK2: 3D task-based parallel SGEMM
 * ====================================================================== */
static void sgemm_task2(int M, int N, int K,
                        float alpha, const float *A, int lda,
                        const float *B, int ldb,
                        float beta,  float *C, int ldc,
                        const sgemm_config_t *cfg)
{
    int MC = cfg->MC, KC = cfg->KC, NC = cfg->NC;
    int r  = (cfg->r_tasks > 0) ? cfg->r_tasks : 1;
    int MR, NR; kernel_get_mr_nr(cfg->kernel, &MR, &NR);
    kernel_fn_t kfn = get_kernel_fn(cfg->kernel);
    int nthd = (cfg->nb_threads > 0) ? cfg->nb_threads : omp_get_max_threads();
    int nr_ic = (M + MC - 1) / MC, nr_jc = (N + NC - 1) / NC;
    int total_tiles = nr_ic * nr_jc, total_subtasks = total_tiles * r;

    float **partial_C = (float **)malloc((size_t)total_subtasks * sizeof(float *));
    for (int i = 0; i < total_subtasks; i++) partial_C[i] = (float *)aligned_alloc(64, (size_t)MC * NC * sizeof(float));

    size_t pA_buf_sz = (size_t)((MC+MR-1)/MR) * KC * MR * sizeof(float);
    size_t pB_buf_sz = (size_t)((NC+NR-1)/NR) * KC * NR * sizeof(float);
    float **thr_pA = (float **)malloc((size_t)nthd * sizeof(float *));
    float **thr_pB = (float **)malloc((size_t)nthd * sizeof(float *));
    for (int t = 0; t < nthd; t++) { thr_pA[t] = (float *)aligned_alloc(64, pA_buf_sz); thr_pB[t] = (float *)aligned_alloc(64, pB_buf_sz); }

    #pragma omp parallel num_threads(nthd)
    {
        #pragma omp single
        {
            for (int jci = 0; jci < nr_jc; jci++) {
                int jc = jci * NC, N_blk = (jc + NC <= N) ? NC : (N - jc), nr_jt = (N_blk + NR - 1) / NR;
                for (int ici = 0; ici < nr_ic; ici++) {
                    int ic = ici * MC, M_blk = (ic + MC <= M) ? MC : (M - ic), tile_idx = ici * nr_jc + jci;
                    for (int kr = 0; kr < r; kr++) {
                        int total_k_slices = (K + KC - 1) / KC, slices_per_r = (total_k_slices + r - 1) / r;
                        int pc_start = kr * slices_per_r * KC, pc_end = (pc_start + slices_per_r * KC > K) ? K : pc_start + slices_per_r * KC;
                        #pragma omp task depend(out: partial_C[tile_idx * r + kr]) \
                                         firstprivate(ic, jc, M_blk, N_blk, pc_start, pc_end, nr_jt, tile_idx, kr) shared(thr_pA, thr_pB, partial_C)
                        {
                            int tid = omp_get_thread_num(); float *pA = thr_pA[tid], *pB = thr_pB[tid], *pCp = partial_C[tile_idx * r + kr];
                            memset(pCp, 0, (size_t)MC * NC * sizeof(float));
                            if (pc_start < K) {
                                for (int pc = pc_start; pc < pc_end; pc += KC) {
                                    int K_blk = (pc + KC <= K) ? KC : (K - pc);
                                    pack_A_panel(A + ic * lda + pc, lda, pA, M_blk, K_blk, MR);
                                    pack_B_strip(B + pc * ldb + jc, ldb, pB, K_blk, N_blk, NR, 0, nr_jt);
                                    for (int jt = 0; jt < nr_jt; jt++) macro_kernel(pA, pB + (size_t)jt * K_blk * NR, pCp + jt * NR, M_blk, K_blk, (jt * NR + NR <= N_blk) ? NR : (N_blk - jt * NR), MR, NR, alpha, 1.0f, NC, kfn, 0);
                                }
                            }
                        }
                    }
                    sgemm_reduce_recursive(C, ldc, partial_C, tile_idx, 0, r, r, ic, jc, M_blk, N_blk, MC, NC, beta);
                }
            }
        }
    }
    for (int t = 0; t < nthd; t++) { free(thr_pA[t]); free(thr_pB[t]); }
    free(thr_pA); free(thr_pB);
    for (int i = 0; i < total_subtasks; i++) free(partial_C[i]);
    free(partial_C);
}

void sgemm_ref(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * s + beta * C[i * ldc + j];
        }
}

void sgemm_small(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, const sgemm_config_t *cfg)
{
    int MR, NR; kernel_get_mr_nr(cfg->kernel, &MR, &NR); kernel_fn_t kfn = get_kernel_fn(cfg->kernel);
    float *pA = (float *)aligned_alloc(64, (size_t)((M+MR-1)/MR)*K*MR*sizeof(float));
    float *pB = (float *)aligned_alloc(64, (size_t)((N+NR-1)/NR)*K*NR*sizeof(float));
    pack_A_panel(A, lda, pA, M, K, MR); pack_B_strip(B, ldb, pB, K, N, NR, 0, (N+NR-1)/NR);
    for (int jt = 0; jt < (N + NR - 1) / NR; jt++) {
        macro_kernel(pA, pB + (size_t)jt * K * NR, C + jt * NR, M, K, (jt * NR + NR <= N) ? NR : (N - jt * NR), MR, NR, alpha, beta, ldc, kfn, 0);
    }
    free(pA); free(pB);
}

void sgemm_ex(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, const sgemm_config_t *cfg)
{
    if ((long long)M*N*K < (128LL*128*128)) { sgemm_small(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg); return; }
    if (cfg->parallel_mode == PARALLEL_3D) sgemm_task2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg);
    else sgemm_task1(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg);
}

void sgemm(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    sgemm_config_t cfg = SGEMM_DEFAULT_CONFIG; sgemm_ex(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, &cfg);
}
