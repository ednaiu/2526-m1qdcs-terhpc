/*
 * sgemm.c  —  Week 2: Generic BLIS-style SGEMM
 *
 * Supports three micro-kernels (8×8, 6×16, 4×24 + asm variant)
 * and two parallelism modes:
 *
 *   PARALLEL_2D  (TASK1): one OpenMP task per (MC×KC) × (NC) tile of C
 *   PARALLEL_3D  (TASK2): K split into r_tasks parts; each (ic,jc,k_r)
 *                         subtask writes a partial result, then a reduction
 *                         task sums the r partial C buffers into real C.
 *
 * All packing functions accept run-time MR/NR so they work for any kernel.
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
                             int, float, float, int);

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
 * Packing helpers  (run-time MR / NR)
 * ====================================================================== */

/*
 * pack_A_panel — copies a M_blk × K_blk submatrix of A into a packed buffer.
 *
 * Output layout:  [ceil(M_blk/MR) strips] × [K_blk steps] × [MR elements]
 * = strip_s, step_k: packed[ s * K_blk * MR + k * MR + r ] = A[(s*MR+r)*lda + k]
 *
 * Edge strips (where M_blk % MR != 0) are zero-padded to a full MR width.
 */
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
                for (int r = 0; r < MR; r++)
                    d[r] = A[(row0 + r) * lda + k];
            }
        } else {
            /* Edge strip: zero-pad to MR */
            for (int k = 0; k < K_blk; k++) {
                float *d = dst + k * MR;
                int r;
                for (r = 0; r < m_curr; r++) d[r] = A[(row0 + r) * lda + k];
                for (; r < MR; r++)           d[r] = 0.0f;
            }
        }
    }
}

/*
 * pack_B_panel — copies a K_blk × N_blk submatrix of B into a packed buffer.
 *
 * Output layout:  [ceil(N_blk/NR) strips] × [K_blk steps] × [NR elements]
 * = strip_t, step_k: packed[ t * K_blk * NR + k * NR + c ] = B[k*ldb + t*NR+c]
 *
 * Uses AVX2 stores when NR is a multiple of 8 and the full strip is present.
 * Edge strips are zero-padded to NR.
 *
 * Can be called inside an #pragma omp for (nowait) for parallel B packing.
 */
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
            /* Fast path: full strip, AVX2 copy */
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
 * Inner macro-kernel: iterates micro-tiles over an (M_blk × K_blk) × N_curr strip
 *
 * N_curr: actual valid columns in this strip (≤ NR).
 * When N_curr < NR (edge strip) or m_curr < MR (edge row), the kernel
 * computes into a scratch C_buf[MR × NR] and scatters only the valid
 * portion back — preventing any out-of-bounds write to the real C matrix.
 * ====================================================================== */
static void macro_kernel(const float * __restrict__ pA_panel,   /* [strips_M × K_blk × MR] */
                         const float * __restrict__ pB_strip,   /* [K_blk × NR]             */
                         float * __restrict__ C,                 /* top-left of this tile    */
                         int M_blk, int K_blk, int N_curr,
                         int MR, int NR,
                         float alpha, float beta_k, int ldc,
                         kernel_fn_t kfn)
{
    int nr_m = (M_blk + MR - 1) / MR;
    float C_buf[MR * NR] __attribute__((aligned(64)));

    for (int it = 0; it < nr_m; it++) {
        int m_curr   = (it * MR + MR <= M_blk) ? MR : (M_blk - it * MR);
        const float *pA = pA_panel + (size_t)it * K_blk * MR;
        float       *Ctile = C + it * MR * ldc;

        if (m_curr == MR && N_curr == NR) {
            /* Full tile: call kernel directly into C (no buffer needed) */
            kfn(pA, pB_strip, Ctile, K_blk, alpha, beta_k, ldc);
        } else {
            /* Edge tile (row or column boundary): use scratch buffer
             * so the kernel never writes out-of-bounds into C. */

            /* Gather valid C values; zero-pad the rest.
             * Skip gather when beta_k==0: kernel ignores old C values. */
            if (beta_k != 0.0f) {
                for (int ei = 0; ei < m_curr; ei++) {
                    for (int ej = 0; ej < N_curr; ej++)
                        C_buf[ei * NR + ej] = Ctile[ei * ldc + ej];
                    for (int ej = N_curr; ej < NR; ej++)
                        C_buf[ei * NR + ej] = 0.0f;
                }
                for (int ei = m_curr; ei < MR; ei++)
                    memset(C_buf + ei * NR, 0, NR * sizeof(float));
            } else {
                memset(C_buf, 0, (size_t)MR * NR * sizeof(float));
            }

            kfn(pA, pB_strip, C_buf, K_blk, alpha, beta_k, NR);

            /* Scatter back only valid elements */
            for (int ei = 0; ei < m_curr; ei++)
                for (int ej = 0; ej < N_curr; ej++)
                    Ctile[ei * ldc + ej] = C_buf[ei * NR + ej];
        }
    }
}

/* ======================================================================
 * TASK1: 2D task-based parallel SGEMM
 *
 * Decomposition:
 *   - Outer loops (jc, pc) run single-threaded.
 *   - For each (pc) slab, B is packed in parallel across all threads.
 *   - The (ic, jt) 2D tile space is distributed as OpenMP tasks,
 *     giving dynamic load balancing at the granularity of one (MC×NR) strip.
 * ====================================================================== */
static void sgemm_task1(int M, int N, int K,
                        float alpha, const float *A, int lda,
                        const float *B, int ldb,
                        float beta,  float *C, int ldc,
                        const sgemm_config_t *cfg)
{
    int MC = cfg->MC, KC = cfg->KC, NC = cfg->NC;
    int MR, NR;
    kernel_get_mr_nr(cfg->kernel, &MR, &NR);
    kernel_fn_t kfn = get_kernel_fn(cfg->kernel);

    int nr_ic = (M + MC - 1) / MC;
    int A_strips_max = (MC + MR - 1) / MR;
    size_t packed_A_stride = (size_t)A_strips_max * KC * MR;

    /* Shared packed buffers — allocated once and reused. */
    int B_strips_max = (NC + NR - 1) / NR;
    float *packed_A_all = (float *)aligned_alloc(64,
                            (size_t)nr_ic * packed_A_stride * sizeof(float));
    float *packed_B = (float *)aligned_alloc(64,
                        (size_t)B_strips_max * KC * NR * sizeof(float));
    if (!packed_A_all) { perror("aligned_alloc packed_A_all"); exit(1); }
    if (!packed_B) { perror("aligned_alloc packed_B"); free(packed_A_all); exit(1); }

    #pragma omp parallel num_threads(cfg->nb_threads > 0 ? cfg->nb_threads : omp_get_max_threads())
    {
        for (int jc = 0; jc < N; jc += NC) {
            int N_blk = (jc + NC <= N) ? NC : (N - jc);
            int nr_jt = (N_blk + NR - 1) / NR;

            for (int pc = 0; pc < K; pc += KC) {
                int K_blk  = (pc + KC <= K) ? KC : (K - pc);
                float beta_k = (pc == 0) ? beta : 1.0f;

                /* Pack current B block in parallel once. */
                #pragma omp for schedule(static)
                for (int t = 0; t < nr_jt; t++) {
                    pack_B_strip(B + pc * ldb + jc, ldb, packed_B,
                                 K_blk, N_blk, NR, t, t + 1);
                }

                /* Pack each A panel once. */
                #pragma omp for schedule(static)
                for (int ic_idx = 0; ic_idx < nr_ic; ic_idx++) {
                    int ic    = ic_idx * MC;
                    int M_blk = (ic + MC <= M) ? MC : (M - ic);
                    float *pA_panel = packed_A_all + (size_t)ic_idx * packed_A_stride;
                    pack_A_panel(A + ic * lda + pc, lda, pA_panel, M_blk, K_blk, MR);
                }

                /* Compute all tiles with prepacked A/B. */
                #pragma omp for collapse(2) schedule(dynamic)
                for (int ic_idx = 0; ic_idx < nr_ic; ic_idx++) {
                    for (int jt = 0; jt < nr_jt; jt++) {
                        int ic     = ic_idx * MC;
                        int M_blk  = (ic + MC <= M) ? MC : (M - ic);
                        int N_curr = (jt * NR + NR <= N_blk) ? NR : (N_blk - jt * NR);

                        const float *pA_panel = packed_A_all + (size_t)ic_idx * packed_A_stride;
                        float *pB_strip = packed_B + (size_t)jt * K_blk * NR;

                        macro_kernel(pA_panel, pB_strip,
                                     C + ic * ldc + jc + jt * NR,
                                     M_blk, K_blk, N_curr, MR, NR,
                                     alpha, beta_k, ldc, kfn);
                    }
                }
            }
        }
    }

    free(packed_B);
    free(packed_A_all);
}

/* ======================================================================
 * TASK2: 3D task-based parallel SGEMM with K-replication + reduction
 *
 * Improvements over naive version:
 *   1. Per-thread pA/pB buffers pre-allocated ONCE before the parallel
 *      region — eliminates aligned_alloc/free inside every OMP task.
 *   2. AVX2-vectorized reduction loop (8 floats/cycle instead of scalar).
 *   3. Partial-C zeroed inside the task (cache-warm on the executing thread).
 * ====================================================================== */
static void sgemm_task2(int M, int N, int K,
                        float alpha, const float *A, int lda,
                        const float *B, int ldb,
                        float beta,  float *C, int ldc,
                        const sgemm_config_t *cfg)
{
    int MC = cfg->MC, KC = cfg->KC, NC = cfg->NC;
    int r  = cfg->r_tasks;
    if (r < 1) r = 1;
    int MR, NR;
    kernel_get_mr_nr(cfg->kernel, &MR, &NR);
    kernel_fn_t kfn = get_kernel_fn(cfg->kernel);

    int nthd   = (cfg->nb_threads > 0) ? cfg->nb_threads : omp_get_max_threads();
    int nr_ic  = (M + MC - 1) / MC;
    int nr_jc  = (N + NC - 1) / NC;

    /* ----------------------------------------------------------------
     * Partial-C buffers: one per (tile, kr).  Zeroed inside the task
     * so the cache lines are owned by the executing thread.
     * ---------------------------------------------------------------- */
    size_t partial_c_size  = (size_t)MC * NC * sizeof(float);
    int    total_tiles     = nr_ic * nr_jc;
    int    total_subtasks  = total_tiles * r;

    float **partial_C = (float **)malloc((size_t)total_subtasks * sizeof(float *));
    if (!partial_C) { perror("malloc partial_C"); exit(1); }
    for (int i = 0; i < total_subtasks; i++) {
        partial_C[i] = (float *)aligned_alloc(64, partial_c_size);
        if (!partial_C[i]) { perror("aligned_alloc partial_C"); exit(1); }
    }

    /* ----------------------------------------------------------------
     * Per-thread packing buffers: allocated ONCE, reused by every task
     * that runs on that thread.  No malloc/free inside the task body.
     * ---------------------------------------------------------------- */
    int    A_strips_max = (MC + MR - 1) / MR;
    int    B_strips_max = (NC + NR - 1) / NR;
    size_t pA_buf_sz    = (size_t)A_strips_max * KC * MR * sizeof(float);
    size_t pB_buf_sz    = (size_t)B_strips_max * KC * NR * sizeof(float);

    float **thr_pA = (float **)malloc((size_t)nthd * sizeof(float *));
    float **thr_pB = (float **)malloc((size_t)nthd * sizeof(float *));
    if (!thr_pA || !thr_pB) { perror("malloc thr_pA/pB"); exit(1); }
    for (int t = 0; t < nthd; t++) {
        thr_pA[t] = (float *)aligned_alloc(64, pA_buf_sz);
        thr_pB[t] = (float *)aligned_alloc(64, pB_buf_sz);
        if (!thr_pA[t] || !thr_pB[t]) { perror("aligned_alloc thr buf"); exit(1); }
    }

    #pragma omp parallel num_threads(nthd)
    #pragma omp single
    {
        for (int jc_idx = 0; jc_idx < nr_jc; jc_idx++) {
            int jc    = jc_idx * NC;
            int N_blk = (jc + NC <= N) ? NC : (N - jc);
            int nr_jt = (N_blk + NR - 1) / NR;

            for (int ic_idx = 0; ic_idx < nr_ic; ic_idx++) {
                int ic    = ic_idx * MC;
                int M_blk = (ic + MC <= M) ? MC : (M - ic);
                int tile_idx = ic_idx * nr_jc + jc_idx;

                /* ---- spawn r independent K-slice tasks ---- */
                for (int kr = 0; kr < r; kr++) {
                    float *pC_partial  = partial_C[tile_idx * r + kr];

                    int total_k_slices = (K + KC - 1) / KC;
                    int slices_per_r   = (total_k_slices + r - 1) / r;
                    int pc_start = kr * slices_per_r * KC;
                    int pc_end   = pc_start + slices_per_r * KC;
                    if (pc_end > K) pc_end = K;
                    if (pc_start >= K) {
                        /* no K slices for this replica — zero partial and skip */
                        memset(pC_partial, 0, partial_c_size);
                        continue;
                    }

                    #pragma omp task firstprivate(ic, jc, M_blk, N_blk, \
                                                  pc_start, pc_end,      \
                                                  pC_partial, nr_jt)     \
                                     shared(thr_pA, thr_pB)
                    {
                        /* Use pre-allocated buffer for THIS thread */
                        int    tid = omp_get_thread_num();
                        float *pA  = thr_pA[tid];
                        float *pB  = thr_pB[tid];

                        /* Zero partial-C on the thread that will write it
                         * (keeps the cache lines local) */
                        memset(pC_partial, 0, partial_c_size);

                        for (int pc = pc_start; pc < pc_end; pc += KC) {
                            int K_blk = (pc + KC <= K) ? KC : (K - pc);

                            pack_A_panel(A + ic * lda + pc, lda,
                                         pA, M_blk, K_blk, MR);
                            pack_B_strip(B + pc * ldb + jc, ldb,
                                         pB, K_blk, N_blk, NR,
                                         0, nr_jt);

                            float bk = (pc == pc_start) ? 0.0f : 1.0f;
                            for (int jt = 0; jt < nr_jt; jt++) {
                                int n_curr = (jt * NR + NR <= N_blk)
                                             ? NR : (N_blk - jt * NR);
                                macro_kernel(pA,
                                             pB + (size_t)jt * K_blk * NR,
                                             pC_partial + jt * NR,
                                             M_blk, K_blk, n_curr, MR, NR,
                                             alpha, bk, NC, kfn);
                            }
                        }
                    } /* end task */
                } /* kr */

                /* ---- wait for all r tasks, then AVX2-vectorized reduce ---- */
                #pragma omp taskwait

                __m256 vbeta = _mm256_set1_ps(beta);
                for (int kr = 0; kr < r; kr++) {
                    float *pCp  = partial_C[tile_idx * r + kr];
                    int    first = (kr == 0);

                    for (int i = 0; i < M_blk; i++) {
                        float *Crow = C   + (ic + i) * ldc + jc;
                        float *Prow = pCp + i * NC;
                        int j = 0;

                        if (first) {
                            /* C[i] = beta*C[i] + partial[i] */
                            for (; j <= N_blk - 8; j += 8) {
                                __m256 cv = _mm256_loadu_ps(Crow + j);
                                __m256 pv = _mm256_loadu_ps(Prow + j);
                                _mm256_storeu_ps(Crow + j,
                                    _mm256_fmadd_ps(vbeta, cv, pv));
                            }
                            for (; j < N_blk; j++)
                                Crow[j] = beta * Crow[j] + Prow[j];
                        } else {
                            /* C[i] += partial[i]  (subsequent replicas) */
                            for (; j <= N_blk - 8; j += 8) {
                                __m256 cv = _mm256_loadu_ps(Crow + j);
                                __m256 pv = _mm256_loadu_ps(Prow + j);
                                _mm256_storeu_ps(Crow + j,
                                    _mm256_add_ps(cv, pv));
                            }
                            for (; j < N_blk; j++)
                                Crow[j] += Prow[j];
                        }
                    }
                }

            } /* ic */
        } /* jc */
    } /* omp single / parallel */

    for (int t = 0; t < nthd; t++) { free(thr_pA[t]); free(thr_pB[t]); }
    free(thr_pA); free(thr_pB);
    for (int i = 0; i < total_subtasks; i++) free(partial_C[i]);
    free(partial_C);
}

/* ======================================================================
 * Public API
 * ====================================================================== */

void sgemm_ref(int M, int N, int K,
               float alpha, const float *A, int lda,
               const float *B, int ldb,
               float beta,  float *C, int ldc)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * s + beta * C[i * ldc + j];
        }
}

void sgemm_ex(int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *B, int ldb,
              float beta,  float *C, int ldc,
              const sgemm_config_t *cfg)
{
    if (cfg->parallel_mode == PARALLEL_3D)
        sgemm_task2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg);
    else
        sgemm_task1(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, cfg);
}

void sgemm(int M, int N, int K,
           float alpha, const float *A, int lda,
           const float *B, int ldb,
           float beta,  float *C, int ldc)
{
    sgemm_config_t cfg = SGEMM_DEFAULT_CONFIG;
    sgemm_ex(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, &cfg);
}
