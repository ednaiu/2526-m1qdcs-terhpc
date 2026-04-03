/*
 * sgemm.h  —  Week 2: Generic SGEMM with configurable micro-kernels
 *             and task-based parallelism
 *
 * Implements:  C = alpha * A * B + beta * C
 *   A : M x K  (row-major, lda >= K)
 *   B : K x N  (row-major, ldb >= N)
 *   C : M x N  (row-major, ldc >= N)
 *
 * Week 3 additions:
 *   - use_nt_store: non-temporal stores for C (bypass cache on write)
 */

#ifndef SGEMM_H
#define SGEMM_H

/* -----------------------------------------------------------------------
 * Micro-kernel types
 *
 * Register budget (AVX2 = 16 YMM registers, each = 8 floats):
 *
 *   KERNEL_8x8:    8 acc + 1 B-vec + 1 A-bcast = 10 regs
 *                  8 FMAs/k-step = 128 FLOP/step
 *
 *   KERNEL_6x16:   12 acc + 2 B-vecs + 1 A-bcast + 1 spare = 16 regs
 *                  12 FMAs/k-step = 192 FLOP/step  (default, most efficient)
 *
 *   KERNEL_4x24:   12 acc + 3 B-vecs + 1 A-bcast = 16 regs
 *                  12 FMAs/k-step = 192 FLOP/step  (same, better for wide N)
 *
 *   KERNEL_6x16_ASM: same as 6x16 but in inline x86-64 assembly
 * ----------------------------------------------------------------------- */
typedef enum {
    KERNEL_8x8      = 0,
    KERNEL_6x16     = 1,   /* default */
    KERNEL_4x24     = 2,
    KERNEL_6x16_ASM = 3,
} kernel_type_t;

/* -----------------------------------------------------------------------
 * Parallelism modes
 *
 *   PARALLEL_2D (TASK1):
 *     Each OpenMP task computes one (MC x NC) tile of C.
 *     Total tasks = ceil(M/MC) * ceil(N/NC).
 *     Simple, good for medium/large matrices.
 *
 *   PARALLEL_3D (TASK2):
 *     K-dimension split into r_tasks parts.
 *     Each task computes a partial (MC x NC) result into a temp buffer.
 *     A reduction task merges all r partial results into real C.
 *     More parallelism for small matrices with many cores.
 * ----------------------------------------------------------------------- */
typedef enum {
    PARALLEL_2D = 0,   /* TASK1: tiles of C, no K-splitting      */
    PARALLEL_3D = 1,   /* TASK2: 3D with K-replication+reduction  */
} parallel_mode_t;

/* -----------------------------------------------------------------------
 * Run-time configuration
 * ----------------------------------------------------------------------- */
typedef struct {
    /* Cache blocking */
    int MC;          /* rows of A packed into L2   (should be multiple of MR) */
    int KC;          /* depth blocked for L2/L3                               */
    int NC;          /* cols of B packed into L3   (should be multiple of NR) */
    /* Execution */
    int nb_threads;  /* 0 = use OMP_NUM_THREADS env var                       */
    kernel_type_t   kernel;
    parallel_mode_t parallel_mode;
    int r_tasks;     /* PARALLEL_3D only: K-replication factor (>= 1)         */
    int use_nt_store;/* 1 = non-temporal stores for C (stream_ps); 0 = normal */
} sgemm_config_t;

/* Sensible defaults (6x16 kernel, 2D tasks, single thread until bench says otherwise) */
#define SGEMM_DEFAULT_CONFIG { \
    .MC = 120, .KC = 512, .NC = 4096, \
    .nb_threads = 0, \
    .kernel = KERNEL_6x16, \
    .parallel_mode = PARALLEL_2D, \
    .r_tasks = 1, \
    .use_nt_store = 0 \
}

/* -----------------------------------------------------------------------
 * Helper: get MR/NR for a kernel type
 * ----------------------------------------------------------------------- */
static inline void kernel_get_mr_nr(kernel_type_t kt, int *mr, int *nr)
{
    switch (kt) {
        case KERNEL_8x8:      *mr = 8;  *nr = 8;  break;
        case KERNEL_6x16:     /* fall */
        case KERNEL_6x16_ASM: *mr = 6;  *nr = 16; break;
        case KERNEL_4x24:     *mr = 4;  *nr = 24; break;
        default:              *mr = 6;  *nr = 16; break;
    }
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

/* Naive triple-loop reference — for correctness testing only */
void sgemm_ref(int M, int N, int K,
               float alpha, const float *A, int lda,
               const float *B, int ldb,
               float beta,  float *C, int ldc);

/* Default SGEMM (uses SGEMM_DEFAULT_CONFIG) */
void sgemm(int M, int N, int K,
           float alpha, const float *A, int lda,
           const float *B, int ldb,
           float beta,  float *C, int ldc);

/* Extended SGEMM — pass a sgemm_config_t for full control */
void sgemm_ex(int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *B, int ldb,
              float beta,  float *C, int ldc,
              const sgemm_config_t *cfg);

#endif /* SGEMM_H */
