/*
 * test_sgemm.c  —  Correctness tests for all micro-kernels and configs
 *
 * Tests:
 *   1. Square matrices (8 → 512)
 *   2. Non-square / non-aligned sizes
 *   3. Alpha/Beta combinations
 *   4. Rectangular matrices (tall/wide/thin-K)
 *   5. All three kernel types (8×8, 6×16, 4×24, asm)
 *   6. Both parallelism modes (TASK1, TASK2 with r=2)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/sgemm.h"

#define TOL  1e-3f

/* -----------------------------------------------------------------------
 * Utilities
 * ----------------------------------------------------------------------- */
static float *alloc_mat(int r, int c)
{
    float *m = (float *)aligned_alloc(64, (size_t)r * c * sizeof(float));
    if (!m) { perror("alloc_mat"); exit(1); }
    return m;
}

static void rand_fill(float *m, int n)
{
    for (int i = 0; i < n; i++)
        m[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

static int check(const float *ref, const float *got, int M, int N, int ldc,
                 const char *label)
{
    int fails = 0;
    for (int i = 0; i < M && fails < 5; i++)
        for (int j = 0; j < N && fails < 5; j++) {
            float d = fabsf(ref[i * ldc + j] - got[i * ldc + j]);
            if (d > TOL * (fabsf(ref[i * ldc + j]) + 1.0f)) {
                if (!fails)
                    printf("  FAIL  %s  C[%d,%d]: ref=%.6f got=%.6f\n",
                           label, i, j, ref[i*ldc+j], got[i*ldc+j]);
                fails++;
            }
        }
    return fails == 0;
}

static int run_test(int M, int N, int K, float alpha, float beta,
                    const sgemm_config_t *cfg, const char *label)
{
    int ldA = K, ldB = N, ldC = N;
    float *A  = alloc_mat(M, K);
    float *B  = alloc_mat(K, N);
    float *C0 = alloc_mat(M, N);  /* reference */
    float *C1 = alloc_mat(M, N);  /* our sgemm  */

    rand_fill(A,  M * K);
    rand_fill(B,  K * N);
    rand_fill(C0, M * N);
    memcpy(C1, C0, (size_t)M * N * sizeof(float));

    sgemm_ref(M, N, K, alpha, A, ldA, B, ldB, beta, C0, ldC);
    sgemm_ex (M, N, K, alpha, A, ldA, B, ldB, beta, C1, ldC, cfg);

    int ok = check(C0, C1, M, N, ldC, label);
    if (ok)  printf("  PASS  %s\n", label);

    free(A); free(B); free(C0); free(C1);
    return ok;
}

/* -----------------------------------------------------------------------
 * Test suites
 * ----------------------------------------------------------------------- */
static int test_suite_sizes(const sgemm_config_t *cfg, const char *name)
{
    int ok = 1;
    int sq[] = {8, 16, 32, 64, 128, 256, 512};
    char label[128];

    for (int i = 0; i < (int)(sizeof sq / sizeof sq[0]); i++) {
        int N = sq[i];
        snprintf(label, sizeof label, "[%s] square %dx%d", name, N, N);
        ok &= run_test(N, N, N, 1.0f, 0.0f, cfg, label);
    }

    /* Non-aligned */
    int cases[][3] = { {13,17,19}, {33,33,33}, {97,67,53}, {100,200,300} };
    for (int i = 0; i < 4; i++) {
        int M = cases[i][0], N = cases[i][1], K = cases[i][2];
        snprintf(label, sizeof label, "[%s] %dx%dx%d", name, M, N, K);
        ok &= run_test(M, N, K, 1.0f, 0.0f, cfg, label);
    }

    /* Alpha/Beta */
    snprintf(label, sizeof label, "[%s] alpha=2 beta=0.5", name);
    ok &= run_test(64, 64, 64, 2.0f, 0.5f, cfg, label);
    snprintf(label, sizeof label, "[%s] alpha=0 beta=1", name);
    ok &= run_test(64, 64, 64, 0.0f, 1.0f, cfg, label);

    /* Rectangular */
    snprintf(label, sizeof label, "[%s] tall 512x32x256", name);
    ok &= run_test(512, 32, 256, 1.0f, 0.0f, cfg, label);
    snprintf(label, sizeof label, "[%s] wide 32x512x256", name);
    ok &= run_test(32, 512, 256, 1.0f, 0.0f, cfg, label);
    snprintf(label, sizeof label, "[%s] thin-K 256x256x8", name);
    ok &= run_test(256, 256, 8, 1.0f, 1.0f, cfg, label);

    return ok;
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(void)
{
    srand(42);
    int total_ok = 1;

    printf("===== SGEMM Week-2 Correctness Tests =====\n\n");

    /* --- Test each kernel type with TASK1 --- */
    struct {
        const char      *name;
        sgemm_config_t   cfg;
    } kernels[] = {
        { "8x8  / TASK1",
          { .MC=128, .KC=512, .NC=512,  .nb_threads=1,
            .kernel=KERNEL_8x8,  .parallel_mode=PARALLEL_2D, .r_tasks=1 } },
        { "6x16 / TASK1",
          { .MC=120, .KC=512, .NC=4096, .nb_threads=1,
            .kernel=KERNEL_6x16, .parallel_mode=PARALLEL_2D, .r_tasks=1 } },
        { "4x24 / TASK1",
          { .MC=120, .KC=512, .NC=4080, .nb_threads=1,
            .kernel=KERNEL_4x24, .parallel_mode=PARALLEL_2D, .r_tasks=1 } },
        { "6x16-asm / TASK1",
          { .MC=120, .KC=512, .NC=4096, .nb_threads=1,
            .kernel=KERNEL_6x16_ASM, .parallel_mode=PARALLEL_2D, .r_tasks=1 } },
        /* TASK2 */
        { "6x16 / TASK2 r=2",
          { .MC=120, .KC=512, .NC=512,  .nb_threads=2,
            .kernel=KERNEL_6x16, .parallel_mode=PARALLEL_3D, .r_tasks=2 } },
        { "6x16 / TASK2 r=4",
          { .MC=120, .KC=512, .NC=512,  .nb_threads=4,
            .kernel=KERNEL_6x16, .parallel_mode=PARALLEL_3D, .r_tasks=4 } },
    };

    for (int i = 0; i < (int)(sizeof kernels / sizeof kernels[0]); i++) {
        printf("-- %s --\n", kernels[i].name);
        total_ok &= test_suite_sizes(&kernels[i].cfg, kernels[i].name);
        printf("\n");
    }

    printf("===== %s =====\n", total_ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return total_ok ? 0 : 1;
}
