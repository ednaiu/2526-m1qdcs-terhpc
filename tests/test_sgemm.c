/*
 * test_sgemm.c  —  Correctness tests for our optimised SGEMM
 *
 * Compares output of sgemm() against sgemm_ref() (naive triple loop)
 * for various matrix sizes and alpha/beta combinations.
 *
 * Compile (Makefile does this):
 *   gcc -O3 -march=native -mavx2 -mfma -fopenmp \
 *       -I../include tests/test_sgemm.c src/sgemm.c -lm -o test_sgemm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../include/sgemm.h"

/* Maximum allowed absolute difference (SP + FMA rounding) */
#define TOL 1e-3f

/* ------------------------------------------------------------------ */
static float *alloc_matrix(int rows, int cols)
{
    float *m = (float *)aligned_alloc(32, (size_t)rows * cols * sizeof(float));
    if (!m) { fprintf(stderr, "OOM\n"); exit(1); }
    return m;
}

static void rand_fill(float *m, int n)
{
    for (int i = 0; i < n; i++)
        m[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

static void copy_matrix(const float *src, float *dst, int n)
{
    memcpy(dst, src, n * sizeof(float));
}

/* Returns maximum absolute difference between two MxN matrices */
static float max_diff(const float *A, const float *B, int n)
{
    float d = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(A[i] - B[i]);
        if (diff > d) d = diff;
    }
    return d;
}

/* ------------------------------------------------------------------ */
static int run_test(int M, int N, int K, float alpha, float beta)
{
    int ok = 1;

    float *A    = alloc_matrix(M, K);
    float *B    = alloc_matrix(K, N);
    float *C_ref = alloc_matrix(M, N);
    float *C_opt = alloc_matrix(M, N);

    rand_fill(A, M * K);
    rand_fill(B, K * N);
    rand_fill(C_ref, M * N);
    copy_matrix(C_ref, C_opt, M * N);   /* both start from the same C */

    /* Reference */
    sgemm_ref(M, N, K, alpha, A, K, B, N, beta, C_ref, N);

    /* Optimised */
    sgemm(M, N, K, alpha, A, K, B, N, beta, C_opt, N);

    float diff = max_diff(C_ref, C_opt, M * N);
    if (diff > TOL) {
        printf("  FAIL: M=%d N=%d K=%d alpha=%.2f beta=%.2f  max_diff=%.6f\n",
               M, N, K, alpha, beta, diff);
        ok = 0;
    } else {
        printf("  PASS: M=%d N=%d K=%d alpha=%.2f beta=%.2f  max_diff=%.2e\n",
               M, N, K, alpha, beta, diff);
    }

    free(A); free(B); free(C_ref); free(C_opt);
    return ok;
}

/* ------------------------------------------------------------------ */
int main(void)
{
    int all_pass = 1;
    srand(42);

    printf("=== SGEMM Correctness Tests ===\n\n");

    /* Test 1: square matrices, various sizes */
    printf("--- Square matrices (alpha=1, beta=0) ---\n");
    all_pass &= run_test(  8,   8,   8, 1.0f, 0.0f);
    all_pass &= run_test( 16,  16,  16, 1.0f, 0.0f);
    all_pass &= run_test( 32,  32,  32, 1.0f, 0.0f);
    all_pass &= run_test( 64,  64,  64, 1.0f, 0.0f);
    all_pass &= run_test(128, 128, 128, 1.0f, 0.0f);
    all_pass &= run_test(256, 256, 256, 1.0f, 0.0f);
    all_pass &= run_test(512, 512, 512, 1.0f, 0.0f);

    /* Test 2: non-multiples of MR/NR (edge case handling) */
    printf("\n--- Non-aligned sizes ---\n");
    all_pass &= run_test( 13,  17,  19, 1.0f, 0.0f);
    all_pass &= run_test( 33,  33,  33, 1.0f, 0.0f);
    all_pass &= run_test(100, 200, 300, 1.0f, 0.0f);

    /* Test 3: alpha/beta scaling */
    printf("\n--- Alpha/Beta scaling ---\n");
    all_pass &= run_test(64, 64, 64,  2.0f,  0.0f);
    all_pass &= run_test(64, 64, 64,  1.0f,  1.0f);
    all_pass &= run_test(64, 64, 64,  3.14f, 0.5f);
    all_pass &= run_test(64, 64, 64,  0.0f,  2.0f);  /* pure beta scaling */

    /* Test 4: rectangular (tall and skinny, short and fat) */
    printf("\n--- Rectangular matrices ---\n");
    all_pass &= run_test(512,  32, 256, 1.0f, 0.0f);  /* tall × skinny */
    all_pass &= run_test( 32, 512, 256, 1.0f, 0.0f);  /* short × fat   */
    all_pass &= run_test(256, 256,   8, 1.0f, 0.0f);  /* thin K        */

    printf("\n%s\n", all_pass ? "=== ALL TESTS PASSED ===" : "=== SOME TESTS FAILED ===");
    return all_pass ? 0 : 1;
}
