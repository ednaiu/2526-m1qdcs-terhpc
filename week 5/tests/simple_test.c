/*
 * simple_test.c — Lightweight SGEMM correctness test
 *
 * Tests core SGEMM functionality without complex memory management
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/sgemm.h"

#define TOL 1e-3f

static int test_basic_square(int N)
{
    printf("Testing %dx%dx%d square matrix...", N, N, N);
    fflush(stdout);
    
    float *A = (float *)malloc((size_t)N * N * sizeof(float));
    float *B = (float *)malloc((size_t)N * N * sizeof(float));
    float *C_ref = (float *)malloc((size_t)N * N * sizeof(float));
    float *C_test = (float *)malloc((size_t)N * N * sizeof(float));
    
    if (!A || !B || !C_ref || !C_test) {
        printf(" SKIP (allocation failed)\n");
        free(A); free(B); free(C_ref); free(C_test);
        return 1;
    }
    
    /* Fill with simple values */
    for (int i = 0; i < N * N; i++) {
        A[i] = (i % 10) * 0.1f;
        B[i] = ((i + 5) % 10) * 0.1f;
        C_ref[i] = C_test[i] = 0.0f;
    }
    
    /* Reference computation */
    sgemm_ref(N, N, N, 1.0f, A, N, B, N, 0.0f, C_ref, N);
    
    /* Test computation */
    sgemm_config_t cfg = {
        .MC = 120, .KC = 512, .NC = 4096,
        .nb_threads = 1,
        .kernel = KERNEL_6x16,
        .parallel_mode = PARALLEL_2D,
        .r_tasks = 1
    };
    sgemm_ex(N, N, N, 1.0f, A, N, B, N, 0.0f, C_test, N, &cfg);
    
    /* Compare */
    int errors = 0;
    for (int i = 0; i < N * N && errors < 5; i++) {
        float diff = fabsf(C_ref[i] - C_test[i]);
        if (diff > TOL * (fabsf(C_ref[i]) + 1.0f)) {
            if (errors == 0)
                printf("\n  Error at [%d]: ref=%.6f test=%.6f diff=%.6e\n",
                       i, C_ref[i], C_test[i], diff);
            errors++;
        }
    }
    
    if (errors == 0)
        printf(" PASS\n");
    else
        printf("  FAIL (%d errors)\n", errors);
    
    free(A); free(B); free(C_ref); free(C_test);
    return errors > 0 ? 1 : 0;
}

int main(void)
{
    printf("===== Simple SGEMM Correctness Test =====\n\n");
    
    int failed = 0;
    int sizes[] = {8, 16, 32, 64, 128, 256};
    
    for (int i = 0; i < 6; i++) {
        failed += test_basic_square(sizes[i]);
    }
    
    printf("\n===== %s =====\n", failed == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return failed > 0 ? 1 : 0;
}
