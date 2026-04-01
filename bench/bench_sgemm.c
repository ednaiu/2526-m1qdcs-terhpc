#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include "../include/sgemm.h"

#define TRIALS 3

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

int main(void)
{
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== SGEMM Benchmark (Single Core) ===\n");
    printf("%6s %15s %15s %10s\n", "Size", "OpenBLAS(GF/s)", "Our SGEMM(GF/s)", "Ratio");
    printf("%6s %15s %15s %10s\n", "----", "--------------", "---------------", "-----");

    for (int i = 0; i < num_sizes; i++) {
        int N = sizes[i];
        float *A = alloc_matrix(N, N);
        float *B = alloc_matrix(N, N);
        float *C1 = alloc_matrix(N, N);
        float *C2 = alloc_matrix(N, N);

        rand_fill(A, N * N);
        rand_fill(B, N * N);
        rand_fill(C1, N * N);
        rand_fill(C2, N * N);

        double openblas_time = 1e9;
        for (int t = 0; t < TRIALS; t++) {
            double start = omp_get_wtime();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0f, A, N, B, N, 0.0f, C1, N);
            double end = omp_get_wtime();
            if (end - start < openblas_time) openblas_time = end - start;
        }

        double our_time = 1e9;
        for (int t = 0; t < TRIALS; t++) {
            double start = omp_get_wtime();
            sgemm(N, N, N, 1.0f, A, N, B, N, 0.0f, C2, N);
            double end = omp_get_wtime();
            if (end - start < our_time) our_time = end - start;
        }

        double gflops = 2.0 * N * N * N / 1e9;
        double openblas_gflops = gflops / openblas_time;
        double our_gflops = gflops / our_time;

        printf("%6d %15.2f %15.2f %9.2fx\n", N, openblas_gflops, our_gflops, our_gflops / openblas_gflops);

        free(A); free(B); free(C1); free(C2);
    }
    return 0;
}
