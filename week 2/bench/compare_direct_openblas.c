#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <omp.h>
#include <cblas.h>

#include "../include/sgemm.h"

/* Provided by OpenBLAS shared library. */
extern void openblas_set_num_threads(int num_threads);

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double median(double *arr, int n)
{
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    if (n % 2 == 1) return arr[n / 2];
    return 0.5 * (arr[n / 2 - 1] + arr[n / 2]);
}

static void pin_to_first_n_cpus(int n)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < n; ++i) {
        CPU_SET(i, &set);
    }
    (void)sched_setaffinity(0, sizeof(set), &set);
}

static void fill_random(float *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        x[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }
}

static double run_openblas_once(int M, int N, int K, const float *A, const float *B, float *C)
{
    double t1 = now_sec();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K,
                B, N,
                0.0f, C, N);
    double t2 = now_sec();
    return t2 - t1;
}

static double run_week2_once(int M, int N, int K,
                             const float *A, const float *B, float *C,
                             int threads)
{
    sgemm_config_t cfg = (sgemm_config_t)SGEMM_DEFAULT_CONFIG;
    cfg.kernel = KERNEL_6x16;
    cfg.parallel_mode = PARALLEL_2D;
    cfg.nb_threads = threads;

    double t1 = now_sec();
    sgemm_ex(M, N, K,
             1.0f, A, K,
             B, N,
             0.0f, C, N,
             &cfg);
    double t2 = now_sec();
    return t2 - t1;
}

static void benchmark_case(int M, int N, int K, int threads, int runs, FILE *csv)
{
    size_t asz = (size_t)M * (size_t)K;
    size_t bsz = (size_t)K * (size_t)N;
    size_t csz = (size_t)M * (size_t)N;

    float *A = (float *)malloc(asz * sizeof(float));
    float *B = (float *)malloc(bsz * sizeof(float));
    float *C_ob = (float *)malloc(csz * sizeof(float));
    float *C_w2 = (float *)malloc(csz * sizeof(float));

    if (!A || !B || !C_ob || !C_w2) {
        fprintf(stderr, "allocation failed for %dx%dx%d\n", M, N, K);
        free(A); free(B); free(C_ob); free(C_w2);
        return;
    }

    fill_random(A, asz);
    fill_random(B, bsz);
    memset(C_ob, 0, csz * sizeof(float));
    memset(C_w2, 0, csz * sizeof(float));

    pin_to_first_n_cpus(threads);
    openblas_set_num_threads(threads);
    omp_set_num_threads(threads);

    /* Warmups */
    run_openblas_once(M, N, K, A, B, C_ob);
    run_week2_once(M, N, K, A, B, C_w2, threads);

    double *ob_gflops = (double *)malloc((size_t)runs * sizeof(double));
    double *w2_gflops = (double *)malloc((size_t)runs * sizeof(double));
    if (!ob_gflops || !w2_gflops) {
        fprintf(stderr, "timing allocation failed\n");
        free(ob_gflops); free(w2_gflops);
        free(A); free(B); free(C_ob); free(C_w2);
        return;
    }

    const double flops = 2.0 * (double)M * (double)N * (double)K;

    for (int i = 0; i < runs; ++i) {
        memset(C_ob, 0, csz * sizeof(float));
        memset(C_w2, 0, csz * sizeof(float));

        double t_ob = run_openblas_once(M, N, K, A, B, C_ob);
        double t_w2 = run_week2_once(M, N, K, A, B, C_w2, threads);

        ob_gflops[i] = flops / t_ob / 1e9;
        w2_gflops[i] = flops / t_w2 / 1e9;
    }

    double ob_med = median(ob_gflops, runs);
    double w2_med = median(w2_gflops, runs);
    double ratio = (ob_med > 0.0) ? (w2_med / ob_med) : 0.0;

    printf("%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%s\n",
           M, N, K, threads, runs, ob_med, w2_med, ratio,
           (ratio >= 1.0 ? "Week2" : "OpenBLAS"));

    if (csv) {
        fprintf(csv, "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%s\n",
                M, N, K, threads, runs, ob_med, w2_med, ratio,
                (ratio >= 1.0 ? "Week2" : "OpenBLAS"));
    }

    free(ob_gflops);
    free(w2_gflops);
    free(A); free(B); free(C_ob); free(C_w2);
}

int main(int argc, char **argv)
{
    int runs = 5;
    if (argc >= 2) {
        runs = atoi(argv[1]);
        if (runs <= 0) runs = 5;
    }

    int sizes[] = {256, 512, 1024};
    int threads_list[] = {8, 16};

    srand(12345);

    FILE *csv = fopen("results/direct_openblas_vs_week2.csv", "w");
    if (!csv) {
        perror("fopen results/direct_openblas_vs_week2.csv");
    }

    printf("M,N,K,threads,runs,openblas_median_gflops,week2_median_gflops,ratio_week2_over_openblas,winner\n");
    if (csv) {
        fprintf(csv, "M,N,K,threads,runs,openblas_median_gflops,week2_median_gflops,ratio_week2_over_openblas,winner\n");
    }

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
        for (size_t ti = 0; ti < sizeof(threads_list) / sizeof(threads_list[0]); ++ti) {
            benchmark_case(sizes[si], sizes[si], sizes[si], threads_list[ti], runs, csv);
        }
    }

    if (csv) fclose(csv);
    return 0;
}
