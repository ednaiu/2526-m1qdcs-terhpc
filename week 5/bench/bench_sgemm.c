/*
 * bench_sgemm.c  —  Week 2: SGEMM benchmark binary
 *
 * Two modes:
 *
 * 1. Single-config mode (used by Python autotuning):
 *    Accepts CLI flags, outputs one JSON object.
 *    Implements adaptive-N: starts with --min-runs, increases until
 *    stddev/median < --target-stddev percent, capped at --max-runs.
 *
 *    Example:
 *      ./bin/bench_sgemm --M 1024 --kernel 6x16 --threads 4 \
 *                        --MC 120 --KC 512 --NC 4096          \
 *                        --min-runs 3 --max-runs 15 --target-stddev 2.0
 *
 * 2. Table mode (default, no --M flag):
 *    Prints a human-readable table comparing all kernels vs OpenBLAS.
 *
 * JSON output fields (single-config mode):
 *   runs, median_s, mean_s, min_s, max_s, stddev_s, stddev_pct,
 *   gflops_median, gflops_mean, gflops_min, gflops_max, all_times[]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "../include/sgemm.h"

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

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_d(double *a, int n)
{
    double tmp[64];
    if (n > 64) n = 64;
    memcpy(tmp, a, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double);
    return (n % 2 == 0) ? (tmp[n/2-1] + tmp[n/2]) / 2.0 : tmp[n/2];
}

static double stddev_d(double *a, int n, double mean)
{
    double s = 0;
    for (int i = 0; i < n; i++) s += (a[i]-mean)*(a[i]-mean);
    return sqrt(s / n);
}

static double gflops(int M, int N, int K, double t)
{
    return (t > 0) ? 2.0 * M * N * K / t / 1e9 : 0.0;
}

/* -----------------------------------------------------------------------
 * Kernel type / parallel mode parsing
 * ----------------------------------------------------------------------- */
static kernel_type_t parse_kernel(const char *s)
{
    if (!strcmp(s, "8x8"))      return KERNEL_8x8;
    if (!strcmp(s, "4x24"))     return KERNEL_4x24;
    if (!strcmp(s, "6x16_asm")) return KERNEL_6x16_ASM;
    return KERNEL_6x16;  /* default */
}

static parallel_mode_t parse_parallel(const char *s)
{
    if (!strcmp(s, "3D"))      return PARALLEL_3D;
    if (!strcmp(s, "DYNAMIC")) return PARALLEL_DYNAMIC;
    return PARALLEL_2D;
}

static sched_mode_t parse_sched(const char *s)
{
    if (!strcmp(s, "task")) return SCHED_TASK;
    return SCHED_LOOP;
}

/* -----------------------------------------------------------------------
 * Single-config mode: adaptive timing, JSON output
 * ----------------------------------------------------------------------- */
static void bench_single(int M, int N, int K,
                         const sgemm_config_t *cfg,
                         int min_runs, int max_runs,
                         double target_cv_pct)
{
    float *A = alloc_mat(M, K);
    float *B = alloc_mat(K, N);
    float *C = alloc_mat(M, N);
    rand_fill(A, M*K);
    rand_fill(B, K*N);
    rand_fill(C, M*N);

    /* warm-up */
    sgemm_ex(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N, cfg);

    double times[64] = {0};
    int n = 0;

    /* Adaptive-N loop */
    for (n = 0; n < max_runs; n++) {
        double t0 = omp_get_wtime();
        sgemm_ex(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N, cfg);
        times[n] = omp_get_wtime() - t0;

        if (n + 1 >= min_runs) {
            double mean = 0;
            for (int i = 0; i <= n; i++) mean += times[i];
            mean /= (n + 1);
            double sd = stddev_d(times, n+1, mean);
            double cv = (mean > 0) ? sd / mean * 100.0 : 100.0;
            if (cv < target_cv_pct) { n++; break; }
        }
    }

    /* Compute stats */
    double mean = 0, mn = times[0], mx = times[0];
    for (int i = 0; i < n; i++) {
        mean += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    mean /= n;
    double med = median_d(times, n);
    double sd  = stddev_d(times, n, mean);
    double cv  = (mean > 0) ? sd / mean * 100.0 : 0.0;

    /* JSON output */
    printf("{\n");
    printf("  \"runs\": %d,\n", n);
    printf("  \"median_s\": %.9f,\n", med);
    printf("  \"mean_s\":   %.9f,\n", mean);
    printf("  \"min_s\":    %.9f,\n", mn);
    printf("  \"max_s\":    %.9f,\n", mx);
    printf("  \"stddev_s\": %.9f,\n", sd);
    printf("  \"stddev_pct\": %.4f,\n", cv);
    printf("  \"gflops_median\": %.4f,\n", gflops(M, N, K, med));
    printf("  \"gflops_mean\":   %.4f,\n", gflops(M, N, K, mean));
    printf("  \"gflops_min\":    %.4f,\n", gflops(M, N, K, mn));
    printf("  \"gflops_max\":    %.4f,\n", gflops(M, N, K, mx));
    printf("  \"all_times\": [");
    for (int i = 0; i < n; i++) printf("%s%.9f", i?",":"", times[i]);
    printf("]\n}\n");

    free(A); free(B); free(C);
}

/* -----------------------------------------------------------------------
 * Table mode: compare all kernels vs OpenBLAS
 * ----------------------------------------------------------------------- */
#ifdef WITH_OPENBLAS
#include <cblas.h>
static void bench_table(int nthd)
{
    int MC = 120, KC = 512, NC = 4096;
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int ns = (int)(sizeof sizes / sizeof sizes[0]);
    int N_TRIALS = 7;

    struct { const char *name; sgemm_config_t cfg; } v[] = {
        {"TASK1 [LOOP]", {MC,KC,NC,nthd,KERNEL_6x16,    PARALLEL_2D, SCHED_LOOP, 1, 0}},
        {"TASK1 [TASK]", {MC,KC,NC,nthd,KERNEL_6x16,    PARALLEL_2D, SCHED_TASK, 1, 0}},
        {"TASK2 [LOOP]", {MC,KC,NC,nthd,KERNEL_6x16,    PARALLEL_3D, SCHED_LOOP, nthd, 0}},
        {"TASK2 [TASK]", {MC,KC,NC,nthd,KERNEL_6x16,    PARALLEL_3D, SCHED_TASK, nthd, 0}},
        {"ASM   [LOOP]", {MC,KC,NC,nthd,KERNEL_6x16_ASM,PARALLEL_2D, SCHED_LOOP, 1, 0}},
    };
    int nv = (int)(sizeof v / sizeof v[0]);

    printf("=== SGEMM Parallel Scheduling Comparison (%d threads, N=%d) ===\n\n",
           nthd, N_TRIALS);
    printf("%-6s  %12s", "Size", "OpenBLAS");
    for (int i=0;i<nv;i++) printf("  %12s", v[i].name);
    printf("\n%-6s  %12s","------","(GF/s)");
    for (int i=0;i<nv;i++) printf("  %11s","GF/s(%%)");
    printf("\n");

    for (int si=0; si<ns; si++) {
        int SZ = sizes[si];
        float *A=alloc_mat(SZ,SZ), *B=alloc_mat(SZ,SZ);
        float *C=alloc_mat(SZ,SZ), *C2=alloc_mat(SZ,SZ);
        rand_fill(A,SZ*SZ); rand_fill(B,SZ*SZ); rand_fill(C,SZ*SZ);

        double ob_times[7];
        for (int t=0;t<N_TRIALS;t++) {
            double t0=omp_get_wtime();
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                        SZ,SZ,SZ,1.f,A,SZ,B,SZ,0.f,C2,SZ);
            ob_times[t]=omp_get_wtime()-t0;
        }
        double ob_med = median_d(ob_times,N_TRIALS);
        double ob_gf  = gflops(SZ,SZ,SZ,ob_med);
        printf("%-6d  %12.2f",SZ,ob_gf);

        for (int vi=0;vi<nv;vi++) {
            double ts[7];
            for (int t=0;t<N_TRIALS;t++) {
                double t0=omp_get_wtime();
                sgemm_ex(SZ,SZ,SZ,1.f,A,SZ,B,SZ,0.f,C,SZ,&v[vi].cfg);
                ts[t]=omp_get_wtime()-t0;
            }
            double med=median_d(ts,N_TRIALS);
            double gf=gflops(SZ,SZ,SZ,med);
            printf("  %8.2f(%3.0f%%)",gf,gf/ob_gf*100.0);
        }
        printf("\n");
        free(A);free(B);free(C);free(C2);
    }
}
#endif

/* -----------------------------------------------------------------------
 * Argument parsing helpers
 * ----------------------------------------------------------------------- */
static int get_int_arg(int argc, char **argv, const char *flag, int def)
{
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], flag)) return atoi(argv[i+1]);
    return def;
}

static double get_dbl_arg(int argc, char **argv, const char *flag, double def)
{
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], flag)) return atof(argv[i+1]);
    return def;
}

static const char *get_str_arg(int argc, char **argv, const char *flag, const char *def)
{
    for (int i = 1; i < argc - 1; i++)
        if (!strcmp(argv[i], flag)) return argv[i+1];
    return def;
}

static int has_flag(int argc, char **argv, const char *flag)
{
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

/* -----------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    srand(42);

    /* Single-config mode: triggered when --M is present */
    if (has_flag(argc, argv, "--M")) {
        int M  = get_int_arg(argc, argv, "--M",  1024);
        int N  = get_int_arg(argc, argv, "--N",  M);
        int K  = get_int_arg(argc, argv, "--K",  M);
        int MC = get_int_arg(argc, argv, "--MC", 120);
        int KC = get_int_arg(argc, argv, "--KC", 512);
        int NC = get_int_arg(argc, argv, "--NC", 4096);
        int threads  = get_int_arg(argc, argv, "--threads",   1);
        int r_tasks  = get_int_arg(argc, argv, "--r-tasks",   1);
        int min_runs = get_int_arg(argc, argv, "--min-runs",  3);
        int max_runs = get_int_arg(argc, argv, "--max-runs",  15);
        double target_cv = get_dbl_arg(argc, argv, "--target-stddev", 2.0);

        const char *kname = get_str_arg(argc, argv, "--kernel",   "6x16");
        const char *pname = get_str_arg(argc, argv, "--parallel", "2D");
        const char *sname = get_str_arg(argc, argv, "--sched",    "loop");

        sgemm_config_t cfg = {
            .MC = MC, .KC = KC, .NC = NC,
            .nb_threads = threads,
            .kernel = parse_kernel(kname),
            .parallel_mode = parse_parallel(pname),
            .sched_mode = parse_sched(sname),
            .r_tasks = r_tasks,
        };

        bench_single(M, N, K, &cfg, min_runs, max_runs, target_cv);
        return 0;
    }

    /* Table mode (default) */
#ifdef WITH_OPENBLAS
    int nthd = omp_get_max_threads();
    bench_table(nthd);
#else
    fprintf(stderr,
        "Table mode requires OpenBLAS.\n"
        "Compile with: make compare\n"
        "Or use single-config mode: ./bin/bench_sgemm --M 1024 ...\n");
    return 1;
#endif
    return 0;
}
