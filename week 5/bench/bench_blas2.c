/*
 * bench_blas2.c — Week 5: BLAS 2 performance vs OpenBLAS
 *
 * Methodology (mirrors bench_blas1.c):
 *   - Adaptive timing: repeat until stddev < 2% (min 5 runs)
 *   - Metric: GB/s (memory bandwidth) — BLAS2 is memory-bound
 *   - Bandwidth formula: bytes read + written per call
 *   - Thread counts: OMP_NUM_THREADS env (default 1)
 *   - Sizes: n = 64, 128, 256, 512, 1024, 2048, 4096
 *   - Comparison: our AVX2+OMP vs OpenBLAS CBLAS
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include "../include/blas2.h"
#include "../include/blas3.h"

/* -----------------------------------------------------------------------
 * Timing
 * --------------------------------------------------------------------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct { double med; double std; int runs; } timing_t;

typedef void (*bench_fn_t)(void *arg);

static timing_t measure(bench_fn_t fn, void *arg) {
    double times[64];
    int n = 5;
    for (;;) {
        for (int i = 0; i < n; i++) {
            double t0 = now_sec();
            fn(arg);
            times[i] = now_sec() - t0;
        }
        /* median */
        for (int i = 0; i < n - 1; i++)
            for (int j = i+1; j < n; j++)
                if (times[j] < times[i]) { double t = times[i]; times[i] = times[j]; times[j] = t; }
        double med = times[n/2];
        double sum = 0; for (int i = 0; i < n; i++) sum += times[i];
        double mean = sum / n;
        double var = 0; for (int i = 0; i < n; i++) var += (times[i]-mean)*(times[i]-mean);
        double std = sqrt(var/n);
        if (std/mean < 0.02 || n >= 50) return (timing_t){med, std, n};
        n += 2; if (n > 50) n = 50;
    }
}

/* -----------------------------------------------------------------------
 * Per-kernel bench helpers (carry all args in a struct)
 * --------------------------------------------------------------------- */
typedef struct {
    int n, m, lda, incx, incy;
    float alpha, beta;
    char uplo, trans, diag;
    float *A, *x, *y;
} args_t;

/* sgemv */
static void run_sgemv_our(void *p) {
    args_t *a = p;
    sgemv(a->trans, a->m, a->n, a->alpha, a->A, a->lda, a->x, a->incx,
          a->beta, a->y, a->incy);
}
static void run_sgemv_ob(void *p) {
    args_t *a = p;
    CBLAS_TRANSPOSE ct = (a->trans=='N') ? CblasNoTrans : CblasTrans;
    cblas_sgemv(CblasRowMajor, ct, a->m, a->n, a->alpha, a->A, a->lda,
                a->x, a->incx, a->beta, a->y, a->incy);
}

/* sger */
static void run_sger_our(void *p) {
    args_t *a = p; sger(a->m, a->n, a->alpha, a->x, 1, a->y, 1, a->A, a->lda);
}
static void run_sger_ob(void *p) {
    args_t *a = p;
    cblas_sger(CblasRowMajor, a->m, a->n, a->alpha, a->x, 1, a->y, 1, a->A, a->lda);
}

/* ssymv */
static void run_ssymv_our(void *p) {
    args_t *a = p;
    ssymv(a->uplo, a->n, a->alpha, a->A, a->lda, a->x, 1, a->beta, a->y, 1);
}
static void run_ssymv_ob(void *p) {
    args_t *a = p;
    CBLAS_UPLO cu = (a->uplo=='U') ? CblasUpper : CblasLower;
    cblas_ssymv(CblasRowMajor, cu, a->n, a->alpha, a->A, a->lda,
                a->x, 1, a->beta, a->y, 1);
}

/* ssyr */
static void run_ssyr_our(void *p) {
    args_t *a = p; ssyr(a->uplo, a->n, a->alpha, a->x, 1, a->A, a->lda);
}
static void run_ssyr_ob(void *p) {
    args_t *a = p;
    CBLAS_UPLO cu = (a->uplo=='U') ? CblasUpper : CblasLower;
    cblas_ssyr(CblasRowMajor, cu, a->n, a->alpha, a->x, 1, a->A, a->lda);
}

/* ssyr2 */
static void run_ssyr2_our(void *p) {
    args_t *a = p; ssyr2(a->uplo, a->n, a->alpha, a->x, 1, a->y, 1, a->A, a->lda);
}
static void run_ssyr2_ob(void *p) {
    args_t *a = p;
    CBLAS_UPLO cu = (a->uplo=='U') ? CblasUpper : CblasLower;
    cblas_ssyr2(CblasRowMajor, cu, a->n, a->alpha, a->x, 1, a->y, 1, a->A, a->lda);
}

/* -----------------------------------------------------------------------
 * Memory bandwidth for BLAS 2 operations (bytes)
 *   sgemv N  : read A(m×n) + x(n) + y(m), write y(m)
 *   sger     : read A(m×n) + x(m) + y(n), write A(m×n)
 *   ssymv    : read A(n×n/2) + x(n) + y(n), write y(n)  [triangular half]
 *   ssyr     : read A(n×n/2) + x(n), write A(n×n/2)
 *   ssyr2    : read A(n×n/2) + x(n) + y(n), write A(n×n/2)
 * --------------------------------------------------------------------- */
static double bw_sgemv(int m, int n) { return (double)(m*n + m + n + m) * 4; }
static double bw_sger (int m, int n) { return (double)(2*m*n + m + n) * 4; }
static double bw_ssymv(int n)        { return (double)((size_t)n*n/2 + 3*n) * 4; }
static double bw_ssyr (int n)        { return (double)((size_t)n*n/2*2 + n) * 4; }
static double bw_ssyr2(int n)        { return (double)((size_t)n*n/2*2 + 2*n) * 4; }

/* -----------------------------------------------------------------------
 * Print helpers
 * --------------------------------------------------------------------- */
static void print_header(const char *name) {
    printf("\n=== %-10s ===\n", name);
    printf("%-12s %8s %8s %8s %8s\n",
           "Size", "Ours GB/s", "OB GB/s", "Ratio", "Runs");
    printf("%-12s %8s %8s %8s %8s\n",
           "------------", "--------", "--------", "--------", "--------");
}
static void print_row(int sz, double our_t, double ob_t,
                      double bytes, int runs_our, int runs_ob) {
    double our_bw = bytes / our_t / 1e9;
    double ob_bw  = bytes / ob_t  / 1e9;
    printf("n=%-9d %8.2f %8.2f %8.2fx (%d/%d runs)\n",
           sz, our_bw, ob_bw, our_bw / ob_bw, runs_our, runs_ob);
}
static void print_row2(int m, int n, double our_t, double ob_t,
                       double bytes, int runs_our, int runs_ob) {
    double our_bw = bytes / our_t / 1e9;
    double ob_bw  = bytes / ob_t  / 1e9;
    printf("%dx%-7d %8.2f %8.2f %8.2fx (%d/%d runs)\n",
           m, n, our_bw, ob_bw, our_bw / ob_bw, runs_our, runs_ob);
}

/* -----------------------------------------------------------------------
 * Benchmark suites
 * --------------------------------------------------------------------- */
static void bench_sgemv(void) {
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int ns = (int)(sizeof(sizes)/sizeof(sizes[0]));

    /* --- NoTrans --- */
    printf("\n--- sgemv NoTrans ---\n");
    printf("%-12s %8s %8s %8s\n", "m×n", "Ours GB/s", "OB GB/s", "Ratio");
    for (int si = 0; si < ns; si++) {
        int n = sizes[si], m = n;
        args_t a = {.m=m,.n=n,.lda=n,.incx=1,.incy=1,
                    .alpha=1.f,.beta=0.f,.trans='N'};
        a.A = malloc((size_t)m * n * 4);
        a.x = malloc((size_t)n * 4);
        a.y = malloc((size_t)m * 4);
        for (int i=0;i<m*n;i++) a.A[i]=(float)rand()/RAND_MAX;
        for (int i=0;i<n;i++) a.x[i]=(float)rand()/RAND_MAX;
        memset(a.y, 0, m * 4);
        timing_t to = measure(run_sgemv_our, &a);
        float *y_save = a.y; a.y = malloc((size_t)m*4); memset(a.y,0,m*4);
        timing_t tb = measure(run_sgemv_ob,  &a);
        free(a.A); free(a.x); free(y_save); free(a.y);
        print_row2(m, n, to.med, tb.med, bw_sgemv(m,n), to.runs, tb.runs);
    }

    /* --- Trans --- */
    printf("\n--- sgemv Trans ---\n");
    printf("%-12s %8s %8s %8s\n", "m×n", "Ours GB/s", "OB GB/s", "Ratio");
    for (int si = 0; si < ns; si++) {
        int n = sizes[si], m = n;
        args_t a = {.m=m,.n=n,.lda=n,.incx=1,.incy=1,
                    .alpha=1.f,.beta=0.f,.trans='T'};
        a.A = malloc((size_t)m * n * 4);
        a.x = malloc((size_t)m * 4);
        a.y = malloc((size_t)n * 4);
        for (int i=0;i<m*n;i++) a.A[i]=(float)rand()/RAND_MAX;
        for (int i=0;i<m;i++) a.x[i]=(float)rand()/RAND_MAX;
        memset(a.y, 0, n * 4);
        timing_t to = measure(run_sgemv_our, &a);
        float *y_save = a.y; a.y = malloc((size_t)n*4); memset(a.y,0,n*4);
        timing_t tb = measure(run_sgemv_ob,  &a);
        free(a.A); free(a.x); free(y_save); free(a.y);
        print_row2(m, n, to.med, tb.med, bw_sgemv(m,n), to.runs, tb.runs);
    }
}

static void bench_sger(void) {
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    int ns = (int)(sizeof(sizes)/sizeof(sizes[0]));
    print_header("sger");
    for (int si = 0; si < ns; si++) {
        int n = sizes[si], m = n;
        args_t a = {.m=m,.n=n,.lda=n,.alpha=1.f};
        a.A = malloc((size_t)m * n * 4);
        a.x = malloc((size_t)m * 4);
        a.y = malloc((size_t)n * 4);
        for (int i=0;i<m*n;i++) a.A[i]=(float)rand()/RAND_MAX;
        for (int i=0;i<m;i++) a.x[i]=(float)rand()/RAND_MAX;
        for (int i=0;i<n;i++) a.y[i]=(float)rand()/RAND_MAX;
        timing_t to = measure(run_sger_our, &a);
        timing_t tb = measure(run_sger_ob,  &a);
        free(a.A); free(a.x); free(a.y);
        print_row(n, to.med, tb.med, bw_sger(m,n), to.runs, tb.runs);
    }
}

static void bench_ssymv(void) {
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int ns = (int)(sizeof(sizes)/sizeof(sizes[0]));
    print_header("ssymv");
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        args_t a = {.n=n,.lda=n,.alpha=1.f,.beta=0.f,.uplo='L'};
        a.A = malloc((size_t)n * n * 4);
        a.x = malloc((size_t)n * 4);
        a.y = malloc((size_t)n * 4);
        for (int i=0;i<n*n;i++) a.A[i]=(float)rand()/RAND_MAX;
        for (int i=0;i<n;i++) a.A[i*n+i]+=(float)n;  /* diagonal dominant */
        for (int i=0;i<n;i++) a.x[i]=(float)rand()/RAND_MAX;
        memset(a.y, 0, n * 4);
        timing_t to = measure(run_ssymv_our, &a);
        float *y_save = a.y; a.y = malloc((size_t)n*4); memset(a.y,0,n*4);
        timing_t tb = measure(run_ssymv_ob,  &a);
        free(a.A); free(a.x); free(y_save); free(a.y);
        print_row(n, to.med, tb.med, bw_ssymv(n), to.runs, tb.runs);
    }
}

static void bench_ssyr(void) {
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int ns = (int)(sizeof(sizes)/sizeof(sizes[0]));
    print_header("ssyr");
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        args_t a = {.n=n,.lda=n,.alpha=1.f,.uplo='L'};
        a.A = malloc((size_t)n * n * 4);
        a.x = malloc((size_t)n * 4);
        for (int i=0;i<n*n;i++) a.A[i]=(float)rand()/RAND_MAX;
        for (int i=0;i<n;i++) a.x[i]=(float)rand()/RAND_MAX;
        timing_t to = measure(run_ssyr_our, &a);
        timing_t tb = measure(run_ssyr_ob,  &a);
        free(a.A); free(a.x);
        print_row(n, to.med, tb.med, bw_ssyr(n), to.runs, tb.runs);
    }
}

static void bench_ssyr2(void) {
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int ns = (int)(sizeof(sizes)/sizeof(sizes[0]));
    print_header("ssyr2");
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        args_t a = {.n=n,.lda=n,.alpha=1.f,.uplo='L'};
        a.A = malloc((size_t)n * n * 4);
        a.x = malloc((size_t)n * 4);
        a.y = malloc((size_t)n * 4);
        for (int i=0;i<n*n;i++) a.A[i]=(float)rand()/RAND_MAX;
        for (int i=0;i<n;i++) { a.x[i]=(float)rand()/RAND_MAX; a.y[i]=(float)rand()/RAND_MAX; }
        timing_t to = measure(run_ssyr2_our, &a);
        timing_t tb = measure(run_ssyr2_ob,  &a);
        free(a.A); free(a.x); free(a.y);
        print_row(n, to.med, tb.med, bw_ssyr2(n), to.runs, tb.runs);
    }
}

/* -----------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------- */
int main(void) {
    int nthd = 1;
    char *e = getenv("OMP_NUM_THREADS");
    if (e) nthd = atoi(e);

    printf("=== BLAS 2 Benchmark vs OpenBLAS ===\n");
    printf("Threads: %d\n", nthd);
    printf("Metric : GB/s (effective memory bandwidth)\n");
    printf("Timing : adaptive (stop when stddev < 2%%)\n\n");

    bench_sgemv();
    bench_sger();
    bench_ssymv();
    bench_ssyr();
    bench_ssyr2();

    return 0;
}
