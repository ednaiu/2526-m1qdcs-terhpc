/*
 * bench_blas1.c — Week 4: Benchmark harness for BLAS 1 kernels
 *
 * For each kernel and each vector size, measures performance of:
 *   - Scalar baseline
 *   - AVX2 single-thread
 *   - AVX2 + prefetch + OpenMP (our 'best')
 *   - OpenBLAS reference
 *
 * Methodology:
 *   - Adaptive timing: run until std-dev < 2%, minimum 5 runs
 *   - Reports median GB/s (memory-bound) or GFLOP/s (compute)
 *   - Outputs formatted table + CSV
 *
 * Build: make bench-blas1
 * Run:   OMP_NUM_THREADS=8 ./bin/bench_blas1
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include "../include/blas1.h"
#include <cblas.h>

/* -----------------------------------------------------------------------
 * Timer
 * --------------------------------------------------------------------- */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* -----------------------------------------------------------------------
 * Adaptive timing
 * Returns median time in seconds over [min 5, adaptive] runs.
 * --------------------------------------------------------------------- */
#define MAX_RUNS 64

static double adaptive_median(void (*fn)(void *), void *arg,
                              double *p_gbs, double bytes_per_call)
{
    double times[MAX_RUNS];
    int nruns = 5;                /* start with 5 */

    for (;;) {
        for (int r = 0; r < nruns; r++) {
            double t0 = now_sec();
            fn(arg);
            times[r] = now_sec() - t0;
        }

        /* compute mean and stddev */
        double sum = 0.0;
        for (int r = 0; r < nruns; r++) sum += times[r];
        double mean = sum / nruns;
        double var = 0.0;
        for (int r = 0; r < nruns; r++) var += (times[r]-mean)*(times[r]-mean);
        double stddev = sqrt(var / nruns);

        if (stddev / mean < 0.02 || nruns >= MAX_RUNS) break;
        if (nruns < MAX_RUNS) nruns = (nruns + 2 < MAX_RUNS) ? nruns + 2 : MAX_RUNS;
    }

    /* sort for median */
    for (int i = 0; i < nruns-1; i++)
        for (int j = i+1; j < nruns; j++)
            if (times[j] < times[i]) { double t=times[i]; times[i]=times[j]; times[j]=t; }

    double med = times[nruns/2];
    if (p_gbs) *p_gbs = bytes_per_call / med * 1e-9;
    return med;
}

/* -----------------------------------------------------------------------
 * Benchmark infrastructure
 * --------------------------------------------------------------------- */
typedef struct {
    const char *kernel;
    int     n;
    double  scalar_gbs;
    double  avx2_gbs;
    double  par_gbs;
    double  openblas_gbs;
} row_t;

static float *make_vec(int n) {
    float *v = (float *)aligned_alloc(64, (n+8)*sizeof(float));
    for (int i = 0; i < n; i++) v[i] = (float)(i % 100) * 0.01f + 0.5f;
    return v;
}

/* ------------------------------------------------------------------
 * Generic bench wrappers (one struct per kernel variant/size)
 * ------------------------------------------------------------------ */

/* sscal */
typedef struct { int n; float alpha; float *x; } sscal_arg;
static void w_sscal_scalar(void *a) { sscal_arg *p = a; sscal_scalar(p->n, p->alpha, p->x, 1); }
static void w_sscal_avx2  (void *a) { sscal_arg *p = a; sscal_avx2  (p->n, p->alpha, p->x, 1); }
static void w_sscal_par   (void *a) { sscal_arg *p = a; sscal       (p->n, p->alpha, p->x, 1); }
static void w_sscal_ob    (void *a) { sscal_arg *p = a; cblas_sscal  (p->n, p->alpha, p->x, 1); }

/* scopy */
typedef struct { int n; const float *x; float *y; } scopy_arg;
static void w_scopy_scalar(void *a) { scopy_arg *p = a; scopy_scalar(p->n, p->x, 1, p->y, 1); }
static void w_scopy_avx2  (void *a) { scopy_arg *p = a; scopy_avx2  (p->n, p->x, 1, p->y, 1); }
static void w_scopy_par   (void *a) { scopy_arg *p = a; scopy       (p->n, p->x, 1, p->y, 1); }
static void w_scopy_ob    (void *a) { scopy_arg *p = a; cblas_scopy  (p->n, p->x, 1, p->y, 1); }

/* sswap */
typedef struct { int n; float *x; float *y; } sswap_arg;
static void w_sswap_scalar(void *a) { sswap_arg *p = a; sswap_scalar(p->n, p->x, 1, p->y, 1); }
static void w_sswap_avx2  (void *a) { sswap_arg *p = a; sswap_avx2  (p->n, p->x, 1, p->y, 1); }
static void w_sswap_par   (void *a) { sswap_arg *p = a; sswap       (p->n, p->x, 1, p->y, 1); }
static void w_sswap_ob    (void *a) { sswap_arg *p = a; cblas_sswap  (p->n, p->x, 1, p->y, 1); }

/* saxpy */
typedef struct { int n; float alpha; const float *x; float *y; } saxpy_arg;
static void w_saxpy_scalar(void *a) { saxpy_arg *p = a; saxpy_scalar(p->n, p->alpha, p->x, 1, p->y, 1); }
static void w_saxpy_avx2  (void *a) { saxpy_arg *p = a; saxpy_avx2  (p->n, p->alpha, p->x, 1, p->y, 1); }
static void w_saxpy_par   (void *a) { saxpy_arg *p = a; saxpy       (p->n, p->alpha, p->x, 1, p->y, 1); }
static void w_saxpy_ob    (void *a) { saxpy_arg *p = a; cblas_saxpy  (p->n, p->alpha, p->x, 1, p->y, 1); }

/* sdot */
typedef struct { int n; const float *x; const float *y; volatile float r; } sdot_arg;
static void w_sdot_scalar(void *a) { sdot_arg *p = a; p->r = sdot_scalar(p->n, p->x, 1, p->y, 1); }
static void w_sdot_avx2  (void *a) { sdot_arg *p = a; p->r = sdot_avx2  (p->n, p->x, 1, p->y, 1); }
static void w_sdot_par   (void *a) { sdot_arg *p = a; p->r = sdot       (p->n, p->x, 1, p->y, 1); }
static void w_sdot_ob    (void *a) { sdot_arg *p = a; p->r = cblas_sdot  (p->n, p->x, 1, p->y, 1); }

/* snrm2 */
typedef struct { int n; const float *x; volatile float r; } snrm2_arg;
static void w_snrm2_scalar(void *a) { snrm2_arg *p = a; p->r = snrm2_scalar(p->n, p->x, 1); }
static void w_snrm2_avx2  (void *a) { snrm2_arg *p = a; p->r = snrm2_avx2  (p->n, p->x, 1); }
static void w_snrm2_par   (void *a) { snrm2_arg *p = a; p->r = snrm2       (p->n, p->x, 1); }
static void w_snrm2_ob    (void *a) { snrm2_arg *p = a; p->r = cblas_snrm2  (p->n, p->x, 1); }

/* sasum */
typedef struct { int n; const float *x; volatile float r; } sasum_arg;
static void w_sasum_scalar(void *a) { sasum_arg *p = a; p->r = sasum_scalar(p->n, p->x, 1); }
static void w_sasum_avx2  (void *a) { sasum_arg *p = a; p->r = sasum_avx2  (p->n, p->x, 1); }
static void w_sasum_par   (void *a) { sasum_arg *p = a; p->r = sasum       (p->n, p->x, 1); }
static void w_sasum_ob    (void *a) { sasum_arg *p = a; p->r = cblas_sasum  (p->n, p->x, 1); }

/* isamax */
typedef struct { int n; const float *x; volatile int r; } isamax_arg;
static void w_isamax_scalar(void *a) { isamax_arg *p = a; p->r = isamax_scalar(p->n, p->x, 1); }
static void w_isamax_avx2  (void *a) { isamax_arg *p = a; p->r = isamax_avx2  (p->n, p->x, 1); }
static void w_isamax_par   (void *a) { isamax_arg *p = a; p->r = isamax       (p->n, p->x, 1); }
static void w_isamax_ob    (void *a) { isamax_arg *p = a; p->r = (int)cblas_isamax(p->n, p->x, 1); }

/* srot */
typedef struct { int n; float *x; float *y; float c; float s; } srot_arg;
static float *g_xsave_rot = NULL;
static float *g_ysave_rot = NULL;
static int    g_nsave_rot = 0;
static void rot_restore(srot_arg *p) {
    if (g_nsave_rot != p->n) {
        free(g_xsave_rot); free(g_ysave_rot);
        g_xsave_rot = make_vec(p->n);
        g_ysave_rot = make_vec(p->n);
        g_nsave_rot = p->n;
    }
    memcpy(g_xsave_rot, p->x, p->n * sizeof(float));
    memcpy(g_ysave_rot, p->y, p->n * sizeof(float));
}
static void w_srot_scalar(void *a) { srot_arg *p = a; srot_scalar(p->n, p->x, 1, p->y, 1, p->c, p->s); rot_restore(p); }
static void w_srot_avx2  (void *a) { srot_arg *p = a; srot_avx2  (p->n, p->x, 1, p->y, 1, p->c, p->s); rot_restore(p); }
static void w_srot_par   (void *a) { srot_arg *p = a; srot       (p->n, p->x, 1, p->y, 1, p->c, p->s); rot_restore(p); }
static void w_srot_ob    (void *a) { srot_arg *p = a; cblas_srot  (p->n, p->x, 1, p->y, 1, p->c, p->s); rot_restore(p); }

/* -----------------------------------------------------------------------
 * Print table header
 * --------------------------------------------------------------------- */
static void print_header(void)
{
    printf("\n%-8s %8s | %9s %9s %9s %9s | %8s\n",
           "Kernel", "N", "Scalar", "AVX2", "AVX2+OMP", "OpenBLAS", "% of OB");
    printf("%-8s %8s | %9s %9s %9s %9s | %8s\n",
           "--------", "--------",
           "---------", "---------", "---------", "---------", "--------");
}

static void print_row(const char *ker, int n,
                      double sc, double av, double pr, double ob)
{
    double pct = ob > 0 ? pr / ob * 100.0 : 0.0;
    printf("%-8s %8d | %9.2f %9.2f %9.2f %9.2f | %7.1f%%\n",
           ker, n, sc, av, pr, ob, pct);
}

/* Unit: GB/s for memory-bound kernels (most BLAS 1),
        GFLOP/s for sdot/saxpy/snrm2/srot  */
static const char *unit_for(const char *ker)
{
    if (strcmp(ker,"sdot")==0 || strcmp(ker,"saxpy")==0 ||
        strcmp(ker,"snrm2")==0 || strcmp(ker,"srot")==0)
        return "GFLOP/s";
    return "GB/s";
}

/* bytes read/written per call (for GB/s) or FLOP/call (for GFLOP/s) */
static double ops_for(const char *ker, int n)
{
    if (strcmp(ker,"sscal")==0)  return (double)n * 2 * sizeof(float); /* 1R+1W */
    if (strcmp(ker,"scopy")==0)  return (double)n * 2 * sizeof(float); /* 1R+1W */
    if (strcmp(ker,"sswap")==0)  return (double)n * 4 * sizeof(float); /* 2R+2W */
    if (strcmp(ker,"saxpy")==0)  return (double)n * 2.0;               /* 2 FLOP/elem */
    if (strcmp(ker,"sdot")==0)   return (double)n * 2.0;               /* 2 FLOP/elem */
    if (strcmp(ker,"snrm2")==0)  return (double)n * 2.0;               /* 2 FLOP/elem */
    if (strcmp(ker,"sasum")==0)  return (double)n * sizeof(float);     /* 1R */
    if (strcmp(ker,"isamax")==0) return (double)n * sizeof(float);     /* 1R */
    if (strcmp(ker,"srot")==0)   return (double)n * 6.0;               /* 6 FLOP/elem */
    return (double)n * sizeof(float);
}

/* -----------------------------------------------------------------------
 * Main
 * --------------------------------------------------------------------- */
int main(void)
{
    int sizes[] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    int nsizes  = sizeof(sizes) / sizeof(sizes[0]);

    printf("Week 4 BLAS 1 Benchmark\n");
    printf("=======================\n");
    printf("Unit: GB/s (memory-bound) or GFLOP/s (compute); median of adaptive runs\n");

    /* CSV header */
    printf("\nCSV:\nkernel,n,scalar,avx2,avx2_omp,openblas,pct_of_ob,unit\n");

    /* ---- sscal ---- */
    {
        print_header();
        printf("[sscal — %s]\n", unit_for("sscal"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n);
            sscal_arg a = {n, 2.5f, x};
            double o, sc, av, pr;
            adaptive_median(w_sscal_scalar, &a, &sc,  ops_for("sscal",n));
            adaptive_median(w_sscal_avx2,   &a, &av,  ops_for("sscal",n));
            adaptive_median(w_sscal_par,     &a, &pr,  ops_for("sscal",n));
            adaptive_median(w_sscal_ob,      &a, &o,   ops_for("sscal",n));
            print_row("sscal", n, sc, av, pr, o);
            printf("sscal,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("sscal"));
            free(x);
        }
    }

    /* ---- scopy ---- */
    {
        print_header();
        printf("[scopy — %s]\n", unit_for("scopy"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n), *y = make_vec(n);
            scopy_arg a = {n, x, y};
            double o, sc, av, pr;
            adaptive_median(w_scopy_scalar, &a, &sc, ops_for("scopy",n));
            adaptive_median(w_scopy_avx2,   &a, &av, ops_for("scopy",n));
            adaptive_median(w_scopy_par,     &a, &pr, ops_for("scopy",n));
            adaptive_median(w_scopy_ob,      &a, &o,  ops_for("scopy",n));
            print_row("scopy", n, sc, av, pr, o);
            printf("scopy,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("scopy"));
            free(x); free(y);
        }
    }

    /* ---- sswap ---- */
    {
        print_header();
        printf("[sswap — %s]\n", unit_for("sswap"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n), *y = make_vec(n);
            sswap_arg a = {n, x, y};
            double o, sc, av, pr;
            adaptive_median(w_sswap_scalar, &a, &sc, ops_for("sswap",n));
            adaptive_median(w_sswap_avx2,   &a, &av, ops_for("sswap",n));
            adaptive_median(w_sswap_par,     &a, &pr, ops_for("sswap",n));
            adaptive_median(w_sswap_ob,      &a, &o,  ops_for("sswap",n));
            print_row("sswap", n, sc, av, pr, o);
            printf("sswap,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("sswap"));
            free(x); free(y);
        }
    }

    /* ---- saxpy ---- */
    {
        print_header();
        printf("[saxpy — %s]\n", unit_for("saxpy"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n), *y = make_vec(n);
            saxpy_arg a = {n, 2.5f, x, y};
            double o, sc, av, pr;
            adaptive_median(w_saxpy_scalar, &a, &sc, ops_for("saxpy",n));
            adaptive_median(w_saxpy_avx2,   &a, &av, ops_for("saxpy",n));
            adaptive_median(w_saxpy_par,     &a, &pr, ops_for("saxpy",n));
            adaptive_median(w_saxpy_ob,      &a, &o,  ops_for("saxpy",n));
            print_row("saxpy", n, sc, av, pr, o);
            printf("saxpy,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("saxpy"));
            free(x); free(y);
        }
    }

    /* ---- sdot ---- */
    {
        print_header();
        printf("[sdot — %s]\n", unit_for("sdot"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n), *y = make_vec(n);
            sdot_arg a = {n, x, y, 0.0f};
            double o, sc, av, pr;
            adaptive_median(w_sdot_scalar, &a, &sc, ops_for("sdot",n));
            adaptive_median(w_sdot_avx2,   &a, &av, ops_for("sdot",n));
            adaptive_median(w_sdot_par,     &a, &pr, ops_for("sdot",n));
            adaptive_median(w_sdot_ob,      &a, &o,  ops_for("sdot",n));
            print_row("sdot", n, sc, av, pr, o);
            printf("sdot,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("sdot"));
            free(x); free(y);
        }
    }

    /* ---- snrm2 ---- */
    {
        print_header();
        printf("[snrm2 — %s]\n", unit_for("snrm2"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n);
            snrm2_arg a = {n, x, 0.0f};
            double o, sc, av, pr;
            adaptive_median(w_snrm2_scalar, &a, &sc, ops_for("snrm2",n));
            adaptive_median(w_snrm2_avx2,   &a, &av, ops_for("snrm2",n));
            adaptive_median(w_snrm2_par,     &a, &pr, ops_for("snrm2",n));
            adaptive_median(w_snrm2_ob,      &a, &o,  ops_for("snrm2",n));
            print_row("snrm2", n, sc, av, pr, o);
            printf("snrm2,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("snrm2"));
            free(x);
        }
    }

    /* ---- sasum ---- */
    {
        print_header();
        printf("[sasum — %s]\n", unit_for("sasum"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n);
            sasum_arg a = {n, x, 0.0f};
            double o, sc, av, pr;
            adaptive_median(w_sasum_scalar, &a, &sc, ops_for("sasum",n));
            adaptive_median(w_sasum_avx2,   &a, &av, ops_for("sasum",n));
            adaptive_median(w_sasum_par,     &a, &pr, ops_for("sasum",n));
            adaptive_median(w_sasum_ob,      &a, &o,  ops_for("sasum",n));
            print_row("sasum", n, sc, av, pr, o);
            printf("sasum,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("sasum"));
            free(x);
        }
    }

    /* ---- isamax ---- */
    {
        print_header();
        printf("[isamax — %s]\n", unit_for("isamax"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n);
            isamax_arg a = {n, x, 0};
            double o, sc, av, pr;
            adaptive_median(w_isamax_scalar, &a, &sc, ops_for("isamax",n));
            adaptive_median(w_isamax_avx2,   &a, &av, ops_for("isamax",n));
            adaptive_median(w_isamax_par,     &a, &pr, ops_for("isamax",n));
            adaptive_median(w_isamax_ob,      &a, &o,  ops_for("isamax",n));
            print_row("isamax", n, sc, av, pr, o);
            printf("isamax,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("isamax"));
            free(x);
        }
    }

    /* ---- srot ---- */
    {
        float c = (float)(1.0/sqrt(2.0)), s = c;
        print_header();
        printf("[srot — %s]\n", unit_for("srot"));
        for (int si = 0; si < nsizes; si++) {
            int n = sizes[si];
            float *x = make_vec(n), *y = make_vec(n);
            srot_arg a = {n, x, y, c, s};
            rot_restore(&a);   /* init saved copies */
            double o, sc, av, pr;
            adaptive_median(w_srot_scalar, &a, &sc, ops_for("srot",n));
            adaptive_median(w_srot_avx2,   &a, &av, ops_for("srot",n));
            adaptive_median(w_srot_par,     &a, &pr, ops_for("srot",n));
            adaptive_median(w_srot_ob,      &a, &o,  ops_for("srot",n));
            print_row("srot", n, sc, av, pr, o);
            printf("srot,%d,%.3f,%.3f,%.3f,%.3f,%.1f,%s\n",
                   n, sc, av, pr, o, o>0?pr/o*100:0, unit_for("srot"));
            free(x); free(y);
        }
    }

    free(g_xsave_rot); free(g_ysave_rot);

    printf("\nBenchmark complete.\n");
    return 0;
}
