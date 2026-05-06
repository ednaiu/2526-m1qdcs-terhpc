#define _POSIX_C_SOURCE 200809L
/*
 * test_lib.c  —  functional and performance check for libmyblas.
 *
 * Tests the Fortran BLAS interface (trailing-underscore symbols) as called
 * by any standard Fortran or C BLAS user.
 *
 * Compile against our library:
 *   gcc -O2 -fopenmp test_lib.c -L./lib -lmyblas -lm -o test_ours
 *
 * Compile against OpenBLAS (same source, no changes):
 *   gcc -O2 -fopenmp test_lib.c -lopenblas -lm -o test_ob
 *
 * Run both and compare the numbers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- Fortran BLAS declarations ------------------------------------------ */
void  sgemm_(char*, char*, int*, int*, int*, float*, const float*, int*,
             const float*, int*, float*, float*, int*);
void  saxpy_(int*, float*, const float*, int*, float*, int*);
void  sscal_(int*, float*, float*, int*);
float sdot_ (int*, const float*, int*, const float*, int*);
float snrm2_(int*, const float*, int*);
float sasum_(int*, const float*, int*);
void  scopy_(int*, const float*, int*, float*, int*);
void  sgemv_(char*, int*, int*, float*, const float*, int*,
             const float*, int*, float*, float*, int*);
void  sger_ (int*, int*, float*, const float*, int*,
             const float*, int*, float*, int*);
void  ssyr2_(char*, int*, float*, const float*, int*,
             const float*, int*, float*, int*);

/* ---- timing ------------------------------------------------------------- */
static double sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ---- data helpers ------------------------------------------------------- */
static float *fvec(int n)
{
    float *p = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) p[i] = (float)(rand() % 1000) / 500.0f - 1.0f;
    return p;
}

static float max_err(const float *a, const float *b, int n)
{
    float e = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

/* ---- reference naive SGEMM (col-major, for small N correctness check) ---- */
static void ref_sgemm_nn(int M, int N, int K,
                         float alpha, const float *A, int lda,
                         const float *B, int ldb,
                         float beta,  float *C, int ldc)
{
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            float s = 0;
            for (int p = 0; p < K; p++)
                s += A[p * lda + i] * B[j * ldb + p];
            C[j * ldc + i] = alpha * s + beta * C[j * ldc + i];
        }
}

/* ========================================================================== */

static void bench_sgemm(void)
{
    static const int SIZES[] = {64, 128, 256, 512, 1024, 2048, 0};
    printf("%-8s  %10s  %10s  %s\n", "N", "GF/s", "ms/call", "err (N<=128)");
    printf("%-8s  %10s  %10s  %s\n", "--------", "----------",
           "----------", "-----------");

    for (int si = 0; SIZES[si]; si++) {
        int N = SIZES[si];
        int reps = (N <= 256) ? 10 : (N <= 512) ? 5 : (N <= 1024) ? 5 : 4;

        float *A  = fvec(N * N);
        float *B  = fvec(N * N);
        float *C  = fvec(N * N);
        float *C0 = malloc((size_t)N * N * sizeof(float));
        memcpy(C0, C, (size_t)N * N * sizeof(float));

        char  ta = 'N', tb = 'N';
        float al = 1.0f, be = 0.0f;
        int   m = N, n = N, k = N, lda = N, ldb = N, ldc = N;

        /* warm-up */
        sgemm_(&ta, &tb, &m, &n, &k, &al, A, &lda, B, &ldb, &be, C, &ldc);

        double t0 = sec();
        for (int r = 0; r < reps; r++) {
            memcpy(C, C0, (size_t)N * N * sizeof(float));
            sgemm_(&ta, &tb, &m, &n, &k, &al, A, &lda, B, &ldb, &be, C, &ldc);
        }
        double dt = (sec() - t0) / reps;
        double gflops = 2.0 * N * N * N / dt * 1e-9;

        /* correctness check for small N */
        float err = 0;
        if (N <= 128) {
            float *Cref = malloc((size_t)N * N * sizeof(float));
            memcpy(Cref, C0, (size_t)N * N * sizeof(float));
            ref_sgemm_nn(N, N, N, al, A, lda, B, ldb, be, Cref, ldc);
            err = max_err(C, Cref, N * N);
            free(Cref);
        }

        if (N <= 128)
            printf("%-8d  %10.1f  %10.2f  %.2e\n", N, gflops, dt * 1e3, err);
        else
            printf("%-8d  %10.1f  %10.2f\n", N, gflops, dt * 1e3);

        free(A); free(B); free(C); free(C0);
    }
}

/* ========================================================================== */

static void bench_blas1(void)
{
    static const int N = 1 << 20;   /* 4 MB per vector, fits in L3 */
    int inc = 1, reps = 30;
    float alpha = 2.5f;

    float *x = fvec(N);
    float *y = fvec(N);
    float *y0 = malloc((size_t)N * sizeof(float));
    memcpy(y0, y, (size_t)N * sizeof(float));

    printf("%-8s  %10s  %10s\n", "kernel", "GF/s", "GB/s-equiv");
    printf("%-8s  %10s  %10s\n", "--------", "----------", "----------");

    /* saxpy: 2 FLOP, 12 B (read x, read+write y) */
    {
        double t0 = sec();
        for (int r = 0; r < reps; r++) {
            memcpy(y, y0, (size_t)N * sizeof(float));
            saxpy_(&N, &alpha, x, &inc, y, &inc);
        }
        double dt = (sec() - t0) / reps;
        printf("%-8s  %10.2f  %10.2f\n", "saxpy",
               2.0 * N / dt * 1e-9, 12.0 * N / dt * 1e-9);
    }

    /* sdot: 2 FLOP, 8 B (read x, read y) */
    {
        double t0 = sec();
        volatile float d = 0;
        for (int r = 0; r < reps; r++) d = sdot_(&N, x, &inc, y, &inc);
        double dt = (sec() - t0) / reps;
        (void)d;
        printf("%-8s  %10.2f  %10.2f\n", "sdot",
               2.0 * N / dt * 1e-9, 8.0 * N / dt * 1e-9);
    }

    /* snrm2: 2 FLOP, 4 B (read x only) */
    {
        double t0 = sec();
        volatile float nr = 0;
        for (int r = 0; r < reps; r++) nr = snrm2_(&N, x, &inc);
        double dt = (sec() - t0) / reps;
        (void)nr;
        printf("%-8s  %10.2f  %10.2f\n", "snrm2",
               2.0 * N / dt * 1e-9, 4.0 * N / dt * 1e-9);
    }

    /* sasum: 1 FLOP, 4 B */
    {
        double t0 = sec();
        volatile float s = 0;
        for (int r = 0; r < reps; r++) s = sasum_(&N, x, &inc);
        double dt = (sec() - t0) / reps;
        (void)s;
        printf("%-8s  %10.2f  %10.2f\n", "sasum",
               1.0 * N / dt * 1e-9, 4.0 * N / dt * 1e-9);
    }

    /* sscal: 1 FLOP, 8 B (read+write x) */
    {
        double t0 = sec();
        for (int r = 0; r < reps; r++) sscal_(&N, &alpha, x, &inc);
        double dt = (sec() - t0) / reps;
        printf("%-8s  %10.2f  %10.2f\n", "sscal",
               1.0 * N / dt * 1e-9, 8.0 * N / dt * 1e-9);
    }

    free(x); free(y); free(y0);
}

/* ========================================================================== */

static void bench_blas2(void)
{
    static const int SIZES[] = {512, 1024, 2048, 0};
    int inc = 1, reps = 20;

    printf("%-8s  %-8s  %10s  %10s\n", "kernel", "N", "GF/s", "GB/s-equiv");
    printf("%-8s  %-8s  %10s  %10s\n", "--------", "--------",
           "----------", "----------");

    for (int si = 0; SIZES[si]; si++) {
        int N = SIZES[si];
        float *A = fvec(N * N);
        float *x = fvec(N);
        float *y = fvec(N);
        float al = 1.0f, be = 0.0f;
        char  tr = 'N';

        /* sgemv */
        {
            double t0 = sec();
            for (int r = 0; r < reps; r++)
                sgemv_(&tr, &N, &N, &al, A, &N, x, &inc, &be, y, &inc);
            double dt = (sec() - t0) / reps;
            printf("%-8s  %-8d  %10.2f  %10.2f\n", "sgemv", N,
                   2.0 * N * N / dt * 1e-9,
                   (4.0 * N * N + 8.0 * N) / dt * 1e-9);
        }

        /* sger */
        {
            double t0 = sec();
            for (int r = 0; r < reps; r++)
                sger_(&N, &N, &al, x, &inc, y, &inc, A, &N);
            double dt = (sec() - t0) / reps;
            printf("%-8s  %-8d  %10.2f  %10.2f\n", "sger", N,
                   2.0 * N * N / dt * 1e-9,
                   (4.0 * N * N + 8.0 * N) / dt * 1e-9);
        }

        free(A); free(x); free(y);
    }
}

/* ========================================================================== */

int main(void)
{
    srand(42);

    printf("\n=== sgemm_ (BLAS 3, Fortran col-major, N,N, 1 thread) ===\n");
    bench_sgemm();

    printf("\n=== BLAS 1  (N=2^20 = 1M floats, OMP_NUM_THREADS threads) ===\n");
    bench_blas1();

    printf("\n=== BLAS 2  (single thread) ===\n");
    bench_blas2();

    return 0;
}
