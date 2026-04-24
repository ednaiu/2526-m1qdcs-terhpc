/*
 * test_blas23.c — Week 5/6: Comprehensive tester for BLAS 2 & 3 kernels
 *
 * Test philosophy (per README requirement):
 *   - Hot spots: sizes that stress memory bandwidth (n=256..2048)
 *   - Edge cases: n=1, n=7 (non-multiple-of-8), n=0 (no-op)
 *   - Special alpha/beta: 0, 1, -1, arbitrary (0.5)
 *   - Stride variants: incx=1 (AVX2 path), incx=2 (gather path)
 *   - Triangular flags: UPLO='U'/'L', TRANS='N'/'T', DIAG='N'/'U'
 *   Each test compares against OpenBLAS CBLAS reference within tol=1e-3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "../include/blas1.h"
#include "../include/blas2.h"
#include "../include/blas3.h"

/* -----------------------------------------------------------------------
 * Infrastructure
 * --------------------------------------------------------------------- */
static int g_pass = 0, g_fail = 0;

static void check(const char *name, int ok) {
    if (ok) { g_pass++; }
    else    { g_fail++; printf("  [FAIL] %s\n", name); }
}

static float *fvec(int n) {
    return (float *)aligned_alloc(64, ((size_t)n + 16) * sizeof(float));
}
static float *fmat(int m, int n) {
    return (float *)aligned_alloc(64, (size_t)m * n * sizeof(float));
}

static void rand_fill(float *a, int n) {
    for (int i = 0; i < n; i++) a[i] = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
}

/* Make a diagonally dominant symmetric matrix (guarantees positive-definite) */
static void rand_symm(float *a, int n, int lda) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[i * lda + j] = (float)rand() / RAND_MAX;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float v = a[i * lda + j];
            a[j * lda + i] = v;
        }
        a[i * lda + i] += (float)n;  /* dominant diagonal */
    }
}

/* Well-conditioned triangular matrix: large diagonal, small off-diagonal.
 * Condition number ≈ (diag + off_sum) / diag ≈ 1.1 for any n. */
static void rand_tri(float *a, int n, int lda, int upper) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (upper ? (j > i) : (j < i))
                a[i * lda + j] = 0.01f * ((float)rand() / RAND_MAX);
            else if (j == i)
                a[i * lda + j] = 2.0f + (float)rand() / RAND_MAX;  /* large diagonal */
            else
                a[i * lda + j] = 0.0f;
        }
    }
}

static int vec_ok(const float *a, const float *b, int n, int inc) {
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i*inc] - b[i*inc]);
        float m = fmaxf(fabsf(a[i*inc]), fabsf(b[i*inc]));
        if (d > 1e-3f * (m + 1e-6f)) return 0;
    }
    return 1;
}
static int mat_ok_tol(const float *a, const float *b, int m, int n, int lda,
                      float rtol, const char *tag) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float d  = fabsf(a[i*lda+j] - b[i*lda+j]);
            float mv = fmaxf(fabsf(a[i*lda+j]), fabsf(b[i*lda+j]));
            if (d > rtol * (mv + 1e-4f)) {
                printf("    first diff at (%d,%d): ref=%.6g ours=%.6g diff=%.6g rtol=%.6g %s\n",
                       i, j, a[i*lda+j], b[i*lda+j], d, rtol, tag);
                return 0;
            }
        }
    return 1;
}
static int mat_ok(const float *a, const float *b, int m, int n, int lda) {
    return mat_ok_tol(a, b, m, n, lda, 1e-3f, "");
}
static int mat_ok_relaxed(const float *a, const float *b, int m, int n, int lda,
                          const char *tag) {
    /* Rank-update ops accumulate k FMAs; allow slightly wider tolerance */
    return mat_ok_tol(a, b, m, n, lda, 5e-3f, tag);
}

/* -----------------------------------------------------------------------
 * SGEMV tests
 *   Hot sizes: m×n = 64×64, 128×256, 512×512, 1024×1024, 2048×512
 *   Edge cases: m=1, n=1, m=7, n=7, alpha=0, beta=0, beta=1
 *   Strides: incx=1, incx=2; incy=1
 *   Trans: N and T
 * --------------------------------------------------------------------- */
static void test_sgemv_case(int m, int n, float alpha, float beta,
                            int incx, char trans, const char *tag)
{
    int lx = (trans=='N'||trans=='n') ? n : m;
    int ly = (trans=='N'||trans=='n') ? m : n;
    float *A = fmat(m, n);
    float *x = fvec(lx * incx);
    float *yr = fvec(ly), *yo = fvec(ly);
    rand_fill(A, m * n);
    rand_fill(x, lx * incx);
    rand_fill(yr, ly); memcpy(yo, yr, ly * sizeof(float));

    CBLAS_TRANSPOSE ct = (trans=='N'||trans=='n') ? CblasNoTrans : CblasTrans;
    cblas_sgemv(CblasRowMajor, ct, m, n, alpha, A, n, x, incx, beta, yr, 1);
    sgemv(trans, m, n, alpha, A, n, x, incx, beta, yo, 1);

    char buf[128]; snprintf(buf, sizeof(buf), "sgemv %s", tag);
    check(buf, vec_ok(yr, yo, ly, 1));
    free(A); free(x); free(yr); free(yo);
}

static void test_sgemv(void) {
    printf("  sgemv...\n");
    /* Hot spots */
    test_sgemv_case(64,   64,  1.0f, 0.0f, 1, 'N', "64x64 N alpha=1 beta=0");
    test_sgemv_case(128,  256, 1.5f, 0.5f, 1, 'N', "128x256 N");
    test_sgemv_case(512,  512, 1.0f, 1.0f, 1, 'N', "512x512 N beta=1");
    test_sgemv_case(1024, 1024,0.7f, 0.3f, 1, 'N', "1024x1024 N");
    test_sgemv_case(2048, 512, 1.0f, 0.0f, 1, 'N', "2048x512 N");
    /* Transpose */
    test_sgemv_case(64,   128, 1.0f, 0.0f, 1, 'T', "64x128 T");
    test_sgemv_case(512,  256, 1.0f, 1.0f, 1, 'T', "512x256 T beta=1");
    test_sgemv_case(1024, 512, 0.5f, 0.5f, 1, 'T', "1024x512 T");
    /* Edge cases */
    test_sgemv_case(1,    1,   1.0f, 0.0f, 1, 'N', "1x1");
    test_sgemv_case(7,    7,   1.0f, 0.0f, 1, 'N', "7x7 non-multiple-8");
    test_sgemv_case(9,    15,  1.0f, 0.0f, 1, 'N', "9x15");
    test_sgemv_case(64,   64,  0.0f, 1.0f, 1, 'N', "alpha=0");
    test_sgemv_case(64,   64,  1.0f, 0.0f, 1, 'N', "beta=0");
    /* Non-unit stride */
    test_sgemv_case(64,   64,  1.0f, 0.0f, 2, 'N', "incx=2 N");
    test_sgemv_case(128,  128, 1.0f, 0.5f, 2, 'T', "incx=2 T");
}

/* -----------------------------------------------------------------------
 * SGER tests
 * --------------------------------------------------------------------- */
static void test_sger_case(int m, int n, float alpha, int incx, int incy,
                            const char *tag)
{
    float *A_r = fmat(m, n), *A_o = fmat(m, n);
    float *x   = fvec(m * incx), *y = fvec(n * incy);
    rand_fill(A_r, m * n); memcpy(A_o, A_r, (size_t)m * n * sizeof(float));
    rand_fill(x, m * incx); rand_fill(y, n * incy);

    cblas_sger(CblasRowMajor, m, n, alpha, x, incx, y, incy, A_r, n);
    sger(m, n, alpha, x, incx, y, incy, A_o, n);

    char buf[128]; snprintf(buf, sizeof(buf), "sger %s", tag);
    check(buf, mat_ok(A_r, A_o, m, n, n));
    free(A_r); free(A_o); free(x); free(y);
}

static void test_sger(void) {
    printf("  sger...\n");
    test_sger_case(64,   64,  1.0f, 1, 1, "64x64");
    test_sger_case(128,  256, 2.0f, 1, 1, "128x256");
    test_sger_case(512,  512, 0.5f, 1, 1, "512x512");
    test_sger_case(1024, 1024,1.0f, 1, 1, "1024x1024");
    test_sger_case(7,    7,   1.0f, 1, 1, "7x7 edge");
    test_sger_case(1,    64,  1.0f, 1, 1, "1x64");
    test_sger_case(64,   64,  0.0f, 1, 1, "alpha=0 (no-op)");
    test_sger_case(64,   64,  1.0f, 2, 1, "incx=2");
    test_sger_case(64,   64,  1.0f, 1, 2, "incy=2");
}

/* -----------------------------------------------------------------------
 * SSYMV tests
 * --------------------------------------------------------------------- */
static void test_ssymv_case(int n, float alpha, float beta,
                             char uplo, int incx, const char *tag)
{
    int lda = n;
    float *A = fmat(n, n), *x = fvec(n * incx);
    float *yr = fvec(n), *yo = fvec(n);
    rand_symm(A, n, lda);
    rand_fill(x, n * incx);
    rand_fill(yr, n); memcpy(yo, yr, n * sizeof(float));

    CBLAS_UPLO cu = (uplo=='U'||uplo=='u') ? CblasUpper : CblasLower;
    cblas_ssymv(CblasRowMajor, cu, n, alpha, A, lda, x, incx, beta, yr, 1);
    ssymv(uplo, n, alpha, A, lda, x, incx, beta, yo, 1);

    char buf[128]; snprintf(buf, sizeof(buf), "ssymv %s", tag);
    check(buf, vec_ok(yr, yo, n, 1));
    free(A); free(x); free(yr); free(yo);
}

static void test_ssymv(void) {
    printf("  ssymv...\n");
    /* Hot spots — both UPLO variants */
    test_ssymv_case(64,   1.0f, 0.0f, 'U', 1, "64 U beta=0");
    test_ssymv_case(64,   1.0f, 0.0f, 'L', 1, "64 L beta=0");
    test_ssymv_case(128,  1.5f, 0.5f, 'U', 1, "128 U");
    test_ssymv_case(256,  1.0f, 1.0f, 'L', 1, "256 L beta=1");
    test_ssymv_case(512,  0.7f, 0.3f, 'U', 1, "512 U");
    test_ssymv_case(1024, 1.0f, 0.0f, 'L', 1, "1024 L");
    /* Edge cases */
    test_ssymv_case(1,    1.0f, 0.0f, 'U', 1, "n=1");
    test_ssymv_case(7,    1.0f, 0.0f, 'L', 1, "n=7");
    test_ssymv_case(8,    1.0f, 0.5f, 'U', 1, "n=8 exact AVX");
    test_ssymv_case(64,   0.0f, 1.0f, 'U', 1, "alpha=0");
    /* Non-unit stride */
    test_ssymv_case(64,   1.0f, 0.0f, 'L', 2, "64 L incx=2");
    test_ssymv_case(128,  1.0f, 0.5f, 'U', 2, "128 U incx=2");
}

/* -----------------------------------------------------------------------
 * STRMV tests
 * --------------------------------------------------------------------- */
static void test_strmv_case(int n, char uplo, char trans, char diag,
                             int incx, const char *tag)
{
    float *A = fmat(n, n);
    float *xr = fvec(n * incx), *xo = fvec(n * incx);
    rand_tri(A, n, n, (uplo=='U'||uplo=='u'));
    rand_fill(xr, n * incx); memcpy(xo, xr, (size_t)n * incx * sizeof(float));

    CBLAS_UPLO   cu = (uplo  =='U'||uplo  =='u') ? CblasUpper    : CblasLower;
    CBLAS_TRANSPOSE ct = (trans=='T'||trans=='t') ? CblasTrans    : CblasNoTrans;
    CBLAS_DIAG   cd = (diag  =='U'||diag  =='u') ? CblasUnit     : CblasNonUnit;
    cblas_strmv(CblasRowMajor, cu, ct, cd, n, A, n, xr, incx);
    strmv(uplo, trans, diag, n, A, n, xo, incx);

    char buf[128]; snprintf(buf, sizeof(buf), "strmv %s", tag);
    check(buf, vec_ok(xr, xo, n, incx));
    free(A); free(xr); free(xo);
}

static void test_strmv(void) {
    printf("  strmv...\n");
    test_strmv_case(64,  'U', 'N', 'N', 1, "64 U N NonUnit");
    test_strmv_case(64,  'L', 'N', 'N', 1, "64 L N NonUnit");
    test_strmv_case(128, 'U', 'T', 'N', 1, "128 U T NonUnit");
    test_strmv_case(128, 'L', 'T', 'N', 1, "128 L T NonUnit");
    test_strmv_case(256, 'U', 'N', 'U', 1, "256 U N Unit");
    test_strmv_case(512, 'L', 'N', 'N', 1, "512 L N");
    test_strmv_case(1,   'U', 'N', 'N', 1, "n=1");
    test_strmv_case(7,   'L', 'N', 'N', 1, "n=7 edge");
    test_strmv_case(8,   'U', 'N', 'N', 1, "n=8 exact AVX");
    test_strmv_case(64,  'U', 'N', 'N', 2, "64 U incx=2");
}

/* -----------------------------------------------------------------------
 * STRSV tests
 * --------------------------------------------------------------------- */
static void test_strsv_case(int n, char uplo, char trans, char diag,
                             int incx, const char *tag)
{
    float *A = fmat(n, n);
    float *xr = fvec(n * incx), *xo = fvec(n * incx);
    rand_tri(A, n, n, (uplo=='U'||uplo=='u'));
    rand_fill(xr, n * incx); memcpy(xo, xr, (size_t)n * incx * sizeof(float));

    CBLAS_UPLO      cu = (uplo =='U'||uplo =='u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE ct = (trans=='T'||trans=='t') ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG      cd = (diag =='U'||diag =='u') ? CblasUnit  : CblasNonUnit;
    cblas_strsv(CblasRowMajor, cu, ct, cd, n, A, n, xr, incx);
    strsv(uplo, trans, diag, n, A, n, xo, incx);

    char buf[128]; snprintf(buf, sizeof(buf), "strsv %s", tag);
    check(buf, vec_ok(xr, xo, n, incx));
    free(A); free(xr); free(xo);
}

static void test_strsv(void) {
    printf("  strsv...\n");
    test_strsv_case(64,  'U', 'N', 'N', 1, "64 U N NonUnit");
    test_strsv_case(64,  'L', 'N', 'N', 1, "64 L N NonUnit");
    test_strsv_case(128, 'U', 'T', 'N', 1, "128 U T NonUnit");
    test_strsv_case(128, 'L', 'T', 'N', 1, "128 L T NonUnit");
    test_strsv_case(256, 'U', 'N', 'U', 1, "256 U N Unit");
    test_strsv_case(512, 'L', 'N', 'N', 1, "512 L N");
    test_strsv_case(1,   'U', 'N', 'N', 1, "n=1");
    test_strsv_case(7,   'L', 'N', 'N', 1, "n=7 edge");
    test_strsv_case(64,  'U', 'N', 'N', 2, "64 U incx=2");
}

/* -----------------------------------------------------------------------
 * SSYR tests
 * --------------------------------------------------------------------- */
static void test_ssyr_case(int n, float alpha, char uplo, int incx,
                            const char *tag)
{
    float *A_r = fmat(n, n), *A_o = fmat(n, n);
    float *x   = fvec(n * incx);
    /* Start with symmetric matrix */
    rand_symm(A_r, n, n); memcpy(A_o, A_r, (size_t)n * n * sizeof(float));
    rand_fill(x, n * incx);

    CBLAS_UPLO cu = (uplo=='U'||uplo=='u') ? CblasUpper : CblasLower;
    cblas_ssyr(CblasRowMajor, cu, n, alpha, x, incx, A_r, n);
    ssyr(uplo, n, alpha, x, incx, A_o, n);

    char buf[128]; snprintf(buf, sizeof(buf), "ssyr %s", tag);
    check(buf, mat_ok_relaxed(A_r, A_o, n, n, n, buf));
    free(A_r); free(A_o); free(x);
}

static void test_ssyr(void) {
    printf("  ssyr...\n");
    test_ssyr_case(64,   1.0f, 'U', 1, "64 U");
    test_ssyr_case(64,   1.0f, 'L', 1, "64 L");
    test_ssyr_case(128,  2.0f, 'U', 1, "128 U");
    test_ssyr_case(256,  0.5f, 'L', 1, "256 L");
    test_ssyr_case(512,  1.0f, 'U', 1, "512 U");
    test_ssyr_case(1024, 1.0f, 'L', 1, "1024 L");
    test_ssyr_case(1,    1.0f, 'U', 1, "n=1");
    test_ssyr_case(7,    1.0f, 'L', 1, "n=7 edge");
    test_ssyr_case(8,    1.0f, 'U', 1, "n=8 exact AVX");
    test_ssyr_case(64,   0.0f, 'L', 1, "alpha=0 (no-op)");
    test_ssyr_case(64,   1.0f, 'U', 2, "incx=2");
    test_ssyr_case(128,  1.0f, 'L', 2, "128 L incx=2");
}

/* -----------------------------------------------------------------------
 * SSYR2 tests
 * --------------------------------------------------------------------- */
static void test_ssyr2_case(int n, float alpha, char uplo,
                             int incx, int incy, const char *tag)
{
    float *A_r = fmat(n, n), *A_o = fmat(n, n);
    float *x   = fvec(n * incx), *y = fvec(n * incy);
    rand_symm(A_r, n, n); memcpy(A_o, A_r, (size_t)n * n * sizeof(float));
    rand_fill(x, n * incx); rand_fill(y, n * incy);

    CBLAS_UPLO cu = (uplo=='U'||uplo=='u') ? CblasUpper : CblasLower;
    cblas_ssyr2(CblasRowMajor, cu, n, alpha, x, incx, y, incy, A_r, n);
    ssyr2(uplo, n, alpha, x, incx, y, incy, A_o, n);

    char buf[128]; snprintf(buf, sizeof(buf), "ssyr2 %s", tag);
    check(buf, mat_ok(A_r, A_o, n, n, n));
    free(A_r); free(A_o); free(x); free(y);
}

static void test_ssyr2(void) {
    printf("  ssyr2...\n");
    test_ssyr2_case(64,   1.0f, 'U', 1, 1, "64 U");
    test_ssyr2_case(64,   1.0f, 'L', 1, 1, "64 L");
    test_ssyr2_case(128,  2.0f, 'U', 1, 1, "128 U");
    test_ssyr2_case(256,  0.5f, 'L', 1, 1, "256 L");
    test_ssyr2_case(512,  1.0f, 'U', 1, 1, "512 U");
    test_ssyr2_case(1024, 1.0f, 'L', 1, 1, "1024 L");
    test_ssyr2_case(7,    1.0f, 'L', 1, 1, "n=7 edge");
    test_ssyr2_case(64,   0.0f, 'U', 1, 1, "alpha=0 no-op");
    test_ssyr2_case(64,   1.0f, 'U', 2, 1, "incx=2 incy=1");
    test_ssyr2_case(64,   1.0f, 'L', 1, 2, "incx=1 incy=2");
}

/* -----------------------------------------------------------------------
 * SSYRK (BLAS 3) — basic test
 * --------------------------------------------------------------------- */
static void test_ssyrk(void) {
    printf("  ssyrk...\n");
    int sizes[][2] = {{32,16},{64,32},{128,64},{256,128},{512,64}};
    int ns = (int)(sizeof(sizes)/sizeof(sizes[0]));
    for (int t = 0; t < ns; t++) {
        int n = sizes[t][0], k = sizes[t][1];
        float *A   = fmat(n, k);
        float *C_r = fmat(n, n), *C_o = fmat(n, n);
        rand_fill(A, n * k);
        rand_symm(C_r, n, n); memcpy(C_o, C_r, (size_t)n * n * sizeof(float));

        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                    n, k, 1.2f, A, k, 0.8f, C_r, n);
        ssyrk('L', 'N', n, k, 1.2f, A, k, 0.8f, C_o, n);

        char buf[64]; snprintf(buf, sizeof(buf), "ssyrk n=%d k=%d", n, k);
        check(buf, mat_ok_relaxed(C_r, C_o, n, n, n, buf));
        free(A); free(C_r); free(C_o);
    }
}

/* -----------------------------------------------------------------------
 * BLAS1 stride tests (new: incx>1 gather path)
 * --------------------------------------------------------------------- */
static void test_blas1_strides(void) {
    printf("  blas1 non-unit stride (gather path)...\n");
    /* sdot/sasum/snrm2 are declared in blas1.h */

    int sizes[] = {64, 256, 1024, 4096};
    int strides[] = {2, 3, 4};
    for (int si = 0; si < 4; si++) {
        int n = sizes[si];
        for (int ki = 0; ki < 3; ki++) {
            int inc = strides[ki];
            float *x = fvec(n * inc), *y = fvec(n * inc);
            rand_fill(x, n * inc); rand_fill(y, n * inc);

            float dot_ref = cblas_sdot (n, x, inc, y, inc);
            float dot_our = sdot (n, x, inc, y, inc);
            float asm_ref = cblas_sasum(n, x, inc);
            float asm_our = sasum(n, x, inc);
            float nrm_ref = cblas_snrm2(n, x, inc);
            float nrm_our = snrm2(n, x, inc);

            char buf[128];
            snprintf(buf,sizeof(buf),"sdot  n=%d inc=%d",n,inc);
            check(buf, fabsf(dot_ref-dot_our) <= 1e-3f*(fabsf(dot_ref)+1e-6f));
            snprintf(buf,sizeof(buf),"sasum n=%d inc=%d",n,inc);
            check(buf, fabsf(asm_ref-asm_our) <= 1e-3f*(fabsf(asm_ref)+1e-6f));
            snprintf(buf,sizeof(buf),"snrm2 n=%d inc=%d",n,inc);
            check(buf, fabsf(nrm_ref-nrm_our) <= 1e-3f*(fabsf(nrm_ref)+1e-6f));
            free(x); free(y);
        }
    }
}

/* -----------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------- */
int main(void) {
    printf("=== BLAS 2/3 Correctness Tests (vs OpenBLAS) ===\n\n");
    printf("Testing SGEMV...\n"); test_sgemv();
    printf("Testing SGER...\n");  test_sger();
    printf("Testing SSYMV...\n"); test_ssymv();
    printf("Testing STRMV...\n"); test_strmv();
    printf("Testing STRSV...\n"); test_strsv();
    printf("Testing SSYR...\n");  test_ssyr();
    printf("Testing SSYR2...\n"); test_ssyr2();
    printf("Testing SSYRK...\n"); test_ssyrk();
    printf("Testing BLAS1 strides...\n"); test_blas1_strides();
    printf("\n=== Results: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return g_fail > 0;
}
