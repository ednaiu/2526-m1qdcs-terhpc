/*
 * test_blas1.c — Week 4: Automated tester for BLAS 1 kernels
 *
 * Compares all three implementation flavours (scalar, avx2, parallel)
 * against OpenBLAS reference for every kernel.
 *
 * Test categories per kernel:
 *   - Normal sizes: 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255,
 *                   256, 1023, 1024, 4096
 *   - Edge / pathological: n=0, all-zeros, all-ones, alternating ±1,
 *                          large values, small values
 *   - Special: alpha=0, alpha=1, c=1 s=0 identity rotation, etc.
 *
 * Build:
 *   gcc -O2 -march=native -fopenmp -mavx2 -mfma \
 *       -o bin/test_blas1 tests/test_blas1.c src/blas1.c \
 *       -lopenblas -lm
 *
 * Compile: make test-blas1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "../include/blas1.h"

/* OpenBLAS C API */
#include <cblas.h>

/* -----------------------------------------------------------------------
 * Global counters
 * --------------------------------------------------------------------- */
static int g_pass = 0, g_fail = 0;

/* -----------------------------------------------------------------------
 * Utility
 * --------------------------------------------------------------------- */
static float *alloc_vec(int n)
{
    float *p = (float *)aligned_alloc(64, ((n + 8) * sizeof(float)));
    if (!p) { perror("alloc_vec"); exit(1); }
    return p;
}

static void fill_seq(float *v, int n, float start, float step)
{
    for (int i = 0; i < n; i++) v[i] = start + i * step;
}

static void fill_alt(float *v, int n)
{
    for (int i = 0; i < n; i++) v[i] = (i & 1) ? -1.0f : 1.0f;
}

static void fill_val(float *v, int n, float val)
{
    for (int i = 0; i < n; i++) v[i] = val;
}

static void copy_vec(float *dst, const float *src, int n)
{
    memcpy(dst, src, n * sizeof(float));
}

/* -----------------------------------------------------------------------
 * Check helpers
 * --------------------------------------------------------------------- */
#define REL_TOL  1e-4f   /* relative tolerance */
#define ABS_TOL  1e-5f   /* absolute tolerance (for near-zero values) */

static int float_close(float a, float b)
{
    float diff = fabsf(a - b);
    /* Accept if absolute diff is tiny */
    if (diff < ABS_TOL) return 1;
    /* Otherwise check relative error */
    float mag = fabsf(b);
    if (mag < ABS_TOL) mag = ABS_TOL;
    return diff / mag < REL_TOL;
}

static int vec_close(const float *a, const float *b, int n)
{
    for (int i = 0; i < n; i++)
        if (!float_close(a[i], b[i])) return 0;
    return 1;
}

static void report(const char *name, int n, const char *variant, int ok)
{
    if (ok) {
        g_pass++;
        /* Uncomment to see all passes: */
        /* printf("  [PASS] %s (%s, n=%d)\n", name, variant, n); */
    } else {
        g_fail++;
        printf("  [FAIL] %s (%s, n=%d)\n", name, variant, n);
    }
}

/* -----------------------------------------------------------------------
 * Test: sscal
 * --------------------------------------------------------------------- */
static void test_sscal_case(int n, float alpha, const float *xin)
{
    float *ref = alloc_vec(n+1), *our = alloc_vec(n+1);
    copy_vec(ref, xin, n);
    copy_vec(our, xin, n);

    /* Reference */
    cblas_sscal(n, alpha, ref, 1);

    /* scalar */
    copy_vec(our, xin, n);
    sscal_scalar(n, alpha, our, 1);
    report("sscal", n, "scalar", vec_close(our, ref, n));

    /* avx2 */
    copy_vec(our, xin, n);
    sscal_avx2(n, alpha, our, 1);
    report("sscal", n, "avx2", vec_close(our, ref, n));

    /* parallel */
    copy_vec(our, xin, n);
    sscal(n, alpha, our, 1);
    report("sscal", n, "parallel", vec_close(our, ref, n));

    free(ref); free(our);
}

static void test_sscal(void)
{
    printf("=== sscal ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);

    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n + 1);
        fill_seq(x, n, -1.5f, 0.37f);
        test_sscal_case(n, 2.5f, x);
        test_sscal_case(n, 0.0f, x);   /* alpha=0: zero out */
        test_sscal_case(n, 1.0f, x);   /* alpha=1: identity */
        test_sscal_case(n, -1.0f, x);  /* negate */
        free(x);
    }

    /* Large values */
    {
        int n = 256;
        float *x = alloc_vec(n);
        fill_val(x, n, 1e38f);
        test_sscal_case(n, 0.5f, x);
        free(x);
    }
    /* Alternating ±1 */
    {
        int n = 256;
        float *x = alloc_vec(n);
        fill_alt(x, n);
        test_sscal_case(n, 3.14f, x);
        free(x);
    }
}

/* -----------------------------------------------------------------------
 * Test: scopy
 * --------------------------------------------------------------------- */
static void test_scopy_case(int n, const float *xin)
{
    float *ref = alloc_vec(n+1), *our = alloc_vec(n+1);
    copy_vec(ref, xin, n);
    copy_vec(our, xin, n);

    float *yref = alloc_vec(n+1);
    float *your = alloc_vec(n+1);
    fill_val(yref, n, 9.9f);
    fill_val(your, n, 9.9f);

    cblas_scopy(n, xin, 1, yref, 1);

    copy_vec(your, yref, n); /* reset our to yref before comparison */
    fill_val(your, n, 9.9f);
    scopy_scalar(n, xin, 1, your, 1);
    report("scopy", n, "scalar", vec_close(your, yref, n));

    fill_val(your, n, 9.9f);
    scopy_avx2(n, xin, 1, your, 1);
    report("scopy", n, "avx2", vec_close(your, yref, n));

    fill_val(your, n, 9.9f);
    scopy(n, xin, 1, your, 1);
    report("scopy", n, "parallel", vec_close(your, yref, n));

    free(ref); free(our); free(yref); free(your);
}

static void test_scopy(void)
{
    printf("=== scopy ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n + 1);
        fill_seq(x, n, -2.0f, 0.5f);
        test_scopy_case(n, x);
        free(x);
    }
}

/* -----------------------------------------------------------------------
 * Test: sswap
 * --------------------------------------------------------------------- */
static void test_sswap_case(int n, const float *xin, const float *yin)
{
    float *xref = alloc_vec(n+1), *yref = alloc_vec(n+1);
    float *xour = alloc_vec(n+1), *your = alloc_vec(n+1);
    copy_vec(xref, xin, n); copy_vec(yref, yin, n);
    cblas_sswap(n, xref, 1, yref, 1);

    copy_vec(xour, xin, n); copy_vec(your, yin, n);
    sswap_scalar(n, xour, 1, your, 1);
    report("sswap", n, "scalar", vec_close(xour, xref, n) && vec_close(your, yref, n));

    copy_vec(xour, xin, n); copy_vec(your, yin, n);
    sswap_avx2(n, xour, 1, your, 1);
    report("sswap", n, "avx2", vec_close(xour, xref, n) && vec_close(your, yref, n));

    copy_vec(xour, xin, n); copy_vec(your, yin, n);
    sswap(n, xour, 1, your, 1);
    report("sswap", n, "parallel", vec_close(xour, xref, n) && vec_close(your, yref, n));

    free(xref); free(yref); free(xour); free(your);
}

static void test_sswap(void)
{
    printf("=== sswap ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n + 1), *y = alloc_vec(n + 1);
        fill_seq(x, n, -2.0f, 0.5f);
        fill_seq(y, n,  1.0f, 0.3f);
        test_sswap_case(n, x, y);
        free(x); free(y);
    }
}

/* -----------------------------------------------------------------------
 * Test: saxpy
 * --------------------------------------------------------------------- */
static void test_saxpy_case(int n, float alpha, const float *xin, const float *yin)
{
    float *yref = alloc_vec(n+1), *your = alloc_vec(n+1);
    copy_vec(yref, yin, n);
    cblas_saxpy(n, alpha, xin, 1, yref, 1);

    copy_vec(your, yin, n);
    saxpy_scalar(n, alpha, xin, 1, your, 1);
    report("saxpy", n, "scalar", vec_close(your, yref, n));

    copy_vec(your, yin, n);
    saxpy_avx2(n, alpha, xin, 1, your, 1);
    report("saxpy", n, "avx2", vec_close(your, yref, n));

    copy_vec(your, yin, n);
    saxpy(n, alpha, xin, 1, your, 1);
    report("saxpy", n, "parallel", vec_close(your, yref, n));

    free(yref); free(your);
}

static void test_saxpy(void)
{
    printf("=== saxpy ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n+1), *y = alloc_vec(n+1);
        fill_seq(x, n, -1.0f, 0.3f);
        fill_seq(y, n,  2.0f, 0.7f);
        test_saxpy_case(n, 2.5f, x, y);
        test_saxpy_case(n, 0.0f, x, y);   /* alpha=0: no-op */
        test_saxpy_case(n, 1.0f, x, y);   /* alpha=1 */
        test_saxpy_case(n, -1.0f, x, y);
        free(x); free(y);
    }
    /* alternating ±1 */
    {
        int n = 512;
        float *x = alloc_vec(n), *y = alloc_vec(n);
        fill_alt(x, n); fill_alt(y, n);
        test_saxpy_case(n, 3.0f, x, y);
        free(x); free(y);
    }
}

/* -----------------------------------------------------------------------
 * Test: sdot
 * --------------------------------------------------------------------- */
static void test_sdot_case(int n, const float *xin, const float *yin)
{
    float ref = cblas_sdot(n, xin, 1, yin, 1);

    float vs = sdot_scalar(n, xin, 1, yin, 1);
    report("sdot", n, "scalar", float_close(vs, ref));

    float va = sdot_avx2(n, xin, 1, yin, 1);
    report("sdot", n, "avx2", float_close(va, ref));

    float vp = sdot(n, xin, 1, yin, 1);
    report("sdot", n, "parallel", float_close(vp, ref));
}

static void test_sdot(void)
{
    printf("=== sdot ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n+1), *y = alloc_vec(n+1);
        fill_seq(x, n, -1.0f, 0.3f);
        fill_seq(y, n,  0.5f, 0.2f);
        test_sdot_case(n, x, y);
        free(x); free(y);
    }
    /* all zeros */
    {
        int n = 128;
        float *x = alloc_vec(n), *y = alloc_vec(n);
        fill_val(x, n, 0.0f); fill_val(y, n, 0.0f);
        test_sdot_case(n, x, y);
        free(x); free(y);
    }
}

/* -----------------------------------------------------------------------
 * Test: snrm2
 * --------------------------------------------------------------------- */
static void test_snrm2_case(int n, const float *xin)
{
    float ref = cblas_snrm2(n, xin, 1);
    float vs = snrm2_scalar(n, xin, 1);
    report("snrm2", n, "scalar", float_close(vs, ref));
    float va = snrm2_avx2(n, xin, 1);
    report("snrm2", n, "avx2", float_close(va, ref));
    float vp = snrm2(n, xin, 1);
    report("snrm2", n, "parallel", float_close(vp, ref));
}

static void test_snrm2(void)
{
    printf("=== snrm2 ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n+1);
        fill_seq(x, n, -2.0f, 0.4f);
        test_snrm2_case(n, x);
        fill_val(x, n, 0.0f);
        test_snrm2_case(n, x);
        fill_val(x, n, 1.0f);
        test_snrm2_case(n, x);
        free(x);
    }
}

/* -----------------------------------------------------------------------
 * Test: sasum
 * --------------------------------------------------------------------- */
static void test_sasum_case(int n, const float *xin)
{
    float ref = cblas_sasum(n, xin, 1);
    float vs = sasum_scalar(n, xin, 1);
    report("sasum", n, "scalar", float_close(vs, ref));
    float va = sasum_avx2(n, xin, 1);
    report("sasum", n, "avx2", float_close(va, ref));
    float vp = sasum(n, xin, 1);
    report("sasum", n, "parallel", float_close(vp, ref));
}

static void test_sasum(void)
{
    printf("=== sasum ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n+1);
        fill_seq(x, n, -2.0f, 0.4f);
        test_sasum_case(n, x);
        fill_alt(x, n);
        test_sasum_case(n, x);
        free(x);
    }
}

/* -----------------------------------------------------------------------
 * Test: isamax
 * --------------------------------------------------------------------- */
static void test_isamax_case(int n, const float *xin, const char *desc)
{
    if (n == 0) {
        /* isamax of empty = -1 (C) / 0 (Fortran), OpenBLAS returns 0 (Fortran) */
        int vs = isamax_scalar(n, xin, 1);
        int va = isamax_avx2(n, xin, 1);
        int vp = isamax(n, xin, 1);
        report("isamax(n=0)", n, "scalar",   vs == -1);
        report("isamax(n=0)", n, "avx2",     va == -1);
        report("isamax(n=0)", n, "parallel", vp == -1);
        return;
    }
    /* OpenBLAS cblas_isamax returns 0-based index */
    int ref = (int)cblas_isamax(n, xin, 1);

    /* For ties, OpenBLAS returns first occurrence; so should we */
    int vs = isamax_scalar(n, xin, 1);
    int va = isamax_avx2(n, xin, 1);
    int vp = isamax(n, xin, 1);

    /* Check: index must point to a value with same |x| as reference */
    float ref_val = fabsf(xin[ref]);
    int ok_s = (fabsf(fabsf(xin[vs]) - ref_val) < 1e-6f);
    int ok_a = (fabsf(fabsf(xin[va]) - ref_val) < 1e-6f);
    int ok_p = (fabsf(fabsf(xin[vp]) - ref_val) < 1e-6f);

    char tag[64];
    snprintf(tag, sizeof(tag), "scalar(%s)", desc);
    report("isamax", n, tag, ok_s);
    snprintf(tag, sizeof(tag), "avx2(%s)", desc);
    report("isamax", n, tag, ok_a);
    snprintf(tag, sizeof(tag), "parallel(%s)", desc);
    report("isamax", n, tag, ok_p);
}

static void test_isamax(void)
{
    printf("=== isamax ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);
    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n + 1);
        fill_seq(x, n, -2.0f, 0.4f);
        test_isamax_case(n, x, "seq");
        fill_alt(x, n);
        test_isamax_case(n, x, "alt");
        /* All equal */
        fill_val(x, n, 5.0f);
        test_isamax_case(n, x, "all5");
        /* Max at end */
        fill_val(x, n, 1.0f);
        if (n > 0) x[n-1] = 999.0f;
        test_isamax_case(n, x, "maxlast");
        /* Max at start */
        fill_val(x, n, 1.0f);
        if (n > 0) x[0] = 999.0f;
        test_isamax_case(n, x, "maxfirst");
        free(x);
    }
}

/* -----------------------------------------------------------------------
 * Test: srot
 * --------------------------------------------------------------------- */
static void test_srot_case(int n, const float *xin, const float *yin, float c, float s)
{
    float *xref = alloc_vec(n+1), *yref = alloc_vec(n+1);
    float *xour = alloc_vec(n+1), *your = alloc_vec(n+1);
    copy_vec(xref, xin, n); copy_vec(yref, yin, n);
    cblas_srot(n, xref, 1, yref, 1, c, s);

    copy_vec(xour, xin, n); copy_vec(your, yin, n);
    srot_scalar(n, xour, 1, your, 1, c, s);
    report("srot", n, "scalar", vec_close(xour, xref, n) && vec_close(your, yref, n));

    copy_vec(xour, xin, n); copy_vec(your, yin, n);
    srot_avx2(n, xour, 1, your, 1, c, s);
    report("srot", n, "avx2", vec_close(xour, xref, n) && vec_close(your, yref, n));

    copy_vec(xour, xin, n); copy_vec(your, yin, n);
    srot(n, xour, 1, your, 1, c, s);
    report("srot", n, "parallel", vec_close(xour, xref, n) && vec_close(your, yref, n));

    free(xref); free(yref); free(xour); free(your);
}

static void test_srot(void)
{
    printf("=== srot ===\n");
    int sizes[] = {0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1023, 1024, 4096};
    int ns = sizeof(sizes)/sizeof(sizes[0]);
    float angle45 = (float)(1.0 / sqrt(2.0));

    for (int si = 0; si < ns; si++) {
        int n = sizes[si];
        float *x = alloc_vec(n+1), *y = alloc_vec(n+1);
        fill_seq(x, n, -1.0f, 0.3f);
        fill_seq(y, n,  0.5f, 0.2f);

        test_srot_case(n, x, y, angle45, angle45);  /* 45° */
        test_srot_case(n, x, y, 1.0f, 0.0f);         /* identity */
        test_srot_case(n, x, y, 0.0f, 1.0f);         /* 90° */
        test_srot_case(n, x, y, -1.0f, 0.0f);        /* 180° */

        free(x); free(y);
    }
}

/* -----------------------------------------------------------------------
 * Main
 * --------------------------------------------------------------------- */
int main(void)
{
    printf("Week 4 BLAS 1 Correctness Tests\n");
    printf("================================\n\n");

    test_sscal();
    test_scopy();
    test_sswap();
    test_saxpy();
    test_sdot();
    test_snrm2();
    test_sasum();
    test_isamax();
    test_srot();

    printf("\n================================\n");
    if (g_fail == 0) {
        printf("ALL %d TESTS PASSED\n", g_pass);
    } else {
        printf("RESULTS: %d passed, %d FAILED\n", g_pass, g_fail);
    }
    return g_fail > 0 ? 1 : 0;
}
