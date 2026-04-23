/*
 * test_blas23.c — Week 5 & 6: Automated tester for BLAS 2 & 3 kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "../include/blas2.h"
#include "../include/blas3.h"

static int g_pass = 0, g_fail = 0;

static void report(const char *name, int n, int ok) {
    if (ok) { g_pass++; }
    else { g_fail++; printf("  [FAIL] %s (n=%d)\n", name, n); }
}

static float *alloc_vec(int n) {
    float *p = (float *)aligned_alloc(64, (n + 16) * sizeof(float));
    return p;
}

static float *alloc_mat(int m, int n) {
    float *p = (float *)aligned_alloc(64, (size_t)m * n * sizeof(float));
    return p;
}

static int vec_close(const float *a, const float *b, int n, int inc) {
    for (int i = 0; i < n; i++)
        if (fabsf(a[i*inc] - b[i*inc]) > 1e-4f) return 0;
    return 1;
}

static int mat_close(const float *a, const float *b, int m, int n, int lda) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (fabsf(a[i*lda + j] - b[i*lda + j]) > 1e-3f) return 0;
    return 1;
}

void test_sgemv() {
    int m = 100, n = 120;
    float *a = alloc_mat(m, n), *x = alloc_vec(n), *y_ref = alloc_vec(m), *y_our = alloc_vec(m);
    for(int i=0; i<m*n; i++) a[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<n; i++) x[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<m; i++) y_ref[i] = y_our[i] = (float)rand()/RAND_MAX;

    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.5f, a, n, x, 1, 0.5f, y_ref, 1);
    sgemv('N', m, n, 1.5f, a, n, x, 1, 0.5f, y_our, 1);
    report("sgemv", m, vec_close(y_ref, y_our, m, 1));
    free(a); free(x); free(y_ref); free(y_our);
}

void test_sger() {
    int m = 50, n = 60;
    float *a_ref = alloc_mat(m, n), *a_our = alloc_mat(m, n), *x = alloc_vec(m), *y = alloc_vec(n);
    for(int i=0; i<m*n; i++) a_ref[i] = a_our[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<m; i++) x[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<n; i++) y[i] = (float)rand()/RAND_MAX;

    cblas_sger(CblasRowMajor, m, n, 2.0f, x, 1, y, 1, a_ref, n);
    sger(m, n, 2.0f, x, 1, y, 1, a_our, n);
    report("sger", m, mat_close(a_ref, a_our, m, n, n));
    free(a_ref); free(a_our); free(x); free(y);
}

void test_ssyrk() {
    int n = 64, k = 32;
    float *a = alloc_mat(n, k), *c_ref = alloc_mat(n, n), *c_our = alloc_mat(n, n);
    for(int i=0; i<n*k; i++) a[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<n*n; i++) c_ref[i] = c_our[i] = (float)rand()/RAND_MAX;

    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, n, k, 1.2f, a, k, 0.8f, c_ref, n);
    ssyrk('L', 'N', n, k, 1.2f, a, k, 0.8f, c_our, n);
    report("ssyrk", n, mat_close(c_ref, c_our, n, n, n));
    free(a); free(c_ref); free(c_our);
}

int main() {
    printf("Testing BLAS 2 & 3 Kernels...\n");
    test_sgemv();
    test_sger();
    test_ssyrk();
    printf("Results: %d PASS, %d FAIL\n", g_pass, g_fail);
    return g_fail > 0;
}
