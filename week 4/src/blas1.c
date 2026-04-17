/*
 * blas1.c — Week 4: Optimised BLAS 1 kernels
 *
 * Kernels: sscal, scopy, sswap, saxpy, sdot, snrm2, sasum, isamax, srot
 *
 * Optimisation layers (measured separately in bench_blas1):
 *   1. Scalar baseline  (_scalar suffix)
 *   2. AVX2 + prefetch  (_avx2 suffix)
 *   3. AVX2 + prefetch + OpenMP  (no suffix, default API)
 *
 * Fortran entry points (trailing underscore) are at the bottom.
 *
 * Assumptions: incX = incY = 1.
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#include "../include/blas1.h"

/* Prefetch distance in floats (2 cache lines ahead = 32 floats) */
#define PREFETCH_DIST 32

/* ============================================================
 * Helper: horizontal sum of 8 floats in an __m256
 * ========================================================== */
static inline float hsum256(__m256 v)
{
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}

/* ============================================================
 * sscal — x[i] *= alpha
 * ============================================================ */

void sscal_scalar(int n, float alpha, float *x)
{
    for (int i = 0; i < n; i++)
        x[i] *= alpha;
}

void sscal_avx2(int n, float alpha, float *x)
{
    if (n <= 0) return;
    __m256 va = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, va));
    }
    for (; i < n; i++)
        x[i] *= alpha;
}

void sscal(int n, float alpha, float *x)
{
    if (n <= 0) return;
    if (alpha == 1.0f) return;
    __m256 va = _mm256_set1_ps(alpha);
    #pragma omp parallel for schedule(static) if(n > 4096)
    for (int i = 0; i < (n & ~7); i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, va));
    }
    /* scalar tail (single thread) */
    for (int i = n & ~7; i < n; i++)
        x[i] *= alpha;
}

/* ============================================================
 * scopy — y[i] = x[i]
 * ============================================================ */

void scopy_scalar(int n, const float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i];
}

void scopy_avx2(int n, const float *x, float *y)
{
    int i = 0;
    for (; i <= n - 8; i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm256_storeu_ps(y + i, _mm256_loadu_ps(x + i));
    }
    for (; i < n; i++)
        y[i] = x[i];
}

void scopy(int n, const float *x, float *y)
{
    if (n <= 0) return;
    #pragma omp parallel for schedule(static) if(n > 4096)
    for (int i = 0; i < (n & ~7); i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm256_storeu_ps(y + i, _mm256_loadu_ps(x + i));
    }
    for (int i = n & ~7; i < n; i++)
        y[i] = x[i];
}

/* ============================================================
 * sswap — x[i] <-> y[i]
 * ============================================================ */

void sswap_scalar(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++) {
        float t = x[i]; x[i] = y[i]; y[i] = t;
    }
}

void sswap_avx2(int n, float *x, float *y)
{
    int i = 0;
    for (; i <= n - 8; i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(x + i, vy);
        _mm256_storeu_ps(y + i, vx);
    }
    for (; i < n; i++) {
        float t = x[i]; x[i] = y[i]; y[i] = t;
    }
}

void sswap(int n, float *x, float *y)
{
    if (n <= 0) return;
    #pragma omp parallel for schedule(static) if(n > 4096)
    for (int i = 0; i < (n & ~7); i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(x + i, vy);
        _mm256_storeu_ps(y + i, vx);
    }
    for (int i = n & ~7; i < n; i++) {
        float t = x[i]; x[i] = y[i]; y[i] = t;
    }
}

/* ============================================================
 * saxpy — y[i] += alpha * x[i]
 * ============================================================ */

void saxpy_scalar(int n, float alpha, const float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] += alpha * x[i];
}

void saxpy_avx2(int n, float alpha, const float *x, float *y)
{
    if (n <= 0) return;
    __m256 va = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
    }
    for (; i < n; i++)
        y[i] += alpha * x[i];
}

void saxpy(int n, float alpha, const float *x, float *y)
{
    if (n <= 0) return;
    if (alpha == 0.0f) return;
    __m256 va = _mm256_set1_ps(alpha);
    #pragma omp parallel for schedule(static) if(n > 4096)
    for (int i = 0; i < (n & ~7); i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
    }
    for (int i = n & ~7; i < n; i++)
        y[i] += alpha * x[i];
}

/* ============================================================
 * sdot — return sum(x[i]*y[i])
 * ============================================================ */

float sdot_scalar(int n, const float *x, const float *y)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}

float sdot_avx2(int n, const float *x, const float *y)
{
    if (n <= 0) return 0.0f;
    /* 4 independent accumulators to hide FMA latency */
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),    _mm256_loadu_ps(y+i),    acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),  _mm256_loadu_ps(y+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+16), _mm256_loadu_ps(y+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+24), _mm256_loadu_ps(y+i+24), acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    float s = hsum256(acc0);
    for (; i < n; i++) s += x[i] * y[i];
    return s;
}

float sdot(int n, const float *x, const float *y)
{
    if (n <= 0) return 0.0f;
    float total = 0.0f;
    #pragma omp parallel reduction(+:total) if(n > 4096)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < (n & ~15); i += 16) {
            _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
            _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),   _mm256_loadu_ps(y+i),   acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8), _mm256_loadu_ps(y+i+8), acc1);
        }
        total += hsum256(_mm256_add_ps(acc0, acc1));
    }
    /* scalar tail */
    for (int i = n & ~15; i < n; i++) total += x[i] * y[i];
    return total;
}

/* ============================================================
 * snrm2 — return sqrt(sum(x[i]^2))
 * ============================================================ */

float snrm2_scalar(int n, const float *x)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return sqrtf(s);
}

float snrm2_avx2(int n, const float *x)
{
    if (n <= 0) return 0.0f;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 v0 = _mm256_loadu_ps(x+i);
        __m256 v1 = _mm256_loadu_ps(x+i+8);
        __m256 v2 = _mm256_loadu_ps(x+i+16);
        __m256 v3 = _mm256_loadu_ps(x+i+24);
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        acc2 = _mm256_fmadd_ps(v2, v2, acc2);
        acc3 = _mm256_fmadd_ps(v3, v3, acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float s = hsum256(acc0);
    for (; i < n; i++) s += x[i] * x[i];
    return sqrtf(s);
}

float snrm2(int n, const float *x)
{
    if (n <= 0) return 0.0f;
    float total = 0.0f;
    #pragma omp parallel reduction(+:total) if(n > 4096)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < (n & ~15); i += 16) {
            _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
            __m256 v0 = _mm256_loadu_ps(x+i);
            __m256 v1 = _mm256_loadu_ps(x+i+8);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }
        total += hsum256(_mm256_add_ps(acc0, acc1));
    }
    for (int i = n & ~15; i < n; i++) total += x[i] * x[i];
    return sqrtf(total);
}

/* ============================================================
 * sasum — return sum(|x[i]|)
 * ============================================================ */

float sasum_scalar(int n, const float *x)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf(x[i]);
    return s;
}

float sasum_avx2(int n, const float *x)
{
    if (n <= 0) return 0.0f;
    /* sign-bit mask: clear bit 31 of each float = absolute value */
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        acc0 = _mm256_add_ps(acc0, _mm256_and_ps(_mm256_loadu_ps(x+i),    sign_mask));
        acc1 = _mm256_add_ps(acc1, _mm256_and_ps(_mm256_loadu_ps(x+i+8),  sign_mask));
        acc2 = _mm256_add_ps(acc2, _mm256_and_ps(_mm256_loadu_ps(x+i+16), sign_mask));
        acc3 = _mm256_add_ps(acc3, _mm256_and_ps(_mm256_loadu_ps(x+i+24), sign_mask));
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float s = hsum256(acc0);
    for (; i < n; i++) s += fabsf(x[i]);
    return s;
}

float sasum(int n, const float *x)
{
    if (n <= 0) return 0.0f;
    float total = 0.0f;
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    #pragma omp parallel reduction(+:total) if(n > 4096)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < (n & ~15); i += 16) {
            _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
            acc0 = _mm256_add_ps(acc0, _mm256_and_ps(_mm256_loadu_ps(x+i),   sign_mask));
            acc1 = _mm256_add_ps(acc1, _mm256_and_ps(_mm256_loadu_ps(x+i+8), sign_mask));
        }
        total += hsum256(_mm256_add_ps(acc0, acc1));
    }
    for (int i = n & ~15; i < n; i++) total += fabsf(x[i]);
    return total;
}

/* ============================================================
 * isamax — index of maximum |x[i]| (0-based C index)
 * ============================================================ */

int isamax_scalar(int n, const float *x)
{
    if (n <= 0) return -1;
    int idx = 0;
    float maxv = fabsf(x[0]);
    for (int i = 1; i < n; i++) {
        float v = fabsf(x[i]);
        if (v > maxv) { maxv = v; idx = i; }
    }
    return idx;
}

int isamax_avx2(int n, const float *x)
{
    if (n <= 0) return -1;
    if (n == 1) return 0;

    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 vmax = _mm256_setzero_ps();

    int i = 0;
    for (; i <= n - 8; i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 v = _mm256_and_ps(_mm256_loadu_ps(x + i), sign_mask);
        vmax = _mm256_max_ps(vmax, v);
    }

    /* Horizontal max of vmax */
    float tmp[8];
    _mm256_storeu_ps(tmp, vmax);
    float global_max = 0.0f;
    for (int j = 0; j < 8; j++)
        if (tmp[j] > global_max) global_max = tmp[j];
    /* Catch remaining elements */
    for (; i < n; i++) {
        float v = fabsf(x[i]);
        if (v > global_max) global_max = v;
    }

    /* Single scalar pass to find first index with that max value */
    for (int j = 0; j < n; j++) {
        if (fabsf(x[j]) >= global_max) return j;
    }
    return 0;
}

int isamax(int n, const float *x)
{
    if (n <= 0) return -1;
    if (n == 1) return 0;
    /* For correctness: parallel requires care (first occurrence of max).
     * Strategy: each thread finds its local argmax, then we reduce.  */
    int global_idx = 0;
    float global_max = fabsf(x[0]);

    #pragma omp parallel if(n > 4096)
    {
        int local_idx = 0;
        float local_max = 0.0f;
        __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; i++) {
            float v = fabsf(x[i]);
            if (v > local_max) { local_max = v; local_idx = i; }
        }

        #pragma omp critical
        {
            if (local_max > global_max ||
                (local_max == global_max && local_idx < global_idx)) {
                global_max = local_max;
                global_idx = local_idx;
            }
        }
        (void)sign_mask;
    }
    return global_idx;
}

/* ============================================================
 * srot — Givens rotation
 *   (x[i], y[i]) = (c*x[i] + s*y[i], -s*x[i] + c*y[i])
 * ============================================================ */

void srot_scalar(int n, float *x, float *y, float c, float s)
{
    for (int i = 0; i < n; i++) {
        float xi = x[i], yi = y[i];
        x[i] =  c * xi + s * yi;
        y[i] = -s * xi + c * yi;
    }
}

void srot_avx2(int n, float *x, float *y, float c, float s)
{
    if (n <= 0) return;
    __m256 vc =  _mm256_set1_ps(c);
    __m256 vs =  _mm256_set1_ps(s);
    __m256 vns = _mm256_set1_ps(-s);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        /* new_x =  c*x + s*y */
        /* new_y = -s*x + c*y */
        __m256 nx = _mm256_fmadd_ps(vc, vx, _mm256_mul_ps(vs,  vy));
        __m256 ny = _mm256_fmadd_ps(vc, vy, _mm256_mul_ps(vns, vx));
        _mm256_storeu_ps(x + i, nx);
        _mm256_storeu_ps(y + i, ny);
    }
    for (; i < n; i++) {
        float xi = x[i], yi = y[i];
        x[i] =  c * xi + s * yi;
        y[i] = -s * xi + c * yi;
    }
}

void srot(int n, float *x, float *y, float c, float s)
{
    if (n <= 0) return;
    __m256 vc  = _mm256_set1_ps(c);
    __m256 vs  = _mm256_set1_ps(s);
    __m256 vns = _mm256_set1_ps(-s);
    #pragma omp parallel for schedule(static) if(n > 4096)
    for (int i = 0; i < (n & ~7); i += 8) {
        _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
        _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 nx = _mm256_fmadd_ps(vc, vx, _mm256_mul_ps(vs,  vy));
        __m256 ny = _mm256_fmadd_ps(vc, vy, _mm256_mul_ps(vns, vx));
        _mm256_storeu_ps(x + i, nx);
        _mm256_storeu_ps(y + i, ny);
    }
    for (int i = n & ~7; i < n; i++) {
        float xi = x[i], yi = y[i];
        x[i] =  c * xi + s * yi;
        y[i] = -s * xi + c * yi;
    }
}

/* ============================================================
 * Fortran BLAS entry points (trailing underscore, args by pointer)
 * ============================================================ */

void sscal_(int *n, float *alpha, float *x, int *incx)
{
    (void)incx; /* incx=1 assumed */
    sscal(*n, *alpha, x);
}

void scopy_(int *n, const float *x, int *incx, float *y, int *incy)
{
    (void)incx; (void)incy;
    scopy(*n, x, y);
}

void sswap_(int *n, float *x, int *incx, float *y, int *incy)
{
    (void)incx; (void)incy;
    sswap(*n, x, y);
}

void saxpy_(int *n, float *alpha, const float *x, int *incx,
            float *y, int *incy)
{
    (void)incx; (void)incy;
    saxpy(*n, *alpha, x, y);
}

float sdot_(int *n, const float *x, int *incx, const float *y, int *incy)
{
    (void)incx; (void)incy;
    return sdot(*n, x, y);
}

float snrm2_(int *n, const float *x, int *incx)
{
    (void)incx;
    return snrm2(*n, x);
}

float sasum_(int *n, const float *x, int *incx)
{
    (void)incx;
    return sasum(*n, x);
}

/* Fortran isamax is 1-based; we convert from 0-based C index */
int isamax_(int *n, const float *x, int *incx)
{
    (void)incx;
    int idx = isamax(*n, x);
    return idx + 1;  /* Fortran 1-based */
}

void srot_(int *n, float *x, int *incx, float *y, int *incy,
           float *c, float *s)
{
    (void)incx; (void)incy;
    srot(*n, x, y, *c, *s);
}
