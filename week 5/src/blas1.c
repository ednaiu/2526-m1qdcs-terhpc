/*
 * blas1.c — Week 4: Optimised BLAS 1 kernels (COMPLETE TEMPLATE)
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include "../include/blas1.h"

#define PREFETCH_DIST 0

static inline float hsum256(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}

void sscal_scalar(int n, float alpha, float *x, int incx) { 
    if (incx == 1) {
        for (int i = 0; i < n; i++) x[i] *= alpha; 
    } else {
        for (int i = 0; i < n; i++) x[i * incx] *= alpha;
    }
}

void sscal_avx2(int n, float alpha, float *x, int incx) {
    if (n <= 0) return;
    if (incx == 1) {
        __m256 va = _mm256_set1_ps(alpha);
        int i = 0;
        for (; i <= n - 8; i += 8) {
            _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
            __m256 vx = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_mul_ps(vx, va));
        }
        for (; i < n; i++) x[i] *= alpha;
    } else {
        sscal_scalar(n, alpha, x, incx);
    }
}
void sscal(int n, float alpha, float *x, int incx) { sscal_avx2(n, alpha, x, incx); }

void scopy_scalar(int n, const float *x, int incx, float *y, int incy) {
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) y[i] = x[i];
    } else {
        for (int i = 0; i < n; i++) y[i * incy] = x[i * incx];
    }
}

void scopy_avx2(int n, const float *x, int incx, float *y, int incy) {
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        int i = 0;
        for (; i <= n - 8; i += 8) {
            _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
            _mm256_storeu_ps(y + i, _mm256_loadu_ps(x + i));
        }
        for (; i < n; i++) y[i] = x[i];
    } else {
        scopy_scalar(n, x, incx, y, incy);
    }
}
void scopy(int n, const float *x, int incx, float *y, int incy) { scopy_avx2(n, x, incx, y, incy); }

void sswap_scalar(int n, float *x, int incx, float *y, int incy) {
    for (int i = 0; i < n; i++) {
        float t = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = t;
    }
}

void sswap_avx2(int n, float *x, int incx, float *y, int incy) {
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        int i = 0;
        for (; i <= n - 8; i += 8) {
            _mm_prefetch((const char *)(x + i + PREFETCH_DIST), _MM_HINT_T0);
            _mm_prefetch((const char *)(y + i + PREFETCH_DIST), _MM_HINT_T0);
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);
            _mm256_storeu_ps(x + i, vy);
            _mm256_storeu_ps(y + i, vx);
        }
        for (; i < n; i++) { float t = x[i]; x[i] = y[i]; y[i] = t; }
    } else {
        sswap_scalar(n, x, incx, y, incy);
    }
}
void sswap(int n, float *x, int incx, float *y, int incy) { sswap_avx2(n, x, incx, y, incy); }

void saxpy_scalar(int n, float alpha, const float *x, int incx, float *y, int incy) {
    for (int i = 0; i < n; i++) y[i * incy] += alpha * x[i * incx];
}

void saxpy_avx2(int n, float alpha, const float *x, int incx, float *y, int incy) {
    if (n <= 0 || alpha == 0.0f) return;
    if (incx == 1 && incy == 1) {
        __m256 va = _mm256_set1_ps(alpha);
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 vx0 = _mm256_loadu_ps(x + i);
            __m256 vy0 = _mm256_loadu_ps(y + i);
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx0, vy0));
        }
        for (; i < n; i++) y[i] += alpha * x[i];
    } else {
        saxpy_scalar(n, alpha, x, incx, y, incy);
    }
}
void saxpy(int n, float alpha, const float *x, int incx, float *y, int incy) { saxpy_avx2(n, alpha, x, incx, y, incy); }

/* sdot */
float sdot_scalar(int n, const float *x, int incx, const float *y, int incy) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i * incx] * y[i * incy];
    return s;
}

float sdot_avx2(int n, const float *x, int incx, const float *y, int incy) {
    if (n <= 0) return 0.0f;
    if (incx == 1 && incy == 1) {
        __m256 acc0 = _mm256_setzero_ps(); __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(); __m256 acc3 = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 32; i += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),    _mm256_loadu_ps(y+i),    acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),  _mm256_loadu_ps(y+i+8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+16), _mm256_loadu_ps(y+i+16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+24), _mm256_loadu_ps(y+i+24), acc3);
        }
        acc0 = _mm256_add_ps(acc0, acc1); acc2 = _mm256_add_ps(acc2, acc3); acc0 = _mm256_add_ps(acc0, acc2);
        float s = hsum256(acc0); for (; i < n; i++) s += x[i] * y[i]; return s;
    } else {
        return sdot_scalar(n, x, incx, y, incy);
    }
}
float sdot(int n, const float *x, int incx, const float *y, int incy) { return sdot_avx2(n, x, incx, y, incy); }

/* snrm2 */
float snrm2_scalar(int n, const float *x, int incx) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) { float xi = x[i * incx]; s += xi * xi; }
    return sqrtf(s);
}

float snrm2_avx2(int n, const float *x, int incx) {
    if (n <= 0) return 0.0f;
    if (incx == 1) {
        __m256 acc0 = _mm256_setzero_ps(); __m256 acc1 = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 16; i += 16) {
            __m256 v0 = _mm256_loadu_ps(x + i); __m256 v1 = _mm256_loadu_ps(x + i + 8);
            acc0 = _mm256_fmadd_ps(v0, v0, acc0); acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }
        float s = hsum256(_mm256_add_ps(acc0, acc1)); for (; i < n; i++) s += x[i] * x[i]; return sqrtf(s);
    } else {
        return snrm2_scalar(n, x, incx);
    }
}
float snrm2(int n, const float *x, int incx) { return snrm2_avx2(n, x, incx); }

/* sasum */
float sasum_scalar(int n, const float *x, int incx) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf(x[i * incx]);
    return s;
}

float sasum_avx2(int n, const float *x, int incx) {
    if (n <= 0) return 0.0f;
    if (incx == 1) {
        __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        __m256 acc0 = _mm256_setzero_ps(); int i = 0;
        for (; i <= n - 8; i += 8) { acc0 = _mm256_add_ps(acc0, _mm256_and_ps(_mm256_loadu_ps(x + i), sign_mask)); }
        float s = hsum256(acc0); for (; i < n; i++) s += fabsf(x[i]); return s;
    } else {
        return sasum_scalar(n, x, incx);
    }
}
float sasum(int n, const float *x, int incx) { return sasum_avx2(n, x, incx); }

/* isamax */
int isamax_scalar(int n, const float *x, int incx) {
    if (n <= 0) return -1;
    int idx = 0;
    float maxv = fabsf(x[0]);
    for (int i = 1; i < n; i++) {
        float v = fabsf(x[i * incx]);
        if (v > maxv) { maxv = v; idx = i; }
    }
    return idx;
}

int isamax_avx2(int n, const float *x, int incx) { return isamax_scalar(n, x, incx); }
int isamax(int n, const float *x, int incx) { return isamax_scalar(n, x, incx); }

/* srot */
void srot_scalar(int n, float *x, int incx, float *y, int incy, float c, float s) {
    for (int i = 0; i < n; i++) {
        float xi = x[i * incx], yi = y[i * incy];
        x[i * incx] = c * xi + s * yi;
        y[i * incy] = -s * xi + c * yi;
    }
}

void srot_avx2(int n, float *x, int incx, float *y, int incy, float c, float s) {
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        __m256 vc = _mm256_set1_ps(c);
        __m256 vs = _mm256_set1_ps(s);
        __m256 vms = _mm256_set1_ps(-s);
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_loadu_ps(y + i);
            _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_mul_ps(vc, vx), _mm256_mul_ps(vs, vy)));
            _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_mul_ps(vms, vx), _mm256_mul_ps(vc, vy)));
        }
        for (; i < n; i++) {
            float xi = x[i], yi = y[i];
            x[i] = c * xi + s * yi;
            y[i] = -s * xi + c * yi;
        }
    } else {
        srot_scalar(n, x, incx, y, incy, c, s);
    }
}
void srot(int n, float *x, int incx, float *y, int incy, float c, float s) { srot_avx2(n, x, incx, y, incy, c, s); }

/* Fortran entries */
void sscal_(int *n, float *alpha, float *x, int *incx) { sscal(*n, *alpha, x, *incx); }
void scopy_(int *n, const float *x, int *incx, float *y, int *incy) { scopy(*n, x, *incx, y, *incy); }
void sswap_(int *n, float *x, int *incx, float *y, int *incy) { sswap(*n, x, *incx, y, *incy); }
void saxpy_(int *n, float *alpha, const float *x, int *incx, float *y, int *incy) { saxpy(*n, *alpha, x, *incx, y, *incy); }
float sdot_(int *n, const float *x, int *incx, const float *y, int *incy) { return sdot(*n, x, *incx, y, *incy); }
float snrm2_(int *n, const float *x, int *incx) { return snrm2(*n, x, *incx); }
float sasum_(int *n, const float *x, int *incx) { return sasum(*n, x, *incx); }
int isamax_(int *n, const float *x, int *incx) { return isamax(*n, x, *incx) + 1; }
void srot_(int *n, float *x, int *incx, float *y, int *incy, float *c, float *s) { srot(*n, x, *incx, y, *incy, *c, *s); }
