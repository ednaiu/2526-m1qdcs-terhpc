/*
 * blas1.c — Week 4/5: Optimised BLAS 1 kernels
 *
 * Optimisation layers:
 *   incx==1  : AVX2 vectorised (8-wide FMA, 2-/4-way accumulator unrolling)
 *   incx>1   : AVX2 gather (non-unit stride, _mm256_i32gather_ps)
 *              — ~2-4× faster than scalar for read-heavy kernels
 *   fallback : scalar reference
 */
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include "../include/blas1.h"

/* Prefetch distance tuned by agent loop (0 = disabled, no benefit found) */
#define PREFETCH_DIST 0

/* -----------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------- */
static inline float hsum256(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}

/* Build stride-based gather index {0, inc, 2*inc, ..., 7*inc} */
static inline __m256i make_gather_idx(int inc) {
    return _mm256_mullo_epi32(
        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
        _mm256_set1_epi32(inc));
}

/* -----------------------------------------------------------------------
 * SSCAL: x *= alpha
 * --------------------------------------------------------------------- */
void sscal_scalar(int n, float alpha, float *x, int incx) {
    for (int i = 0; i < n; i++) x[i * incx] *= alpha;
}

void sscal_avx2(int n, float alpha, float *x, int incx) {
    if (n <= 0) return;
    if (incx == 1) {
        __m256 va = _mm256_set1_ps(alpha);
        int i = 0;
        for (; i <= n - 8; i += 8)
            _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), va));
        for (; i < n; i++) x[i] *= alpha;
    } else {
        sscal_scalar(n, alpha, x, incx);
    }
}
void sscal(int n, float alpha, float *x, int incx) { sscal_avx2(n, alpha, x, incx); }

/* -----------------------------------------------------------------------
 * SCOPY: y = x
 * --------------------------------------------------------------------- */
void scopy_scalar(int n, const float *x, int incx, float *y, int incy) {
    for (int i = 0; i < n; i++) y[i * incy] = x[i * incx];
}

void scopy_avx2(int n, const float *x, int incx, float *y, int incy) {
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        int i = 0;
        for (; i <= n - 8; i += 8)
            _mm256_storeu_ps(y + i, _mm256_loadu_ps(x + i));
        for (; i < n; i++) y[i] = x[i];
    } else {
        scopy_scalar(n, x, incx, y, incy);
    }
}
void scopy(int n, const float *x, int incx, float *y, int incy) { scopy_avx2(n, x, incx, y, incy); }

/* -----------------------------------------------------------------------
 * SSWAP: x <-> y
 * --------------------------------------------------------------------- */
void sswap_scalar(int n, float *x, int incx, float *y, int incy) {
    for (int i = 0; i < n; i++) {
        float t = x[i * incx]; x[i * incx] = y[i * incy]; y[i * incy] = t;
    }
}

void sswap_avx2(int n, float *x, int incx, float *y, int incy) {
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        int i = 0;
        for (; i <= n - 8; i += 8) {
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

/* -----------------------------------------------------------------------
 * SAXPY: y += alpha*x
 *
 * Non-unit stride: gather x (read-only stride), store y contiguously
 * when incy==1 — the most common mixed-stride call pattern.
 * --------------------------------------------------------------------- */
void saxpy_scalar(int n, float alpha, const float *x, int incx, float *y, int incy) {
    for (int i = 0; i < n; i++) y[i * incy] += alpha * x[i * incx];
}

void saxpy_avx2(int n, float alpha, const float *x, int incx, float *y, int incy) {
    if (n <= 0 || alpha == 0.0f) return;
    __m256 va = _mm256_set1_ps(alpha);
    if (incx == 1 && incy == 1) {
        int i = 0;
        for (; i <= n - 8; i += 8)
            _mm256_storeu_ps(y + i,
                _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
        for (; i < n; i++) y[i] += alpha * x[i];
    } else if (incx > 1 && incy == 1) {
        /* Gather x, store y contiguous */
        __m256i vidx = make_gather_idx(incx);
        int i = 0;
        for (; i <= n - 8; i += 8)
            _mm256_storeu_ps(y + i,
                _mm256_fmadd_ps(va,
                    _mm256_i32gather_ps(x + (size_t)i * incx, vidx, 4),
                    _mm256_loadu_ps(y + i)));
        for (; i < n; i++) y[i] += alpha * x[i * incx];
    } else {
        saxpy_scalar(n, alpha, x, incx, y, incy);
    }
}
void saxpy(int n, float alpha, const float *x, int incx, float *y, int incy) {
    saxpy_avx2(n, alpha, x, incx, y, incy);
}

/* -----------------------------------------------------------------------
 * SDOT: return sum x[i]*y[i]
 *
 * incx==incy==1 : 4-way unrolled AVX2 (hides 5-cycle FMA latency)
 * non-unit stride: gather both x and y
 * --------------------------------------------------------------------- */
float sdot_scalar(int n, const float *x, int incx, const float *y, int incy) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i * incx] * y[i * incy];
    return s;
}

float sdot_avx2(int n, const float *x, int incx, const float *y, int incy) {
    if (n <= 0) return 0.0f;
    if (incx == 1 && incy == 1) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 32; i += 32) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),    _mm256_loadu_ps(y+i),    a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),  _mm256_loadu_ps(y+i+8),  a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+16), _mm256_loadu_ps(y+i+16), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+24), _mm256_loadu_ps(y+i+24), a3);
        }
        a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        float s = hsum256(a0);
        for (; i < n; i++) s += x[i] * y[i];
        return s;
    } else if (incx > 0 && incy > 0 && incx <= 1024 && incy <= 1024) {
        /* Gather both vectors — faster than scalar for large n */
        __m256i vix = make_gather_idx(incx);
        __m256i viy = make_gather_idx(incy);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 16; i += 16) {
            acc0 = _mm256_fmadd_ps(
                _mm256_i32gather_ps(x + (size_t)i * incx, vix, 4),
                _mm256_i32gather_ps(y + (size_t)i * incy, viy, 4), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_i32gather_ps(x + (size_t)(i+8) * incx, vix, 4),
                _mm256_i32gather_ps(y + (size_t)(i+8) * incy, viy, 4), acc1);
        }
        float s = hsum256(_mm256_add_ps(acc0, acc1));
        for (; i < n; i++) s += x[i * incx] * y[i * incy];
        return s;
    } else {
        return sdot_scalar(n, x, incx, y, incy);
    }
}
float sdot(int n, const float *x, int incx, const float *y, int incy) {
    return sdot_avx2(n, x, incx, y, incy);
}

/* -----------------------------------------------------------------------
 * SNRM2: return ||x||_2
 *
 * incx==1   : 2-way accumulator AVX2
 * incx>1    : gather x
 * --------------------------------------------------------------------- */
float snrm2_scalar(int n, const float *x, int incx) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) { float xi = x[i * incx]; s += xi * xi; }
    return sqrtf(s);
}

float snrm2_avx2(int n, const float *x, int incx) {
    if (n <= 0) return 0.0f;
    if (incx == 1) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 16; i += 16) {
            __m256 v0 = _mm256_loadu_ps(x + i);
            __m256 v1 = _mm256_loadu_ps(x + i + 8);
            a0 = _mm256_fmadd_ps(v0, v0, a0);
            a1 = _mm256_fmadd_ps(v1, v1, a1);
        }
        float s = hsum256(_mm256_add_ps(a0, a1));
        for (; i < n; i++) s += x[i] * x[i];
        return sqrtf(s);
    } else if (incx > 0 && incx <= 1024) {
        __m256i vidx = make_gather_idx(incx);
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 v = _mm256_i32gather_ps(x + (size_t)i * incx, vidx, 4);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        float s = hsum256(acc);
        for (; i < n; i++) { float xi = x[i * incx]; s += xi * xi; }
        return sqrtf(s);
    } else {
        return snrm2_scalar(n, x, incx);
    }
}
float snrm2(int n, const float *x, int incx) { return snrm2_avx2(n, x, incx); }

/* -----------------------------------------------------------------------
 * SASUM: return sum |x[i]|
 *
 * incx==1 : AVX2 abs via bitmask
 * incx>1  : gather + abs
 * --------------------------------------------------------------------- */
float sasum_scalar(int n, const float *x, int incx) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf(x[i * incx]);
    return s;
}

float sasum_avx2(int n, const float *x, int incx) {
    if (n <= 0) return 0.0f;
    __m256 smask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    if (incx == 1) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8)
            acc = _mm256_add_ps(acc, _mm256_and_ps(_mm256_loadu_ps(x + i), smask));
        float s = hsum256(acc);
        for (; i < n; i++) s += fabsf(x[i]);
        return s;
    } else if (incx > 0 && incx <= 1024) {
        __m256i vidx = make_gather_idx(incx);
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8)
            acc = _mm256_add_ps(acc,
                _mm256_and_ps(_mm256_i32gather_ps(x + (size_t)i * incx, vidx, 4), smask));
        float s = hsum256(acc);
        for (; i < n; i++) s += fabsf(x[i * incx]);
        return s;
    } else {
        return sasum_scalar(n, x, incx);
    }
}
float sasum(int n, const float *x, int incx) { return sasum_avx2(n, x, incx); }

/* -----------------------------------------------------------------------
 * ISAMAX: return index of max |x[i]|
 *
 * AVX2 find 8-wide max, then scalar over chunks to track index.
 * Non-unit stride: gather + same approach.
 * --------------------------------------------------------------------- */
int isamax_scalar(int n, const float *x, int incx) {
    if (n <= 0) return -1;
    int idx = 0; float maxv = fabsf(x[0]);
    for (int i = 1; i < n; i++) {
        float v = fabsf(x[i * incx]);
        if (v > maxv) { maxv = v; idx = i; }
    }
    return idx;
}

int isamax_avx2(int n, const float *x, int incx) {
    if (n <= 0) return -1;
    if (n < 16) return isamax_scalar(n, x, incx);

    __m256 smask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    /* Pass 1: find global max value via AVX2 */
    __m256 vmax = _mm256_setzero_ps();
    int i = 0;
    if (incx == 1) {
        for (; i <= n - 8; i += 8)
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x + i), smask));
    } else if (incx > 0 && incx <= 1024) {
        __m256i vidx = make_gather_idx(incx);
        for (; i <= n - 8; i += 8)
            vmax = _mm256_max_ps(vmax,
                _mm256_and_ps(_mm256_i32gather_ps(x + (size_t)i * incx, vidx, 4), smask));
    }
    /* Horizontal max */
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 m4 = _mm_max_ps(lo, hi);
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    float global_max = _mm_cvtss_f32(m4);
    /* scalar tail */
    for (; i < n; i++) {
        float v = fabsf(x[i * incx]);
        if (v > global_max) global_max = v;
    }

    /* Pass 2: find first index achieving that max */
    for (int j = 0; j < n; j++)
        if (fabsf(x[j * incx]) == global_max) return j;
    return 0;
}
int isamax(int n, const float *x, int incx) { return isamax_avx2(n, x, incx); }

/* -----------------------------------------------------------------------
 * SROT: Givens rotation
 * --------------------------------------------------------------------- */
void srot_scalar(int n, float *x, int incx, float *y, int incy, float c, float s) {
    for (int i = 0; i < n; i++) {
        float xi = x[i * incx], yi = y[i * incy];
        x[i * incx] =  c * xi + s * yi;
        y[i * incy] = -s * xi + c * yi;
    }
}

void srot_avx2(int n, float *x, int incx, float *y, int incy, float c, float s) {
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        __m256 vc = _mm256_set1_ps(c), vs = _mm256_set1_ps(s), vms = _mm256_set1_ps(-s);
        int i = 0;
        for (; i <= n - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i), vy = _mm256_loadu_ps(y + i);
            _mm256_storeu_ps(x + i, _mm256_fmadd_ps(vc, vx, _mm256_mul_ps(vs, vy)));
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(vms, vx, _mm256_mul_ps(vc, vy)));
        }
        for (; i < n; i++) {
            float xi = x[i], yi = y[i];
            x[i] =  c * xi + s * yi;
            y[i] = -s * xi + c * yi;
        }
    } else {
        srot_scalar(n, x, incx, y, incy, c, s);
    }
}
void srot(int n, float *x, int incx, float *y, int incy, float c, float s) {
    srot_avx2(n, x, incx, y, incy, c, s);
}

/* -----------------------------------------------------------------------
 * Fortran BLAS entry points (trailing _, args by pointer)
 * --------------------------------------------------------------------- */
void sscal_(int *n, float *alpha, float *x, int *incx)
    { sscal(*n, *alpha, x, *incx); }
void scopy_(int *n, const float *x, int *incx, float *y, int *incy)
    { scopy(*n, x, *incx, y, *incy); }
void sswap_(int *n, float *x, int *incx, float *y, int *incy)
    { sswap(*n, x, *incx, y, *incy); }
void saxpy_(int *n, float *alpha, const float *x, int *incx, float *y, int *incy)
    { saxpy(*n, *alpha, x, *incx, y, *incy); }
float sdot_ (int *n, const float *x, int *incx, const float *y, int *incy)
    { return sdot(*n, x, *incx, y, *incy); }
float snrm2_(int *n, const float *x, int *incx)
    { return snrm2(*n, x, *incx); }
float sasum_(int *n, const float *x, int *incx)
    { return sasum(*n, x, *incx); }
int   isamax_(int *n, const float *x, int *incx)
    { return isamax(*n, x, *incx) + 1; }
void srot_  (int *n, float *x, int *incx, float *y, int *incy, float *c, float *s)
    { srot(*n, x, *incx, y, *incy, *c, *s); }
