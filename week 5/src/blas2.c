/*
 * blas2.c — Week 5: Optimised BLAS 2 kernels
 *
 * Optimisation layers per kernel:
 *   sgemv  : AVX2 (NoTrans row-dot, Trans column-partition), OpenMP
 *   sger   : AVX2 row FMA, OpenMP
 *   ssymv  : AVX2 for stored triangle (contiguous row), scalar reflected half, OpenMP
 *   strmv  : AVX2 for non-transposed paths (contiguous row), scalar for trans
 *   strsv  : sequential by nature (dependency chain)
 *   ssyr   : AVX2 row SAXPY, OpenMP
 *   ssyr2  : AVX2 row dual-FMA, OpenMP
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <ctype.h>
#include "../include/blas2.h"

/* -----------------------------------------------------------------------
 * Horizontal sum of 8-wide AVX float vector
 * --------------------------------------------------------------------- */
static inline float hsum256(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 s   = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

/* -----------------------------------------------------------------------
 * SGEMV: y = alpha*A*x + beta*y   (row-major A)
 *
 * NoTrans: each row of A is an independent dot product → AVX2 + OpenMP.
 * Trans  : partition y across threads, scalar reduction over columns.
 * --------------------------------------------------------------------- */
void sgemv(char trans, int m, int n, float alpha, const float *a, int lda,
           const float *x, int incx, float beta, float *y, int incy)
{
    if (m <= 0 || n <= 0) return;

    int rows_y = (trans == 'N' || trans == 'n') ? m : n;

    if (beta == 0.0f) {
        for (int i = 0; i < rows_y; i++) y[i * incy] = 0.0f;
    } else if (beta != 1.0f) {
        for (int i = 0; i < rows_y; i++) y[i * incy] *= beta;
    }
    if (alpha == 0.0f) return;

    int ta = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    if (!ta) {
        /* --- NoTrans: y[i] += alpha * dot(A[i,:], x) --- */
        #pragma omp parallel
        #pragma omp single
        #pragma omp taskloop grainsize(1)
        for (int i = 0; i < m; i++) {
            const float *Ar = a + (size_t)i * lda;
            float sum = 0.0f;
            if (incx == 1) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                int j = 0;
                for (; j <= n - 16; j += 16) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j),
                                           _mm256_loadu_ps(x + j), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j + 8),
                                           _mm256_loadu_ps(x + j + 8), acc1);
                }
                for (; j <= n - 8; j += 8)
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j),
                                           _mm256_loadu_ps(x + j), acc0);
                sum = hsum256(_mm256_add_ps(acc0, acc1));
                for (; j < n; j++) sum += Ar[j] * x[j];
            } else {
                for (int j = 0; j < n; j++) sum += Ar[j] * x[j * incx];
            }
            y[i * incy] += alpha * sum;
        }
    } else {
        /* --- Trans: y[j] += alpha * sum_i A[i,j] * x[i]
         * Partition output y across threads, scalar inner loop.
         * Column access (A[i*lda+j]) is non-contiguous → no AVX2.       */
        #pragma omp parallel
        {
            int nt    = omp_get_num_threads();
            int tid   = omp_get_thread_num();
            int chunk = (n + nt - 1) / nt;
            int j0    = tid * chunk;
            int j1    = j0 + chunk; if (j1 > n) j1 = n;
            for (int j = j0; j < j1; j++) {
                float sum = 0.0f;
                for (int i = 0; i < m; i++) sum += a[(size_t)i * lda + j] * x[i * incx];
                y[j * incy] += alpha * sum;
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * SGER: A = alpha*x*y^T + A   (rank-1 update, row-major A)
 *
 * Each row i is an independent SAXPY on y → AVX2 FMA + OpenMP.
 * --------------------------------------------------------------------- */
void sger(int m, int n, float alpha, const float *x, int incx,
          const float *y, int incy, float *a, int lda)
{
    if (m <= 0 || n <= 0 || alpha == 0.0f) return;

    #pragma omp parallel
    #pragma omp single
    #pragma omp taskloop grainsize(8)
    for (int i = 0; i < m; i++) {
        float axi = alpha * x[i * incx];
        float *Ar = a + (size_t)i * lda;
        if (incy == 1) {
            __m256 vaxi = _mm256_set1_ps(axi);
            int j = 0;
            for (; j <= n - 8; j += 8)
                _mm256_storeu_ps(Ar + j,
                    _mm256_fmadd_ps(vaxi, _mm256_loadu_ps(y + j),
                                    _mm256_loadu_ps(Ar + j)));
            for (; j < n; j++) Ar[j] += axi * y[j];
        } else {
            for (int j = 0; j < n; j++) Ar[j] += axi * y[j * incy];
        }
    }
}

/* -----------------------------------------------------------------------
 * SSYMV: y = alpha*A*x + beta*y   (A symmetric, row-major stored half)
 *
 * Optimization strategy — avoid branch inside hot loop:
 *   For each row i, split into two disjoint parts:
 *     (1) Stored triangle — contiguous row access → AVX2
 *     (2) Reflected triangle — column access (stride lda) → scalar
 *
 *   UPLO='L':  stored  = a[i*lda + 0..i],  reflected = a[j*lda+i] j>i
 *   UPLO='U':  stored  = a[i*lda + i..n-1], reflected = a[j*lda+i] j<i
 *
 * This eliminates the conditional branch from the inner loop, enabling
 * full AVX2 vectorisation of ~50% of the work (stored half).
 * --------------------------------------------------------------------- */
void ssymv(char uplo, int n, float alpha, const float *a, int lda,
           const float *x, int incx, float beta, float *y, int incy)
{
    if (n <= 0) return;

    if (beta == 0.0f) {
        for (int i = 0; i < n; i++) y[i * incy] = 0.0f;
    } else if (beta != 1.0f) {
        for (int i = 0; i < n; i++) y[i * incy] *= beta;
    }
    if (alpha == 0.0f) return;

    int upper = (uplo == 'U' || uplo == 'u');

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;

        if (upper) {
            /* Stored:   a[i*lda + i .. n-1]  ×  x[i .. n-1]  — contiguous */
            const float *Ar = a + (size_t)i * lda + i;
            int len = n - i;
            int j = 0;
            if (incx == 1) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                const float *xp = x + i;
                for (; j <= len - 16; j += 16) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j),
                                           _mm256_loadu_ps(xp + j), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j + 8),
                                           _mm256_loadu_ps(xp + j + 8), acc1);
                }
                for (; j <= len - 8; j += 8)
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j),
                                           _mm256_loadu_ps(xp + j), acc0);
                sum = hsum256(_mm256_add_ps(acc0, acc1));
                for (; j < len; j++) sum += Ar[j] * xp[j];
            } else {
                for (j = 0; j < len; j++) sum += Ar[j] * x[(i + j) * incx];
            }
            /* Reflected: a[j*lda + i] for j < i — column, scalar */
            for (int j2 = 0; j2 < i; j2++)
                sum += a[(size_t)j2 * lda + i] * x[j2 * incx];
        } else {
            /* Stored:   a[i*lda + 0 .. i]  ×  x[0 .. i]  — contiguous */
            const float *Ar = a + (size_t)i * lda;
            int len = i + 1;
            int j = 0;
            if (incx == 1) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (; j <= len - 16; j += 16) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j),
                                           _mm256_loadu_ps(x + j), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j + 8),
                                           _mm256_loadu_ps(x + j + 8), acc1);
                }
                for (; j <= len - 8; j += 8)
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j),
                                           _mm256_loadu_ps(x + j), acc0);
                sum = hsum256(_mm256_add_ps(acc0, acc1));
                for (; j < len; j++) sum += Ar[j] * x[j];
            } else {
                for (j = 0; j < len; j++) sum += Ar[j] * x[j * incx];
            }
            /* Reflected: a[j*lda + i] for j > i — column, scalar */
            for (int j2 = i + 1; j2 < n; j2++)
                sum += a[(size_t)j2 * lda + i] * x[j2 * incx];
        }

        y[i * incy] += alpha * sum;
    }
}

/* -----------------------------------------------------------------------
 * STRMV: x = A*x   (A triangular)
 *
 * Optimisation: copy x → xcopy (stack for n≤2048, heap otherwise),
 * then for non-transposed cases each row of A is contiguous → AVX2.
 * Transposed case uses column access (stride lda) → scalar.
 * --------------------------------------------------------------------- */
void strmv(char uplo, char trans, char diag, int n, const float *a, int lda,
           float *x, int incx)
{
    if (n <= 0) return;

    /* Copy x into a contiguous scratch buffer (stride-1) */
    float  stack_buf[2048];
    float *xcopy = (n <= 2048) ? stack_buf : (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) xcopy[i] = x[i * incx];

    int upper = (uplo == 'U' || uplo == 'u');
    int ta    = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    int unit  = (diag == 'U' || diag == 'u');

    if (!ta) {
        if (upper) {
            /* x[i] = sum_{j=i}^{n-1} A[i,j] * xcopy[j]  — row contiguous */
            for (int i = 0; i < n; i++) {
                const float *Ar = a + (size_t)i * lda;
                float sum = unit ? xcopy[i] : Ar[i] * xcopy[i];
                int j = i + 1;
                __m256 acc = _mm256_setzero_ps();
                for (; j <= n - 8; j += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j),
                                          _mm256_loadu_ps(xcopy + j), acc);
                sum += hsum256(acc);
                for (; j < n; j++) sum += Ar[j] * xcopy[j];
                x[i * incx] = sum;
            }
        } else {
            /* x[i] = sum_{j=0}^{i} A[i,j] * xcopy[j]  — row contiguous */
            for (int i = 0; i < n; i++) {
                const float *Ar = a + (size_t)i * lda;
                __m256 acc = _mm256_setzero_ps();
                int j = 0;
                for (; j <= i - 8; j += 8)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(Ar + j),
                                          _mm256_loadu_ps(xcopy + j), acc);
                float sum = hsum256(acc);
                for (; j < i; j++) sum += Ar[j] * xcopy[j];
                sum += unit ? xcopy[i] : Ar[i] * xcopy[i];
                x[i * incx] = sum;
            }
        }
    } else {
        /* Transposed: x = A^T * xcopy
         * Upper A → A^T is lower: x[i] = sum_{j=0}^{i} A[j,i]*xcopy[j]  (column i)
         * Lower A → A^T is upper: x[i] = sum_{j=i}^{n-1} A[j,i]*xcopy[j] (column i) */
        if (upper) {
            for (int i = 0; i < n; i++) {
                float sum = 0.0f;
                for (int j = 0; j < i; j++)
                    sum += a[(size_t)j * lda + i] * xcopy[j];
                sum += unit ? xcopy[i] : a[(size_t)i * lda + i] * xcopy[i];
                x[i * incx] = sum;
            }
        } else {
            for (int i = n - 1; i >= 0; i--) {
                float sum = 0.0f;
                for (int j = i + 1; j < n; j++)
                    sum += a[(size_t)j * lda + i] * xcopy[j];
                sum += unit ? xcopy[i] : a[(size_t)i * lda + i] * xcopy[i];
                x[i * incx] = sum;
            }
        }
    }

    if (n > 2048) free(xcopy);
}

/* -----------------------------------------------------------------------
 * STRSV: A*x = b  (triangular solve, b stored in x on entry)
 *
 * Inherently sequential (data dependency chain) — scalar only.
 * --------------------------------------------------------------------- */
void strsv(char uplo, char trans, char diag, int n, const float *a, int lda,
           float *x, int incx)
{
    if (n <= 0) return;
    int upper = (uplo == 'U' || uplo == 'u');
    int ta    = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    int unit  = (diag == 'U' || diag == 'u');

    if (!ta) {
        if (upper) {
            for (int i = n - 1; i >= 0; i--) {
                float sum = x[i * incx];
                for (int j = i + 1; j < n; j++)
                    sum -= a[(size_t)i * lda + j] * x[j * incx];
                if (!unit) sum /= a[(size_t)i * lda + i];
                x[i * incx] = sum;
            }
        } else {
            for (int i = 0; i < n; i++) {
                float sum = x[i * incx];
                for (int j = 0; j < i; j++)
                    sum -= a[(size_t)i * lda + j] * x[j * incx];
                if (!unit) sum /= a[(size_t)i * lda + i];
                x[i * incx] = sum;
            }
        }
    } else {
        if (upper) {
            for (int i = 0; i < n; i++) {
                float sum = x[i * incx];
                for (int j = 0; j < i; j++)
                    sum -= a[(size_t)j * lda + i] * x[j * incx];
                if (!unit) sum /= a[(size_t)i * lda + i];
                x[i * incx] = sum;
            }
        } else {
            for (int i = n - 1; i >= 0; i--) {
                float sum = x[i * incx];
                for (int j = i + 1; j < n; j++)
                    sum -= a[(size_t)j * lda + i] * x[j * incx];
                if (!unit) sum /= a[(size_t)i * lda + i];
                x[i * incx] = sum;
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * SSYR: A = alpha*x*x^T + A   (symmetric rank-1 update)
 *
 * For each row i: A[i, tri_start..tri_end] += alpha*x[i] * x[tri_start..]
 * The inner row segment is contiguous → AVX2 FMA.
 * --------------------------------------------------------------------- */
void ssyr(char uplo, int n, float alpha, const float *x, int incx,
          float *a, int lda)
{
    if (n <= 0 || alpha == 0.0f) return;
    int upper = (uplo == 'U' || uplo == 'u');

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float axi = alpha * x[i * incx];
        __m256 vaxi = _mm256_set1_ps(axi);
        if (upper) {
            float *Ar = a + (size_t)i * lda + i;
            int len   = n - i;
            int j = 0;
            if (incx == 1) {
                const float *xp = x + i;
                for (; j <= len - 8; j += 8)
                    _mm256_storeu_ps(Ar + j,
                        _mm256_fmadd_ps(vaxi, _mm256_loadu_ps(xp + j),
                                        _mm256_loadu_ps(Ar + j)));
                for (; j < len; j++) Ar[j] += axi * x[i + j];
            } else {
                for (j = 0; j < len; j++) Ar[j] += axi * x[(i + j) * incx];
            }
        } else {
            float *Ar = a + (size_t)i * lda;
            int j = 0;
            if (incx == 1) {
                for (; j <= i - 8; j += 8)
                    _mm256_storeu_ps(Ar + j,
                        _mm256_fmadd_ps(vaxi, _mm256_loadu_ps(x + j),
                                        _mm256_loadu_ps(Ar + j)));
            }
            for (; j <= i; j++) Ar[j] += axi * x[j * incx];
        }
    }
}

/* -----------------------------------------------------------------------
 * SSYR2: A = alpha*x*y^T + alpha*y*x^T + A   (symmetric rank-2 update)
 *
 * Row i: A[i,j] += alpha*x[i]*y[j] + alpha*y[i]*x[j]
 *      = fmadd(alpha*x[i], y[j], fmadd(alpha*y[i], x[j], A[i,j]))
 * Two FMAs per element, both y and x are contiguous → full AVX2.
 * --------------------------------------------------------------------- */
void ssyr2(char uplo, int n, float alpha, const float *x, int incx,
           const float *y, int incy, float *a, int lda)
{
    if (n <= 0 || alpha == 0.0f) return;
    int upper = (uplo == 'U' || uplo == 'u');

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float axi = alpha * x[i * incx];
        float ayi = alpha * y[i * incy];
        __m256 vaxi = _mm256_set1_ps(axi);
        __m256 vayi = _mm256_set1_ps(ayi);

        if (upper) {
            float *Ar     = a + (size_t)i * lda + i;
            int len       = n - i;
            int j = 0;
            if (incx == 1 && incy == 1) {
                const float *xp = x + i;
                const float *yp = y + i;
                for (; j <= len - 8; j += 8) {
                    __m256 va = _mm256_loadu_ps(Ar + j);
                    va = _mm256_fmadd_ps(vaxi, _mm256_loadu_ps(yp + j), va);
                    va = _mm256_fmadd_ps(vayi, _mm256_loadu_ps(xp + j), va);
                    _mm256_storeu_ps(Ar + j, va);
                }
                for (; j < len; j++)
                    Ar[j] += axi * y[i + j] + ayi * x[i + j];
            } else {
                for (j = 0; j < len; j++)
                    Ar[j] += axi * y[(i + j) * incy] + ayi * x[(i + j) * incx];
            }
        } else {
            float *Ar = a + (size_t)i * lda;
            int j = 0;
            if (incx == 1 && incy == 1) {
                for (; j <= i - 8; j += 8) {
                    __m256 va = _mm256_loadu_ps(Ar + j);
                    va = _mm256_fmadd_ps(vaxi, _mm256_loadu_ps(y + j), va);
                    va = _mm256_fmadd_ps(vayi, _mm256_loadu_ps(x + j), va);
                    _mm256_storeu_ps(Ar + j, va);
                }
            }
            for (; j <= i; j++) Ar[j] += axi * y[j * incy] + ayi * x[j * incx];
        }
    }
}

/* -----------------------------------------------------------------------
 * Fortran BLAS entry points (trailing _, args by pointer)
 * --------------------------------------------------------------------- */
void sgemv_(char *trans, int *m, int *n, float *alpha, const float *a, int *lda,
            const float *x, int *incx, float *beta, float *y, int *incy)
{ sgemv(*trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy); }

void sger_(int *m, int *n, float *alpha, const float *x, int *incx,
           const float *y, int *incy, float *a, int *lda)
{ sger(*m, *n, *alpha, x, *incx, y, *incy, a, *lda); }

void ssymv_(char *uplo, int *n, float *alpha, const float *a, int *lda,
            const float *x, int *incx, float *beta, float *y, int *incy)
{ ssymv(*uplo, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy); }

void strmv_(char *uplo, char *trans, char *diag, int *n, const float *a, int *lda,
            float *x, int *incx)
{ strmv(*uplo, *trans, *diag, *n, a, *lda, x, *incx); }

void strsv_(char *uplo, char *trans, char *diag, int *n, const float *a, int *lda,
            float *x, int *incx)
{ strsv(*uplo, *trans, *diag, *n, a, *lda, x, *incx); }

void ssyr_(char *uplo, int *n, float *alpha, const float *x, int *incx,
           float *a, int *lda)
{ ssyr(*uplo, *n, *alpha, x, *incx, a, *lda); }

void ssyr2_(char *uplo, int *n, float *alpha, const float *x, int *incx,
            const float *y, int *incy, float *a, int *lda)
{ ssyr2(*uplo, *n, *alpha, x, *incx, y, *incy, a, *lda); }
