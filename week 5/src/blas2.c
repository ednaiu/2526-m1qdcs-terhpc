/*
 * blas2.c — Week 5: Optimised BLAS 2 kernels
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <ctype.h>
#include "../include/blas2.h"

/* Helper for horizontal sum */
static inline float hsum256(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(sum);
}

/* -----------------------------------------------------------------------
 * SGEMV: y = alpha*A*x + beta*y
 * --------------------------------------------------------------------- */

void sgemv(char trans, int m, int n, float alpha, const float *a, int lda,
           const float *x, int incx, float beta, float *y, int incy)
{
    if (m <= 0 || n <= 0) return;

    /* Handle beta first */
    if (beta == 0.0f) {
        if (incy == 1) {
            for (int i = 0; i < (trans == 'N' || trans == 'n' ? m : n); i++) y[i] = 0.0f;
        } else {
            for (int i = 0; i < (trans == 'N' || trans == 'n' ? m : n); i++) y[i * incy] = 0.0f;
        }
    } else if (beta != 1.0f) {
        if (incy == 1) {
            for (int i = 0; i < (trans == 'N' || trans == 'n' ? m : n); i++) y[i] *= beta;
        } else {
            for (int i = 0; i < (trans == 'N' || trans == 'n' ? m : n); i++) y[i * incy] *= beta;
        }
    }

    if (alpha == 0.0f) return;

    int ta = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    if (!ta) {
        /* y = alpha*A*x + y (Row-major A) 
         * Each row of A is a dot product with x.
         */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            const float *Arow = a + i * lda;
            float sum = 0.0f;
            if (incx == 1) {
                __m256 vacc = _mm256_setzero_ps();
                int j = 0;
                for (; j <= n - 8; j += 8) {
                    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(Arow + j), _mm256_loadu_ps(x + j), vacc);
                }
                sum = hsum256(vacc);
                for (; j < n; j++) sum += Arow[j] * x[j];
            } else {
                for (int j = 0; j < n; j++) sum += Arow[j] * x[j * incx];
            }
            y[i * incy] += alpha * sum;
        }
    } else {
        /* y = alpha*A^T*x + y (Row-major A)
         * y_j += alpha * sum_{i} A_{ij} * x_i
         * This is like: y += alpha * x_i * A_{i,:}
         */
        if (incy == 1) {
            #pragma omp parallel
            {
                /* Each thread can process a set of rows of A and update Y. 
                 * But multiple threads would update the same Y. 
                 * Alternative: shared Y with atomic or reduction, or partition Y.
                 * Partitioning Y is better for large N.
                 */
                int nthd = omp_get_num_threads();
                int tid  = omp_get_thread_num();
                int n_per_thd = (n + nthd - 1) / nthd;
                int n_start = tid * n_per_thd;
                int n_end = n_start + n_per_thd;
                if (n_end > n) n_end = n;

                for (int j = n_start; j < n_end; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < m; i++) {
                        sum += a[i * lda + j] * x[i * incx];
                    }
                    y[j] += alpha * sum;
                }
            }
        } else {
            /* Slow path for incy != 1 */
            for (int i = 0; i < m; i++) {
                float xi = alpha * x[i * incx];
                const float *Arow = a + i * lda;
                for (int j = 0; j < n; j++) {
                    y[j * incy] += xi * Arow[j];
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * SGER: A = alpha*x*y^T + A
 * --------------------------------------------------------------------- */

void sger(int m, int n, float alpha, const float *x, int incx,
          const float *y, int incy, float *a, int lda)
{
    if (m <= 0 || n <= 0 || alpha == 0.0f) return;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        float axi = alpha * x[i * incx];
        float *Arow = a + i * lda;
        if (incy == 1) {
            __m256 vaxi = _mm256_set1_ps(axi);
            int j = 0;
            for (; j <= n - 8; j += 8) {
                __m256 vy = _mm256_loadu_ps(y + j);
                __m256 va = _mm256_loadu_ps(Arow + j);
                _mm256_storeu_ps(Arow + j, _mm256_fmadd_ps(vaxi, vy, va));
            }
            for (; j < n; j++) Arow[j] += axi * y[j];
        } else {
            for (int j = 0; j < n; j++) Arow[j] += axi * y[j * incy];
        }
    }
}

/* -----------------------------------------------------------------------
 * SSYMV: y = alpha*A*x + beta*y (A symmetric)
 * --------------------------------------------------------------------- */

void ssymv(char uplo, int n, float alpha, const float *a, int lda,
           const float *x, int incx, float beta, float *y, int incy)
{
    /* Simplified: treat ssymv as sgemv for now by expanding the symmetric half, 
     * but a better impl would respect UPLO. 
     * Let's do it right.
     */
    if (n <= 0) return;
    
    /* Handle beta */
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
        for (int j = 0; j < n; j++) {
            float a_ij;
            if (upper) {
                a_ij = (i <= j) ? a[i * lda + j] : a[j * lda + i];
            } else {
                a_ij = (i >= j) ? a[i * lda + j] : a[j * lda + i];
            }
            sum += a_ij * x[j * incx];
        }
        y[i * incy] += alpha * sum;
    }
}

/* -----------------------------------------------------------------------
 * STRMV: x = A*x (A triangular)
 * --------------------------------------------------------------------- */

void strmv(char uplo, char trans, char diag, int n, const float *a, int lda,
           float *x, int incx)
{
    if (n <= 0) return;
    float *xcopy = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) xcopy[i] = x[i * incx];

    int upper = (uplo == 'U' || uplo == 'u');
    int ta    = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    int unit  = (diag == 'U' || diag == 'u');

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            float a_ij = 0.0f;
            int r = ta ? j : i;
            int c = ta ? i : j;

            if (r == c) {
                a_ij = unit ? 1.0f : a[r * lda + c];
            } else if (upper) {
                if (r < c) a_ij = a[r * lda + c];
            } else {
                if (r > c) a_ij = a[r * lda + c];
            }
            sum += a_ij * xcopy[j];
        }
        x[i * incx] = sum;
    }
    free(xcopy);
}

/* -----------------------------------------------------------------------
 * STRSV: A*x = b (solve for x, original x contains b)
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
            /* Backward substitution */
            for (int i = n - 1; i >= 0; i--) {
                float sum = x[i * incx];
                for (int j = i + 1; j < n; j++) {
                    sum -= a[i * lda + j] * x[j * incx];
                }
                if (!unit) sum /= a[i * lda + i];
                x[i * incx] = sum;
            }
        } else {
            /* Forward substitution */
            for (int i = 0; i < n; i++) {
                float sum = x[i * incx];
                for (int j = 0; j < i; j++) {
                    sum -= a[i * lda + j] * x[j * incx];
                }
                if (!unit) sum /= a[i * lda + i];
                x[i * incx] = sum;
            }
        }
    } else {
        /* A^T * x = b */
        if (upper) {
            /* Forward substitution for A^T (U^T is Lower) */
            for (int i = 0; i < n; i++) {
                float sum = x[i * incx];
                for (int j = 0; j < i; j++) {
                    sum -= a[j * lda + i] * x[j * incx];
                }
                if (!unit) sum /= a[i * lda + i];
                x[i * incx] = sum;
            }
        } else {
            /* Backward substitution for A^T (L^T is Upper) */
            for (int i = n - 1; i >= 0; i--) {
                float sum = x[i * incx];
                for (int j = i + 1; j < n; j++) {
                    sum -= a[j * lda + i] * x[j * incx];
                }
                if (!unit) sum /= a[i * lda + i];
                x[i * incx] = sum;
            }
        }
    }
}

void ssyr(char uplo, int n, float alpha, const float *x, int incx,
          float *a, int lda)
{
    if (n <= 0 || alpha == 0.0f) return;
    int upper = (uplo == 'U' || uplo == 'u');

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float axi = alpha * x[i * incx];
        if (upper) {
            for (int j = i; j < n; j++) a[i * lda + j] += axi * x[j * incx];
        } else {
            for (int j = 0; j <= i; j++) a[i * lda + j] += axi * x[j * incx];
        }
    }
}

void ssyr2(char uplo, int n, float alpha, const float *x, int incx,
           const float *y, int incy, float *a, int lda)
{
    if (n <= 0 || alpha == 0.0f) return;
    int upper = (uplo == 'U' || uplo == 'u');

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float axi = alpha * x[i * incx];
        float ayi = alpha * y[i * incy];
        if (upper) {
            for (int j = i; j < n; j++) a[i * lda + j] += axi * y[j * incy] + ayi * x[j * incx];
        } else {
            for (int j = 0; j <= i; j++) a[i * lda + j] += axi * y[j * incy] + ayi * x[j * incx];
        }
    }
}

/* -----------------------------------------------------------------------
 * Fortran wrappers
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
