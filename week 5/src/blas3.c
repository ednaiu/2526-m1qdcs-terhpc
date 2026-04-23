/*
 * blas3.c — Week 6: Other BLAS 3 kernels
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <ctype.h>
#include "../include/sgemm.h"
#include "../include/blas3.h"

/* -----------------------------------------------------------------------
 * SSYRK: C = alpha*A*A' + beta*C
 * --------------------------------------------------------------------- */

void ssyrk(char uplo, char trans, int n, int k, float alpha, const float *a, int lda,
           float beta, float *c, int ldc)
{
    if (n <= 0) return;
    int upper = (uplo == 'U' || uplo == 'u');
    int ta    = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    /* Tiling to reuse sgemm */
    const int block = 128;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < n; i += block) {
        for (int j = 0; j < n; j += block) {
            int ib = (i + block <= n) ? block : (n - i);
            int jb = (j + block <= n) ? block : (n - j);

            if (upper) {
                if (i > j + jb - 1) continue; /* block below triangle */
            } else {
                if (j > i + ib - 1) continue; /* block above triangle */
            }

            /* If block is fully inside triangle, use sgemm_ex.
             * If block intersects diagonal, handle carefully or use scalar fallback.
             */
            int fully_inside = 0;
            if (upper && (i + ib - 1 <= j)) fully_inside = 1;
            if (!upper && (j + jb - 1 <= i)) fully_inside = 1;

            if (fully_inside) {
                sgemm_config_t cfg = SGEMM_DEFAULT_CONFIG;
                if (!ta) {
                    /* C[i,j] = alpha * A[i,:] * A[j,:]^T + beta*C */
                    sgemm_ex(ib, jb, k, alpha, a + i * lda, lda, a + j * lda, lda, beta, c + i * ldc + j, ldc, &cfg);
                } else {
                    /* C[i,j] = alpha * A[:,i]^T * A[:,j] + beta*C */
                    sgemm_ex(ib, jb, k, alpha, a + i, lda, a + j, lda, beta, c + i * ldc + j, ldc, &cfg);
                }
            } else {
                /* Diagonal-intersecting block: simple scalar version */
                for (int ii = i; ii < i + ib; ii++) {
                    for (int jj = j; jj < j + jb; jj++) {
                        if (upper && ii > jj) continue;
                        if (!upper && ii < jj) continue;
                        float s = 0.0f;
                        for (int kk = 0; kk < k; kk++) {
                            float aik = (!ta) ? a[ii * lda + kk] : a[kk * lda + ii];
                            float ajk = (!ta) ? a[jj * lda + kk] : a[kk * lda + jj];
                            s += aik * ajk;
                        }
                        c[ii * ldc + jj] = alpha * s + beta * c[ii * ldc + jj];
                    }
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * STRMM: B = alpha*op(A)*B or B = alpha*B*op(A)
 * --------------------------------------------------------------------- */

void strmm(char side, char uplo, char transa, char diag, int m, int n,
           float alpha, const float *a, int lda, float *b, int ldb)
{
    /* Simplified implementation by temporary buffer to act as C in sgemm */
    if (m <= 0 || n <= 0) return;
    float *bcopy = malloc((size_t)m * n * sizeof(float));
    memcpy(bcopy, b, (size_t)m * n * sizeof(float));

    int left  = (side == 'L' || side == 'l');
    int upper = (uplo == 'U' || uplo == 'u');
    int ta    = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');
    int unit  = (diag == 'U' || diag == 'u');

    /* B = alpha * A * B_copy */
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            if (left) {
                /* Dot product of row i of op(A) and column j of B */
                for (int k = 0; k < m; k++) {
                    float aik = 0.0f;
                    int r = ta ? k : i;
                    int c = ta ? i : k;
                    if (r == c) aik = unit ? 1.0f : a[r * lda + c];
                    else if (upper && r < c) aik = a[r * lda + c];
                    else if (!upper && r > c) aik = a[r * lda + c];
                    
                    if (aik != 0.0f) sum += aik * bcopy[k * ldb + j];
                }
            } else {
                /* B = alpha * B * op(A) */
                for (int k = 0; k < n; k++) {
                    float akj = 0.0f;
                    int r = ta ? j : k;
                    int c = ta ? k : j;
                    if (r == c) akj = unit ? 1.0f : a[r * lda + c];
                    else if (upper && r < c) akj = a[r * lda + c];
                    else if (!upper && r > c) akj = a[r * lda + c];
                    
                    if (akj != 0.0f) sum += bcopy[i * ldb + k] * akj;
                }
            }
            b[i * ldb + j] = alpha * sum;
        }
    }
    free(bcopy);
}

/* -----------------------------------------------------------------------
 * STRSM: op(A)*X = alpha*B or X*op(A) = alpha*B
 * --------------------------------------------------------------------- */

void strsm(char side, char uplo, char transa, char diag, int m, int n,
           float alpha, const float *a, int lda, float *b, int ldb)
{
    if (m <= 0 || n <= 0) return;
    int left  = (side == 'L' || side == 'l');
    int upper = (uplo == 'U' || uplo == 'u');
    int ta    = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');
    int unit  = (diag == 'U' || diag == 'u');

    if (left) {
        /* op(A) * X = alpha * B */
        for (int j = 0; j < n; j++) {
            if (!ta) {
                if (upper) {
                    for (int i = m - 1; i >= 0; i--) {
                        float sum = alpha * b[i * ldb + j];
                        for (int k = i + 1; k < m; k++) sum -= a[i * lda + k] * b[k * ldb + j];
                        if (!unit) sum /= a[i * lda + i];
                        b[i * ldb + j] = sum;
                    }
                } else {
                    for (int i = 0; i < m; i++) {
                        float sum = alpha * b[i * ldb + j];
                        for (int k = 0; k < i; k++) sum -= a[i * lda + k] * b[k * ldb + j];
                        if (!unit) sum /= a[i * lda + i];
                        b[i * ldb + j] = sum;
                    }
                }
            } else {
                /* A^T * X = B */
                if (upper) {
                    for (int i = 0; i < m; i++) {
                        float sum = alpha * b[i * ldb + j];
                        for (int k = 0; k < i; k++) sum -= a[k * lda + i] * b[k * ldb + j];
                        if (!unit) sum /= a[i * lda + i];
                        b[i * ldb + j] = sum;
                    }
                } else {
                    for (int i = m - 1; i >= 0; i--) {
                        float sum = alpha * b[i * ldb + j];
                        for (int k = i + 1; k < m; k++) sum -= a[k * lda + i] * b[k * ldb + j];
                        if (!unit) sum /= a[i * lda + i];
                        b[i * ldb + j] = sum;
                    }
                }
            }
        }
    } else {
        /* X * op(A) = alpha * B */
        for (int i = 0; i < m; i++) {
            if (!ta) {
                if (upper) {
                    for (int j = 0; j < n; j++) {
                        float sum = alpha * b[i * ldb + j];
                        for (int k = 0; k < j; k++) sum -= b[i * ldb + k] * a[k * lda + j];
                        if (!unit) sum /= a[j * lda + j];
                        b[i * ldb + j] = sum;
                    }
                } else {
                    for (int j = n - 1; j >= 0; j--) {
                        float sum = alpha * b[i * ldb + j];
                        for (int k = j + 1; k < n; k++) sum -= b[i * ldb + k] * a[k * lda + j];
                        if (!unit) sum /= a[j * lda + j];
                        b[i * ldb + j] = sum;
                    }
                }
            } else {
                /* X * A^T = B */
                if (upper) {
                    for (int j = n-1; j >= 0; j--) {
                        float sum = alpha * b[i * ldb + j];
                        for (int k = j + 1; k < n; k++) sum -= b[i * ldb + k] * a[j * lda + k];
                        if (!unit) sum /= a[j * lda + j];
                        b[i * ldb + j] = sum;
                    }
                } else {
                    for (int j = 0; j < n; j++) {
                        float sum = alpha * b[i * ldb + j];
                        for (int k = 0; k < j; k++) sum -= b[i * ldb + k] * a[j * lda + k];
                        if (!unit) sum /= a[j * lda + j];
                        b[i * ldb + j] = sum;
                    }
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * Fortran wrappers
 * --------------------------------------------------------------------- */

void ssyrk_(char *uplo, char *trans, int *n, int *k, float *alpha, const float *a, int *lda,
            float *beta, float *c, int *ldc)
{ ssyrk(*uplo, *trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc); }

void strmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
            float *alpha, const float *a, int *lda, float *b, int *ldb)
{ strmm(*side, *uplo, *transa, *diag, *m, *n, *alpha, a, *lda, b, *ldb); }

void strsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
            float *alpha, const float *a, int *lda, float *b, int *ldb)
{ strsm(*side, *uplo, *transa, *diag, *m, *n, *alpha, a, *lda, b, *ldb); }
