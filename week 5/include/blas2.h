/*
 * blas2.h — Week 5: Optimised AVX2 BLAS 2 kernels
 *
 * Kernels: sgemv, ssymv, strmv, strsv, sger, ssyr, ssyr2
 */

#ifndef BLAS2_H
#define BLAS2_H

#include <stddef.h>

/* -----------------------------------------------------------------------
 * C API — Row-major assumed for A
 * --------------------------------------------------------------------- */

/* Matrix-vector multiplication: y = alpha*A*x + beta*y */
void sgemv(char trans, int m, int n, float alpha, const float *a, int lda,
           const float *x, int incx, float beta, float *y, int incy);

/* Rank-1 update: A = alpha*x*y' + A */
void sger(int m, int n, float alpha, const float *x, int incx,
          const float *y, int incy, float *a, int lda);

/* Symmetric Matrix-vector: y = alpha*A*x + beta*y */
void ssymv(char uplo, int n, float alpha, const float *a, int lda,
           const float *x, int incx, float beta, float *y, int incy);

/* Triangular Matrix-vector: x = A*x */
void strmv(char uplo, char trans, char diag, int n, const float *a, int lda,
           float *x, int incx);

/* Triangular Solve: A*x = b (result in x) */
void strsv(char uplo, char trans, char diag, int n, const float *a, int lda,
           float *x, int incx);

/* Symmetric Rank-1 update: A = alpha*x*x' + A */
void ssyr(char uplo, int n, float alpha, const float *x, int incx,
          float *a, int lda);

/* Symmetric Rank-2 update: A = alpha*x*y' + alpha*y*x' + A */
void ssyr2(char uplo, int n, float alpha, const float *x, int incx,
           const float *y, int incy, float *a, int lda);

/* -----------------------------------------------------------------------
 * Fortran BLAS entry points
 * --------------------------------------------------------------------- */

void sgemv_ (char *trans, int *m, int *n, float *alpha, const float *a, int *lda,
             const float *x, int *incx, float *beta, float *y, int *incy);
void sger_  (int *m, int *n, float *alpha, const float *x, int *incx,
             const float *y, int *incy, float *a, int *lda);
void ssymv_ (char *uplo, int *n, float *alpha, const float *a, int *lda,
             const float *x, int *incx, float *beta, float *y, int *incy);
void strmv_ (char *uplo, char *trans, char *diag, int *n, const float *a, int *lda,
             float *x, int *incx);
void strsv_ (char *uplo, char *trans, char *diag, int *n, const float *a, int *lda,
             float *x, int *incx);
void ssyr_  (char *uplo, int *n, float *alpha, const float *x, int *incx,
             float *a, int *lda);
void ssyr2_ (char *uplo, int *n, float *alpha, const float *x, int *incx,
             const float *y, int *incy, float *a, int *lda);

#endif /* BLAS2_H */
