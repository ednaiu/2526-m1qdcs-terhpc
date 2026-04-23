/*
 * blas3.h — Week 6: Other BLAS 3 kernels
 *
 * Kernels: syrk, trmm, trsm
 */

#ifndef BLAS3_H
#define BLAS3_H

#include <stddef.h>

/* -----------------------------------------------------------------------
 * C API
 * --------------------------------------------------------------------- */

/* Symmetric rank-k update: C = alpha*A*A' + beta*C */
void ssyrk(char uplo, char trans, int n, int k, float alpha, const float *a, int lda,
           float beta, float *c, int ldc);

/* Triangular matrix-matrix multiply: B = alpha*op(A)*B  or  B = alpha*B*op(A) */
void strmm(char side, char uplo, char transa, char diag, int m, int n,
           float alpha, const float *a, int lda, float *b, int ldb);

/* Triangular solve with multiple right-hand sides: op(A)*X = alpha*B  or  X*op(A) = alpha*B */
void strsm(char side, char uplo, char transa, char diag, int m, int n,
           float alpha, const float *a, int lda, float *b, int ldb);

/* -----------------------------------------------------------------------
 * Fortran BLAS entry points
 * --------------------------------------------------------------------- */

void ssyrk_(char *uplo, char *trans, int *n, int *k, float *alpha, const float *a, int *lda,
            float *beta, float *c, int *ldc);
void strmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
            float *alpha, const float *a, int *lda, float *b, int *ldb);
void strsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
            float *alpha, const float *a, int *lda, float *b, int *ldb);

#endif /* BLAS3_H */
