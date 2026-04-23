/*
 * blas1.h — Week 4: Optimised AVX2 BLAS 1 kernels
 *
 * Kernels: sscal, scopy, sswap, saxpy, sdot, snrm2, sasum, isamax, srot
 *
 * Assumptions: incX = incY = 1 (as per assignment)
 *
 * Each kernel exists in three flavours:
 *   <name>_scalar   — pure scalar reference
 *   <name>_avx2     — single-threaded AVX2 + prefetch
 *   <name>          — parallel AVX2 (OpenMP), matches Fortran C ABI name
 *
 * Fortran BLAS entry points (trailing underscore, all args by pointer)
 * are provided in blas1.c.
 */

#ifndef BLAS1_H
#define BLAS1_H

#include <stddef.h>

/* -----------------------------------------------------------------------
 * C API — arguments by value, return by value
 * --------------------------------------------------------------------- */

/* Scale: x[i] *= alpha */
void sscal_scalar (int n, float alpha, float *x, int incx);
void sscal_avx2   (int n, float alpha, float *x, int incx);
void sscal         (int n, float alpha, float *x, int incx);

/* Copy: y[i] = x[i] */
void scopy_scalar (int n, const float *x, int incx, float *y, int incy);
void scopy_avx2   (int n, const float *x, int incx, float *y, int incy);
void scopy         (int n, const float *x, int incx, float *y, int incy);

/* Swap: x[i] <-> y[i] */
void sswap_scalar (int n, float *x, int incx, float *y, int incy);
void sswap_avx2   (int n, float *x, int incx, float *y, int incy);
void sswap         (int n, float *x, int incx, float *y, int incy);

/* AXPY: y[i] += alpha * x[i] */
void saxpy_scalar (int n, float alpha, const float *x, int incx, float *y, int incy);
void saxpy_avx2   (int n, float alpha, const float *x, int incx, float *y, int incy);
void saxpy         (int n, float alpha, const float *x, int incx, float *y, int incy);

/* Dot product: return sum(x[i]*y[i]) */
float sdot_scalar (int n, const float *x, int incx, const float *y, int incy);
float sdot_avx2   (int n, const float *x, int incx, const float *y, int incy);
float sdot         (int n, const float *x, int incx, const float *y, int incy);

/* Euclidean norm: return sqrt(sum(x[i]^2)) */
float snrm2_scalar (int n, const float *x, int incx);
float snrm2_avx2   (int n, const float *x, int incx);
float snrm2         (int n, const float *x, int incx);

/* Sum of absolute values: return sum(|x[i]|) */
float sasum_scalar (int n, const float *x, int incx);
float sasum_avx2   (int n, const float *x, int incx);
float sasum         (int n, const float *x, int incx);

/* Index of max absolute value (0-based): return argmax(|x[i]|) */
int isamax_scalar (int n, const float *x, int incx);
int isamax_avx2   (int n, const float *x, int incx);
int isamax         (int n, const float *x, int incx);

/* Givens rotation: (x[i], y[i]) = (c*x[i]+s*y[i], -s*x[i]+c*y[i]) */
void srot_scalar (int n, float *x, int incx, float *y, int incy, float c, float s);
void srot_avx2   (int n, float *x, int incx, float *y, int incy, float c, float s);
void srot         (int n, float *x, int incx, float *y, int incy, float c, float s);

/* -----------------------------------------------------------------------
 * Fortran BLAS entry points (Fortran calls pass all args by pointer,
 * integer types are int* for gfortran default -fdefault-integer-4)
 * --------------------------------------------------------------------- */

void sscal_ (int *n, float *alpha, float *x, int *incx);
void scopy_ (int *n, const float *x, int *incx, float *y, int *incy);
void sswap_ (int *n, float *x, int *incx, float *y, int *incy);
void saxpy_ (int *n, float *alpha, const float *x, int *incx,
             float *y, int *incy);
float sdot_  (int *n, const float *x, int *incx, const float *y, int *incy);
float snrm2_ (int *n, const float *x, int *incx);
float sasum_ (int *n, const float *x, int *incx);
int   isamax_(int *n, const float *x, int *incx);  /* 1-based as per Fortran */
void srot_   (int *n, float *x, int *incx, float *y, int *incy,
              float *c, float *s);

#endif /* BLAS1_H */
