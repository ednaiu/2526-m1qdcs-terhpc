/*
 * blas_fortran.c — Fortran BLAS entry point for SGEMM.
 *
 * All other Fortran symbols (sscal_, sdot_, sgemv_, ssyrk_, …) are already
 * implemented at the bottom of blas1.c, blas2.c, and blas3.c respectively.
 * This file adds only sgemm_(), which requires a non-trivial storage bridge.
 *
 * Storage bridge: Fortran uses column-major matrices; our sgemm() uses
 * row-major.  The key identity is:
 *
 *   A col-major M×K matrix with ld=L, when the same bytes are read as a
 *   row-major K×M matrix with ld=L, yields the mathematical transpose A^T.
 *
 * Consequence for sgemm_():
 *
 *   Fortran: C_col(M×N) = α · op(A) · op(B) + β·C_col
 *   ⟺  C^T(N×M) = α · [op(B)]^T · [op(A)]^T + β·C^T
 *
 * Case N,N — no copies needed:
 *   B_ptr (col-maj K×N, ld=LDB) read as row-maj N×K = B^T  → first arg
 *   A_ptr (col-maj M×K, ld=LDA) read as row-maj K×M = A^T  → second arg
 *   C^T = α·B^T·A^T + β·C^T = [α·A·B + β·C]^T  ✓
 *   call: sgemm(N, M, K, α, B_ptr, LDB, A_ptr, LDA, β, C_ptr, LDC)
 *
 * Case T,N (TRANSA='T'):
 *   A stored col-maj K×M (op(A)=A^T is M×K).
 *   Need A as K×M row-major → physical copy via colmaj_to_rowmaj().
 *   B unchanged.
 *   C^T = α·B^T·A + β·C^T  ✓
 *
 * Case N,T (TRANSB='T'):
 *   B stored col-maj N×K (op(B)=B^T is K×N).
 *   Need B as N×K row-major → physical copy.
 *   A unchanged.
 *   C^T = α·B·A^T + β·C^T  ✓
 *
 * Case T,T:
 *   Both copies needed.
 *   C^T = α·B·A + β·C^T  ✓
 *
 * Fortran ABI note (gfortran, Linux x86-64):
 *   Character arguments are passed as char* with a hidden by-value length
 *   appended after all visible arguments.  We declare only the visible
 *   arguments; the hidden lengths land in registers / stack slots that C
 *   never reads — harmless on System V AMD64 ABI.
 */

#include <stdlib.h>
#include "../include/sgemm.h"

/*
 * Convert col-major (rows×cols, ld=ld_src) to row-major (rows×cols, ld=cols).
 * dst must have space for rows*cols floats.
 */
static void colmaj_to_rowmaj(const float *src, float *dst,
                              int rows, int cols, int ld_src)
{
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[r * cols + c] = src[c * ld_src + r];
}

void sgemm_(char *transa, char *transb,
            int *m,  int *n,  int *k,
            float *alpha,
            const float *a, int *lda,
            const float *b, int *ldb,
            float *beta,
            float *c, int *ldc)
{
    const int M = *m, N = *n, K = *k;
    const int LDA = *lda, LDB = *ldb, LDC = *ldc;

    const int ta = (*transa == 'T' || *transa == 't');
    const int tb = (*transb == 'T' || *transb == 't');

    const float *a_rm  = a;       /* pointer used in the row-major call */
    const float *b_rm  = b;
    int          lda_rm = LDA;
    int          ldb_rm = LDB;
    float       *a_tmp  = NULL;
    float       *b_tmp  = NULL;

    if (ta) {
        /*
         * A stored K×M col-major (op(A) = A^T is M×K).
         * Produce K×M row-major in a_tmp so that our sgemm
         * reads it as the K×M matrix A directly.
         */
        a_tmp  = malloc((size_t)K * M * sizeof(float));
        colmaj_to_rowmaj(a, a_tmp, K, M, LDA);
        a_rm   = a_tmp;
        lda_rm = M;
    }

    if (tb) {
        /*
         * B stored N×K col-major (op(B) = B^T is K×N).
         * Produce N×K row-major in b_tmp.
         */
        b_tmp  = malloc((size_t)N * K * sizeof(float));
        colmaj_to_rowmaj(b, b_tmp, N, K, LDB);
        b_rm   = b_tmp;
        ldb_rm = K;
    }

    /*
     * Row-major equivalent:
     *   C^T (N×M, ld=LDC) = α · b_rm (N×K) · a_rm (K×M) + β · C^T
     */
    sgemm(N, M, K, *alpha, b_rm, ldb_rm, a_rm, lda_rm, *beta, c, LDC);

    free(a_tmp);
    free(b_tmp);
}
