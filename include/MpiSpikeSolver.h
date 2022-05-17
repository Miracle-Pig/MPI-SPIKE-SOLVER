#ifndef __INCLUDE_MPISPIKESOLVER_H__
#define __INCLUDE_MPISPIKESOLVER_H__

#include "enum.h"
#include "macro.h"
#include "util.h"
#include "factorization.h"
#include "solve.h"
#include "facandsolve.h"

#include <math.h>
#include <algorithm>
#include <iomanip>

// #include </opt/intel/oneapi/mkl/2021.3.0/include/mkl.h>
/* Interface of  LApack*/
extern "C"
{
    double dlangb_(char *norm, int *n, int *kl, int *ku, double *ab, int *ldab, double *work);
    void dlacpy_(char *uplo, int *m, int *n, double *a, int *lda, double *b, int *ldb);
    void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
    void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
    void dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *a, int *lda, double *b, int *ldb);
    void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
    void dgbtrf_(int *m, int *n, int *kl, int *ku, double *ab, int *ldab, int *ipiv, int *info);
    void dswap_(int *n, double *dx, int *incx, double *dy, int *incy);
    void dgbtrs_(char *trans, int *n, int *kl, int *ku, int *nrhs, double *ab, int *ldab, int *ipiv, double *b, int *ldb, int *info);
}

/* Interface of spike-1.0 */
extern "C"
{
    void dgbalu_(int *n, int *kl, int *ku, double *a, int *lda, double *nzero, double *norm, int *info);
    void dgbaul_(int *n, int *kl, int *ku, double *a, int *lda, double *nzero, double *norm, int *info);
    void dtbsm_(char *uplo, char *trans, char *diag, int *n, int *rhs, int *kd, double *A, int *lda, double *B, int *ldb);
    void dgbtrsl_(char *trans, int *n, int *kl, int *ku, int *nrhs, double *ab, int *ldab, int *ipiv, double *B, int *ldb, int *info);
    void dgbtrful_(int *m, int *n, int *kl, int *ku, double *ab, int *ldab, int *ipiv, int *info);
    void dgbtrsu_(char *trans, int *n, int *kl, int *ku, int *nrhs, double *ab, int *ldab, int *ipiv, double *b, int *ldb, int *info);
}

#endif
