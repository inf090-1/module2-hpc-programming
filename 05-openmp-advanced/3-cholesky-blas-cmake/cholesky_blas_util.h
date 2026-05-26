#ifndef CHOLESKY_BLAS_UTIL_H
#define CHOLESKY_BLAS_UTIL_H

#include <stddef.h>

void cholesky_util_xmalloc(void **p, size_t bytes);

// Constructs A = L*L^T where L is random lower-triangular.
// A and L are column-major (Fortran-style), size n*n.
void cholesky_util_fill_spd_from_L(int n, double *A_colmajor, double *L_lower_colmajor, unsigned seed);

// Computes Frobenius residual: ||A_orig - L*L^T||_F.
// A_orig is the original SPD matrix (full), L_colmajor is the in-place Cholesky factor.
double cholesky_util_residual_frob(int n, const double *A_orig, const double *L_colmajor);

void cholesky_util_print_usage(const char *prog);

#endif
