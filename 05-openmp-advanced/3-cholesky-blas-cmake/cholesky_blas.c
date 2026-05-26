#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "cholesky_blas_util.h"

// Fortran BLAS symbols (column-major)
extern void dtrsm_(char *side, char *uplo, char *transa, char *diag,
                    int *m, int *n, double *alpha,
                    double *a, int *lda,
                    double *b, int *ldb);

extern void dsyrk_(char *uplo, char *trans, int *n, int *k,
                    double *alpha,
                    double *a, int *lda,
                    double *beta,
                    double *c, int *ldc);

extern void dgemm_(char *transa, char *transb,
                    int *m, int *n, int *k,
                    double *alpha,
                    double *a, int *lda,
                    double *b, int *ldb,
                    double *beta,
                    double *c, int *ldc);

static inline int idx(int i, int j, int n) { return i + j * n; }

static void chol_unblocked_lower(int n, double *A, int k0, int bs)
{
  for (int j = 0; j < bs; ++j) {
    int gj = k0 + j;

    double sum = A[idx(gj, gj, n)];
    for (int p = 0; p < j; ++p) {
      double ljp = A[idx(gj, k0 + p, n)];
      sum -= ljp * ljp;
    }
    if (sum <= 0.0) {
      fprintf(stderr, "Non-SPD encountered (sum=%.6e) at diag %d\n", sum, gj);
      exit(1);
    }

    A[idx(gj, gj, n)] = sqrt(sum);

    for (int i = j + 1; i < bs; ++i) {
      int gi = k0 + i;
      double s = A[idx(gi, gj, n)];
      for (int p = 0; p < j; ++p) {
        s -= A[idx(gi, k0 + p, n)] * A[idx(gj, k0 + p, n)];
      }
      A[idx(gi, gj, n)] = s / A[idx(gj, gj, n)];
    }
  }
}

static void cholesky_blocked_blas_tasks(int n, double *A, int blockSize)
{
  int nb = (n + blockSize - 1) / blockSize;
  int ts[nb];
  for (int b = 0; b < nb; ++b) {
    int start = b * blockSize;
    int rem = n - start;
    ts[b] = rem < blockSize ? rem : blockSize;
  }

  // Staged tasking with taskwait between phases (factor -> trsm -> updates).
  // This keeps the task graph correct for an in-place Cholesky implementation
  // while still parallelizing the main BLAS Level-3 kernels.

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (int k = 0; k < nb; ++k) {
        int k0 = k * blockSize;
        int bk = ts[k];

        // 1) Factor diagonal tile (k,k)
        #pragma omp task firstprivate(k0, bk)
        {
          chol_unblocked_lower(n, A, k0, bk);
        }
        #pragma omp taskwait

        // 2) Compute panel blocks L(i,k) for i>k
        for (int i = k + 1; i < nb; ++i) {
          int i0 = i * blockSize;
          int bi = ts[i];

          #pragma omp task firstprivate(i0, bi, k0, bk)
          {
            char side = 'R';
            char uplo = 'L';
            char transa = 'T';
            char diag = 'N';
            double alpha = 1.0;
            int m = bi;      // rows of B
            int nrhs = bk;   // cols of B
            int lda = n;
            int ldb = n;

            double *A11 = &A[idx(k0, k0, n)];
            double *Aik = &A[idx(i0, k0, n)];

            dtrsm_(&side, &uplo, &transa, &diag,
                   &m, &nrhs, &alpha,
                   A11, &lda,
                   Aik, &ldb);
          }
        }
        #pragma omp taskwait

        // 3) Update trailing submatrix (i,j) for i>j>k and diagonal (i,i)
        for (int i = k + 1; i < nb; ++i) {
          int i0 = i * blockSize;
          int bi = ts[i];

          // Diagonal update: A(i,i) -= A(i,k) * A(i,k)^T
          #pragma omp task firstprivate(i0, bi, k0, bk)
          {
            char uplo = 'L';
            char trans = 'N';
            double alpha = -1.0;
            double beta = 1.0;
            int N = bi;
            int K = bk;
            int lda = n;
            int ldc = n;

            double *Aik = &A[idx(i0, k0, n)];
            double *Aii = &A[idx(i0, i0, n)];

            dsyrk_(&uplo, &trans, &N, &K,
                   &alpha,
                   Aik, &lda,
                   &beta,
                   Aii, &ldc);
          }

          // Off-diagonal updates: A(i,j) -= A(i,k) * A(j,k)^T
          for (int j = k + 1; j < i; ++j) {
            int j0 = j * blockSize;
            int bj = ts[j];

            #pragma omp task firstprivate(i0, bi, j0, bj, k0, bk)
            {
              char transa = 'N';
              char transb = 'T';
              double alpha = -1.0;
              double beta = 1.0;

              int m = bi;
              int ncol = bj;
              int kdim = bk;

              int lda = n;
              int ldb = n;
              int ldc = n;

              double *Aik = &A[idx(i0, k0, n)];
              double *Ajk = &A[idx(j0, k0, n)];
              double *Aij = &A[idx(i0, j0, n)];

              dgemm_(&transa, &transb,
                     &m, &ncol, &kdim,
                     &alpha,
                     Aik, &lda,
                     Ajk, &ldb,
                     &beta,
                     Aij, &ldc);
            }
          }
        }
        #pragma omp taskwait
      }
    }
  }
}

int main(int argc, char **argv)
{
  int n = 512;
  int block = 64;
  unsigned seed = 0;

  if (argc >= 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
    cholesky_util_print_usage(argv[0]);
    return 0;
  }

  if (argc >= 2) n = atoi(argv[1]);
  if (argc >= 3) block = atoi(argv[2]);
  if (argc >= 4) seed = (unsigned)strtoul(argv[3], NULL, 10);

  if (n <= 0 || block <= 0) {
    cholesky_util_print_usage(argv[0]);
    return 1;
  }

  size_t bytes = (size_t)n * (size_t)n * sizeof(double);

  double *A = NULL;
  double *A_orig = NULL;
  double *Ltmp = NULL;

  cholesky_util_xmalloc((void **)&A, bytes);
  cholesky_util_xmalloc((void **)&A_orig, bytes);
  cholesky_util_xmalloc((void **)&Ltmp, bytes);

  cholesky_util_fill_spd_from_L(n, A, Ltmp, seed);
  memcpy(A_orig, A, bytes);

  cholesky_blocked_blas_tasks(n, A, block);

  double res = cholesky_util_residual_frob(n, A_orig, A);

  double diag_sum = 0.0;
  for (int i = 0; i < n; ++i) diag_sum += A[idx(i, i, n)];

  printf("Cholesky (OpenMP tasks + BLAS) OK\n");
  printf("  n=%d block=%d seed=%u\n", n, block, seed);
  printf("  diag_sum=%.12e\n", diag_sum);
  printf("  residual_Frob=%.6e\n", res);

  free(A);
  free(A_orig);
  free(Ltmp);

  return 0;
}
