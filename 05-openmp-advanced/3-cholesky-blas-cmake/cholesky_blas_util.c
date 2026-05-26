#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "cholesky_blas_util.h"

static void die(const char *msg)
{
  fprintf(stderr, "%s\n", msg);
  exit(1);
}

void cholesky_util_xmalloc(void **p, size_t bytes)
{
  if (bytes == 0) die("allocation requested with 0 bytes");

  void *ptr = NULL;
#if defined(_ISOC11_SOURCE)
  ptr = aligned_alloc(64, bytes);
#else
  // Fallback (posix_memalign is available on most HPC Linux setups)
  if (posix_memalign(&ptr, 64, bytes) != 0) ptr = NULL;
#endif
  if (!ptr) die("allocation failed");
  *p = ptr;
}

void cholesky_util_fill_spd_from_L(int n, double *A_colmajor, double *L_lower_colmajor, unsigned seed)
{
  srand(seed);

  // Build a random lower-triangular L with strictly positive diagonal
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (i < j) {
        L_lower_colmajor[i + j * n] = 0.0;
      } else {
        double r = (double)rand() / (double)RAND_MAX;
        // Add a bit more spread on off-diagonals
        L_lower_colmajor[i + j * n] = 0.5 * (2.0 * r - 1.0);
      }
    }
    // Make diagonal safely positive
    double diag = 1.0 + 0.1 * ((double)rand() / (double)RAND_MAX);
    L_lower_colmajor[j + j * n] = diag;
  }

  // A = L * L^T (symmetric positive definite)
  // Only fill lower triangle; upper is filled for convenience/validation.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = 0.0;
      int kmax = (j < i ? j : i);
      for (int k = 0; k <= kmax; ++k) {
        sum += L_lower_colmajor[i + k * n] * L_lower_colmajor[j + k * n];
      }
      A_colmajor[i + j * n] = sum;
      A_colmajor[j + i * n] = sum; // symmetry
    }
  }
}

double cholesky_util_residual_frob(int n, const double *A_orig, const double *L_colmajor)
{
  // residual = ||A_orig - L*L^T||_F over full matrix (computed via symmetry)
  double res2 = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = 0.0;
      int kmax = (j < i ? j : i);
      for (int k = 0; k <= kmax; ++k) {
        sum += L_colmajor[i + k * n] * L_colmajor[j + k * n];
      }
      double diff = A_orig[i + j * n] - sum;
      res2 += (i == j ? diff * diff : 2.0 * diff * diff);
    }
  }
  return sqrt(res2);
}

void cholesky_util_print_usage(const char *prog)
{
  fprintf(stderr,
          "Usage: %s [n] [block] [seed]\n"
          "  n     : matrix size (default 512)\n"
          "  block : Cholesky block size (default 64)\n"
          "  seed  : RNG seed (default 0)\n",
          prog);
}
