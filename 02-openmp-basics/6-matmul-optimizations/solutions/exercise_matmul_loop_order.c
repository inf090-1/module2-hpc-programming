#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static void matmul_serial_ijk(int n, const double *A, const double *B, double *C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }

    // i-j-k loop order
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void matmul_omp_ijk(int n, const double *A, const double *B, double *C) {
    // Zero C
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }

    // Parallelize i,j (DOALL across (i,j) cells)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void matmul_omp_ikj_simd(int n, const double *A, const double *B, double *C) {
    // Zero C in parallel.
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }

    // Optimized i-k-j loop order for cache locality on B.
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            const double a_ik = A[i * n + k];
            #pragma omp simd
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

static double max_abs_diff(int n, const double *X, const double *Y) {
    double m = 0.0;
    for (int i = 0; i < n * n; i++) {
        const double d = fabs(X[i] - Y[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    const int n = 96;
    const double eps = 1e-9;

    double *A = (double *)malloc((size_t)n * n * sizeof(double));
    double *B = (double *)malloc((size_t)n * n * sizeof(double));
    double *Cref = (double *)malloc((size_t)n * n * sizeof(double));
    double *Cijk = (double *)malloc((size_t)n * n * sizeof(double));
    double *Cikj = (double *)malloc((size_t)n * n * sizeof(double));
    if (!A || !B || !Cref || !Cijk || !Cikj) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n * n; i++) {
        A[i] = (double)((i % 17) + 1) * 0.01;
        B[i] = (double)((i % 13) + 1) * 0.02;
    }

    matmul_serial_ijk(n, A, B, Cref);

    const double t0 = omp_get_wtime();
    matmul_omp_ijk(n, A, B, Cijk);
    const double t1 = omp_get_wtime();
    matmul_omp_ikj_simd(n, A, B, Cikj);
    const double t2 = omp_get_wtime();

    const double diff_ijk = max_abs_diff(n, Cijk, Cref);
    const double diff_ikj = max_abs_diff(n, Cikj, Cref);

    const int ok = (diff_ijk < eps) && (diff_ikj < eps);

    printf("[exercise] diff_ijk=%.3e diff_ikj=%.3e | %s\n",
           diff_ijk, diff_ikj, ok ? "PASS" : "FAIL");
    printf("[exercise] time_ijk=%f s time_ikj=%f s\n",
           t1 - t0, t2 - t1);

    free(A);
    free(B);
    free(Cref);
    free(Cijk);
    free(Cikj);
    return ok ? 0 : 1;
}
