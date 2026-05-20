#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matmul_omp_simd(int n, double *A, double *B, double *C) {
    // TODO: Zero out C in parallel (e.g., #pragma omp parallel for collapse(2)).
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
        }
    }

    // Optimized loop ordering (i, k, j) for cache efficiency and vectorization
    // TODO: Parallelize the outer loop with #pragma omp parallel for.
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i*n + k];
            // TODO: Add #pragma omp simd on the inner j loop.
            for (int j = 0; j < n; j++) {
                C[i*n + j] += a_ik * B[k*n + j];
            }
        }
    }
}

int main() {
    int n = 1024;
    double *A = (double*)malloc(n * n * sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    double *C = (double*)malloc(n * n * sizeof(double));

    // Initialize
    for (int i = 0; i < n * n; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    double start = omp_get_wtime();
    matmul_omp_simd(n, A, B, C);
    double end = omp_get_wtime();

    printf("OpenMP + SIMD Matmul (%dx%d) Time: %f seconds\n", n, n, end - start);

    free(A); free(B); free(C);
    return 0;
}
