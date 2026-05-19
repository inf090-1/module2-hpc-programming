#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
/* Note: requires openblas to be installed. Compile with -lopenblas */
#include <cblas.h>

int main() {
    int n = 1024;
    double *A = (double*)malloc(n * n * sizeof(double));
    double *B = (double*)malloc(n * n * sizeof(double));
    double *C = (double*)malloc(n * n * sizeof(double));

    // Initialize
    for (int i = 0; i < n * n; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    double start = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 1.0, A, n, B, n, 0.0, C, n);
    double end = omp_get_wtime();

    printf("OpenMP + BLAS Matmul (%dx%d) Time: %f seconds\n", n, n, end - start);

    free(A); free(B); free(C);
    return 0;
}
