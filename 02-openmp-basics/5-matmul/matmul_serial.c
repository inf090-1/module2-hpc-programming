#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matmul_serial(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A[i*n + k];
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

    for (int i = 0; i < n * n; i++) {
        A[i] = 1.0; B[i] = 2.0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_serial(n, A, B, C);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Serial Matmul (%dx%d) Time: %f seconds\n", n, n, time_taken);

    free(A); free(B); free(C);
    return 0;
}
