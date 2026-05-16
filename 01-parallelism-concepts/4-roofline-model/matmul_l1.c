#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024

void matmul(float A[N][N], float B[N][N], float C[N][N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }
}

int main() {
    // Allocate on heap to avoid stack overflow with N=1024
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));
    
    // Initialize
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0f;
            B[i][j] = 2.0f;
            C[i][j] = 0.0f;
        }
    }
    
    // Warm-up
    matmul(A, B, C);
    
    int num_runs = 10;
    double total_time = 0.0;
    
    for (int r = 0; r < num_runs; r++) {
        double start = omp_get_wtime();
        matmul(A, B, C);
        double end = omp_get_wtime();
        total_time += (end - start);
    }
    
    printf("Algorithm=MATMUL, Time=%f\n", total_time / num_runs);
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}
