#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000000 // 100M elements, ~800MB total memory

void saxpy(float *X, float *Y, float a) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        Y[i] = a * X[i] + Y[i];
    }
}

int main() {
    float *X = (float*)malloc(N * sizeof(float));
    float *Y = (float*)malloc(N * sizeof(float));
    float a = 2.0f;
    
    // Initialize
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        X[i] = 1.0f;
        Y[i] = 2.0f;
    }
    
    // Warm-up
    saxpy(X, Y, a);
    
    int num_runs = 5;
    double total_time = 0.0;
    
    for (int r = 0; r < num_runs; r++) {
        double start = omp_get_wtime();
        saxpy(X, Y, a);
        double end = omp_get_wtime();
        total_time += (end - start);
    }
    
    printf("Algorithm=SAXPY, Time=%f\n", total_time / num_runs);
    
    free(X);
    free(Y);
    return 0;
}
