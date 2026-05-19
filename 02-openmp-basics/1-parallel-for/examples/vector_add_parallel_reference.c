#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void vector_add_parallel(float* a, float* b, float* c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    const int n = 1000;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *c = malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    
    vector_add_parallel(a, b, c, n);
    
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            ok = 0;
            break;
        }
    }
    
    printf("[example] result=%s | %s\n", ok ? "correct" : "incorrect", ok ? "PASS" : "FAIL");
    
    free(a);
    free(b);
    free(c);
    return ok ? 0 : 1;
}