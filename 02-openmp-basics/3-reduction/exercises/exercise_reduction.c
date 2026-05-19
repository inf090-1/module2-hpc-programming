#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// TODO: Implement parallel dot product using OpenMP reduction
// Dot product = sum(a[i] * b[i]) for all i
double parallel_dot_product(double* a, double* b, int n) {
    // TODO: Use #pragma omp parallel for with reduction(+:sum)
    // TODO: Accumulate a[i] * b[i] for each element
    // Hint: Initialize sum = 0.0, then add a[i] * b[i] in loop
    (void)a;
    (void)b;
    (void)n;
    return 0.0;
}

int main(void) {
    const int n = 1000;
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 1.0;
    }
    
    const double expected = n * 1.0 * 1.0;
    const double result = parallel_dot_product(a, b, n);
    
    int ok = (result - expected < 1e-9);
    
    printf("[exercise] dot=%.2f expected=%.2f | %s\n", result, expected, ok ? "PASS" : "FAIL");
    
    free(a);
    free(b);
    return ok ? 0 : 1;
}