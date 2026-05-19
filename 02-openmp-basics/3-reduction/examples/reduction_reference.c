#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static double parallel_dot_product(const double *a, const double *b, int n) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

int main(void) {
    const int n = 1000;
    double *a = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));

    if (a == NULL || b == NULL) {
        free(a);
        free(b);
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        a[i] = 1.0;
        b[i] = 1.0;
    }

    const double expected = (double)n;
    const double result = parallel_dot_product(a, b, n);
    const int ok = (result == expected);

    printf("[example] dot=%.2f expected=%.2f | %s\n",
           result, expected, ok ? "PASS" : "FAIL");

    free(a);
    free(b);
    return ok ? 0 : 1;
}
