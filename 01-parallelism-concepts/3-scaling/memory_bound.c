#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void memory_heavy(double *a, double *b, double *c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <size_multiplier>\n", argv[0]);
        return 1;
    }

    size_t multiplier = atoll(argv[1]);
    size_t base_size = 50000000; // 50M elements
    size_t n = base_size * multiplier;

    double *a = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *c = (double *)malloc(n * sizeof(double));

    if (!a || !b || !c) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    // Warm-up thread pool and memory pages
    memory_heavy(a, b, c, n);

    int num_runs = 5;
    double total_time = 0.0;

    for (int r = 0; r < num_runs; r++) {
        double start = omp_get_wtime();
        memory_heavy(a, b, c, n);
        double end = omp_get_wtime();
        total_time += (end - start);
    }

    printf("N=%zu, Time=%f seconds\n", n, total_time / num_runs);

    free(a);
    free(b);
    free(c);

    return 0;
}
