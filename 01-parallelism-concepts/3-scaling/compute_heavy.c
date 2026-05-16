#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void compute_heavy(double *a, double *c, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        double val = a[i];
        // Artificial high arithmetic intensity loop
        for (int step = 0; step < 1000; step++) {
            val = val * 1.000001 + 0.000001;
        }
        c[i] = val;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <size_multiplier>\n", argv[0]);
        return 1;
    }

    size_t multiplier = atoll(argv[1]);
    size_t base_size = 500000; // 500k elements ~ plenty of compute
    size_t n = base_size * multiplier;

    double *a = (double *)malloc(n * sizeof(double));
    double *c = (double *)malloc(n * sizeof(double));

    if (!a || !c) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        a[i] = i * 0.001;
        c[i] = 0.0;
    }

    // Warm-up thread pool
    compute_heavy(a, c, 1000);

    int num_runs = 5;
    double total_time = 0.0;

    for (int r = 0; r < num_runs; r++) {
        double start = omp_get_wtime();
        compute_heavy(a, c, n);
        double end = omp_get_wtime();
        total_time += (end - start);
    }

    printf("N=%zu, Time=%f seconds\n", n, total_time / num_runs);

    free(a);
    free(c);

    return 0;
}
