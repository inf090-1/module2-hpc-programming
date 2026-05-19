#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Solution: Compute sum of absolute differences using OpenMP reduction.
long long parallel_abs_diff_sum(const int *a, const int *b, int n) {
    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        int diff = a[i] - b[i];
        if (diff < 0) diff = -diff;
        sum += diff;
    }

    return sum;
}

int main(void) {
    const int n = 200;
    int a[n];
    int b[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i / 2;
    }

    long long expected = 0;
    for (int i = 0; i < n; i++) {
        int diff = a[i] - b[i];
        if (diff < 0) diff = -diff;
        expected += diff;
    }

    const double t0 = omp_get_wtime();
    const long long result = parallel_abs_diff_sum(a, b, n);
    const double t1 = omp_get_wtime();
    const int ok = (result == expected);

    printf("[exercise] abs_diff_sum=%lld expected=%lld | %s\n", result, expected, ok ? "PASS" : "FAIL");
    printf("[exercise] time=%f s\n", t1 - t0);
    return ok ? 0 : 1;
}
