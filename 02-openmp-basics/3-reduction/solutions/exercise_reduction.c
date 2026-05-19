#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Solution: Implement a reduction-based sum of cubes.
// Compute sum_{i=0..n-1} (a[i]^3)
long long parallel_sum_cubes(const int *a, int n) {
    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        const long long v = (long long)a[i];
        sum += v * v * v;
    }

    return sum;
}

int main(void) {
    const int n = 50;
    int a[n];
    for (int i = 0; i < n; i++) {
        a[i] = i + 1;
    }

    // sum_{k=1..n} k^3 = (n(n+1)/2)^2
    const long long t = (long long)n * (long long)(n + 1) / 2LL;
    const long long expected = t * t;
    const double t0 = omp_get_wtime();
    const long long result = parallel_sum_cubes(a, n);
    const double t1 = omp_get_wtime();
    const int ok = (result == expected);

    printf("[exercise] sum_cubes=%lld expected=%lld | %s\n", result, expected, ok ? "PASS" : "FAIL");
    printf("[exercise] time=%f s\n", t1 - t0);
    return ok ? 0 : 1;
}
