#include <stdio.h>
#include <omp.h>

// Solution: Use an OpenMP lock to protect a shared sum.
// Compute sum_{i=0..iterations-1} i^2.
long long parallel_locked_square_sum(int iterations) {
    long long sum = 0;
    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
        omp_set_lock(&lock);
        sum += (long long)i * (long long)i;
        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);
    return sum;
}

int main(void) {
    const int iterations = 1000;
    const long long expected = (long long)(iterations - 1) * (long long)iterations * (2LL * iterations - 1) / 6LL;
    const double t0 = omp_get_wtime();
    const long long result = parallel_locked_square_sum(iterations);
    const double t1 = omp_get_wtime();
    const int ok = (result == expected);

    printf("[exercise] lock_square_sum=%lld expected=%lld | %s\n",
           result, expected, ok ? "PASS" : "FAIL");
    printf("[exercise] time=%f s\n", t1 - t0);
    return ok ? 0 : 1;
}
