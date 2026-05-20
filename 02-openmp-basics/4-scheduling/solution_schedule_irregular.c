#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

static int is_prime(uint32_t n) {
    if (n < 2) return 0;
    if ((n & 1U) == 0U) return n == 2U;
    for (uint32_t d = 3; d * d <= n; d += 2) {
        if (n % d == 0U) return 0;
    }
    return 1;
}

static uint32_t serial_count_primes(uint32_t limit) {
    uint32_t count = 0;
    for (uint32_t n = 2; n < limit; n++) {
        count += (uint32_t)is_prime(n);
    }
    return count;
}

uint32_t parallel_count_primes(uint32_t limit) {
    uint32_t count = 0;
    #pragma omp parallel for schedule(guided, 64) reduction(+:count)
    for (uint32_t n = 2; n < limit; n++) {
        count += (uint32_t)is_prime(n);
    }
    return count;
}

int main(int argc, char **argv) {
    int exp = 20;
    if (argc > 1) {
        exp = atoi(argv[1]);
        if (exp < 10) exp = 10;
        if (exp > 26) exp = 26;
    }

    const uint32_t limit = 1U << exp;

    const double t0 = omp_get_wtime();
    const uint32_t serial = serial_count_primes(limit);
    const double t1 = omp_get_wtime();
    const uint32_t parallel = parallel_count_primes(limit);
    const double t2 = omp_get_wtime();

    const int ok = (serial == parallel);
    const double t_serial = t1 - t0;
    const double t_parallel = t2 - t1;
    const double speedup = t_parallel > 0.0 ? t_serial / t_parallel : 0.0;

    printf("[exercise] primes<2^%d = %u | %s\n", exp, parallel, ok ? "PASS" : "FAIL");
    printf("[exercise] serial=%f s parallel=%f s speedup=%.2f\n", t_serial, t_parallel, speedup);
    return ok ? 0 : 1;
}
