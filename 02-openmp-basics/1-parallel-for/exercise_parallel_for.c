#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static void serial_work(const uint64_t *input, uint64_t *output, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t v = input[i];
        for (int it = 0; it < 256; it++) {
            v = mix64(v + (uint64_t)it);
        }
        output[i] = v;
    }
}

void parallel_work(const uint64_t *input, uint64_t *output, int n) {
    // TODO: Add #pragma omp parallel for to speed up the loop.
    for (int i = 0; i < n; i++) {
        uint64_t v = input[i];
        for (int it = 0; it < 256; it++) {
            v = mix64(v + (uint64_t)it);
        }
        output[i] = v;
    }
}

static uint64_t checksum(const uint64_t *data, int n) {
    uint64_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

int main(void) {
    const int n = 8000000;
    uint64_t *input = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *out_serial = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    uint64_t *out_parallel = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    if (!input || !out_serial || !out_parallel) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        input[i] = (uint64_t)i * 11400714819323198485ULL + 17ULL;
    }

    const double t0 = omp_get_wtime();
    serial_work(input, out_serial, n);
    const double t1 = omp_get_wtime();
    parallel_work(input, out_parallel, n);
    const double t2 = omp_get_wtime();

    const uint64_t s0 = checksum(out_serial, n);
    const uint64_t s1 = checksum(out_parallel, n);
    const int ok = (s0 == s1);
    const double t_serial = t1 - t0;
    const double t_parallel = t2 - t1;
    const double speedup = t_parallel > 0.0 ? t_serial / t_parallel : 0.0;

    printf("[exercise] checksum=%llu | %s\n", (unsigned long long)s1, ok ? "PASS" : "FAIL");
    printf("[exercise] serial=%f s parallel=%f s speedup=%.2f\n", t_serial, t_parallel, speedup);

    free(input);
    free(out_serial);
    free(out_parallel);
    return ok ? 0 : 1;
}
