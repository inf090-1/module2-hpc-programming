#include <stdio.h>
#include <omp.h>

static int parallel_counter(int iterations) {
    int counter = 0;
    omp_lock_t lock;

    omp_init_lock(&lock);

    #pragma omp parallel for
    for (int i = 0; i < iterations; ++i) {
        omp_set_lock(&lock);
        counter++;
        omp_unset_lock(&lock);
    }

    omp_destroy_lock(&lock);
    return counter;
}

int main(void) {
    const int iterations = 100000;
    const int result = parallel_counter(iterations);
    const int ok = (result == iterations);

    printf("[example] lock_counter=%d expected=%d | %s\n",
           result, iterations, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
