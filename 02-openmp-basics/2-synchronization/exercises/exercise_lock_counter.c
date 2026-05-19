#include <stdio.h>
#include <omp.h>

// TODO: Use an OpenMP lock to protect a shared counter.
int parallel_counter(int iterations) {
    (void)iterations;
    return 0;
}

int main(void) {
    const int iterations = 100000;
    const int result = parallel_counter(iterations);
    const int ok = (result == iterations);

    printf("[exercise] lock_counter=%d expected=%d | %s\n",
           result, iterations, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
