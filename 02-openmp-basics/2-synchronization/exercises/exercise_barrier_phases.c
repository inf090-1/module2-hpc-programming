#include <stdio.h>
#include <omp.h>

// TODO: Keep the two phases in order.
// Hint: phase 1 uses #pragma omp for nowait.
// Hint: add an explicit #pragma omp barrier before phase 2.
int staged_sum(const int *values, int n) {
    int partial[128] = {0};
    int total = 0;

    #pragma omp parallel shared(partial, total)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        #pragma omp for nowait
        for (int i = 0; i < n; ++i) {
            partial[tid] += values[i];
        }

        // TODO: synchronize all threads here with #pragma omp barrier.

        #pragma omp single
        {
            const int limit = nthreads < 128 ? nthreads : 128;
            for (int i = 0; i < limit; ++i) {
                total += partial[i];
            }
        }
    }

    return total;
}

int main(void) {
    const int values[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const int expected = 36;
    const int result = staged_sum(values, (int)(sizeof(values) / sizeof(values[0])));

    printf("[exercise] barrier_sum=%d expected=%d | %s\n",
           result, expected, result == expected ? "PASS" : "FAIL");
    return result == expected ? 0 : 1;
}
