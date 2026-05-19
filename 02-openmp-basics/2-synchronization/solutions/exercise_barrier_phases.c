#include <stdio.h>
#include <omp.h>

// Solution: Keep the two phases in order.
// Hint: phase 1 uses #pragma omp for nowait.
// Hint: add an explicit #pragma omp barrier before phase 2.
int staged_square_sum(const int *values, int n) {
    int partial[128] = {0};
    int total = 0;

    #pragma omp parallel shared(partial, total)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

        #pragma omp for nowait
        for (int i = 0; i < n; ++i) {
            partial[tid] += values[i] * values[i];
        }

        // Ensure all threads finished updating partial before aggregation.
        #pragma omp barrier

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
    const int values[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
    const int expected = 207;
    const double t0 = omp_get_wtime();
    const int result = staged_square_sum(values, (int)(sizeof(values) / sizeof(values[0])));
    const double t1 = omp_get_wtime();

    printf("[exercise] barrier_square_sum=%d expected=%d | %s\n",
           result, expected, result == expected ? "PASS" : "FAIL");
    printf("[exercise] time=%f s\n", t1 - t0);
    return result == expected ? 0 : 1;
}
