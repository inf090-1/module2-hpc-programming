#include <stdio.h>
#include <omp.h>

// Solution: Compute sum of values above the threshold using an atomic update.
// Hint: use #pragma omp atomic update for the shared sum.
int sum_above_threshold(const int *values, int n, int threshold) {
    int sum = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (values[i] > threshold) {
            #pragma omp atomic update
            sum += values[i];
        }
    }

    return sum;
}

int main(void) {
    const int values[] = {5, 1, 4, 9, 2, 6, 3, 8, 7};
    const int threshold = 4;
    const int expected = 35;
    const double t0 = omp_get_wtime();
    const int result = sum_above_threshold(values, (int)(sizeof(values) / sizeof(values[0])), threshold);
    const double t1 = omp_get_wtime();

    printf("[exercise] atomic_sum=%d expected=%d | %s\n",
           result, expected, result == expected ? "PASS" : "FAIL");
    printf("[exercise] time=%f s\n", t1 - t0);
    return result == expected ? 0 : 1;
}
