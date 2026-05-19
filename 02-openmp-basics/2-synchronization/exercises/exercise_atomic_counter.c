#include <stdio.h>
#include <omp.h>

// TODO: Count how many values are above the threshold.
// Hint: use #pragma omp atomic update for the shared counter.
int count_above_threshold(const int *values, int n, int threshold) {
    (void)values;
    (void)n;
    (void)threshold;
    return 0;
}

int main(void) {
    const int values[] = {1, 4, 7, 9, 2, 11, 13, 3};
    const int threshold = 6;
    const int expected = 4;
    const int result = count_above_threshold(values, (int)(sizeof(values) / sizeof(values[0])), threshold);

    printf("[exercise] atomic_count=%d expected=%d | %s\n",
           result, expected, result == expected ? "PASS" : "FAIL");
    return result == expected ? 0 : 1;
}
