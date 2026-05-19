#include <stdio.h>
#include <omp.h>

// TODO: Find the maximum value in the array.
// Hint: use #pragma omp critical to protect the shared maximum update.
int parallel_max(const int *values, int n) {
    (void)values;
    (void)n;
    return 0;
}

int main(void) {
    const int values[] = {3, 8, 1, 17, 9, 5, 12, 4};
    const int expected = 17;
    const int result = parallel_max(values, (int)(sizeof(values) / sizeof(values[0])));

    printf("[exercise] max=%d expected=%d | %s\n",
           result, expected, result == expected ? "PASS" : "FAIL");
    return result == expected ? 0 : 1;
}
