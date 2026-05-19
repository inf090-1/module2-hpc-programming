#include <stdio.h>
#include <omp.h>

static int count_above_threshold(const int *values, int n, int threshold) {
    int count = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        if (values[i] > threshold) {
            #pragma omp atomic update
            count++;
        }
    }

    return count;
}

int main(void) {
    const int values[] = {1, 4, 7, 9, 2, 11, 13, 3};
    const int threshold = 6;
    const int expected = 4;
    const int result = count_above_threshold(values, (int)(sizeof(values) / sizeof(values[0])), threshold);

    printf("[example] atomic_count=%d expected=%d | %s\n",
           result, expected, result == expected ? "PASS" : "FAIL");
    return result == expected ? 0 : 1;
}
