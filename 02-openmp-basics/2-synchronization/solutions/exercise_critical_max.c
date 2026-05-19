#include <stdio.h>
#include <omp.h>

// Solution: Find the maximum value in the array.
// Hint: use #pragma omp critical to protect the shared maximum update.
int parallel_max(const int *values, int n) {
    int max_val = values[0];

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const int v = values[i];

        // Protect update of the shared max.
        #pragma omp critical
        {
            if (v > max_val) {
                max_val = v;
            }
        }
    }

    return max_val;
}

int main(void) {
    const int values[] = {6, 2, 9, 1, 8, 3, 7, 4};
    const int expected = 9;
    const double t0 = omp_get_wtime();
    const int result = parallel_max(values, (int)(sizeof(values) / sizeof(values[0])));
    const double t1 = omp_get_wtime();

    printf("[exercise] max=%d expected=%d | %s\n",
           result, expected, result == expected ? "PASS" : "FAIL");
    printf("[exercise] time=%f s\n", t1 - t0);
    return result == expected ? 0 : 1;
}
