#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// TODO: Implement parallel sum using OpenMP
// Use #pragma omp parallel for with reduction(+:sum)
int parallel_sum(int* arr, int n) {
    // TODO: Add #pragma omp parallel for before the loop
    // TODO: Add reduction(+:sum) clause to avoid race conditions
    // Hint: Initialize sum = 0, then add arr[i] in parallel loop
    (void)arr;
    (void)n;
    return 0;
}

int main(void) {
    const int n = 100;
    int arr[100];
    for (int i = 0; i < n; i++) {
        arr[i] = i + 1;
    }
    
    const int expected = n * (n + 1) / 2;
    const int result = parallel_sum(arr, n);
    
    int ok = (result == expected);
    
    printf("[exercise] sum=%d expected=%d | %s\n", result, expected, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}