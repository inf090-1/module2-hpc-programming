#include <stdio.h>
#include <omp.h>

int main(void) {
    const int iterations = 100000;
    int counter = 0;

    #pragma omp parallel for
    for (int i = 0; i < iterations; ++i) {
        #pragma omp critical
        {
            counter++;
        }
    }

    const int ok = (counter == iterations);
    printf("[example] counter=%d expected=%d | %s\n",
           counter, iterations, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
