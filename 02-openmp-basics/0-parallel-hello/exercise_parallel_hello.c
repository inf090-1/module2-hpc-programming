#include <stdio.h>
#include <omp.h>

int main(void) {
    // TODO: Add a parallel region so each thread prints a hello line.
    // Inside the region, use omp_get_thread_num() and omp_get_num_threads().
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", tid, nthreads);
    return 0;
}
