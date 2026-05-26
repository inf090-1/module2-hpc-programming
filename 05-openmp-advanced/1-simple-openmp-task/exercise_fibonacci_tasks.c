#include <stdio.h>
#include <omp.h>

// Cutoff: only create tasks for n > CUTOFF, otherwise compute serially.
#ifndef FIB_CUTOFF
#define FIB_CUTOFF 20
#endif

static int fib_serial(int n);

// TODO: Implement Fibonacci using OpenMP tasks.
static int fib_task(int n) {
    if (n < 2) return n;

    // Use serial recursion below the cutoff to limit task overhead.
    if (n <= FIB_CUTOFF) {
        return fib_serial(n);
    }

    int x = 0;
    int y = 0;

    // TODO: Create tasks for fib_task(n-1) and fib_task(n-2) within a taskgroup.

    return x + y;
}

static int fib_serial(int n) {
    if (n < 2) return n;
    return fib_serial(n - 1) + fib_serial(n - 2);
}

// CPU baseline using OpenMP parallel for.
// We parallelize only the top-level split:
//   fib(n) = fib(n-1) + fib(n-2)
static int fib_omp_parallel_for_top(int n) {
    if (n < 2) return n;

    int x = 0;
    int y = 0;

    #pragma omp parallel for schedule(static) num_threads(2) shared(x, y)
    for (int t = 0; t < 2; ++t) {
        if (t == 0) x = fib_serial(n - 1);
        else y = fib_serial(n - 2);
    }

    return x + y;
}

int main(void) {
    const int n = 30;
    const int expected = 832040;

    // Baseline 1: CPU serial.
    double t_serial0 = omp_get_wtime();
    int serial_res = fib_serial(n);
    double t_serial1 = omp_get_wtime();
    double t_serial = t_serial1 - t_serial0;

    // Baseline 2: CPU OpenMP parallel-for.
    double t_omp0 = omp_get_wtime();
    int omp_pf_res = fib_omp_parallel_for_top(n);
    double t_omp1 = omp_get_wtime();
    double t_omp = t_omp1 - t_omp0;

    // Exercise: Fibonacci using OpenMP tasks.
    int result = 0;
    double t_task0 = omp_get_wtime();
    // TODO: Create the parallel region and call the parallel fibonacci function.
    double t_task1 = omp_get_wtime();
    double t_task = t_task1 - t_task0;

    const int ok = (result == expected);

    printf("[fib] n=%d expected=%d\n", n, expected);
    printf("[fib] cpu_serial=%d time=%.6f s\n", serial_res, t_serial);
    printf("[fib] cpu_omp_parallel_for=%d time=%.6f s\n", omp_pf_res, t_omp);
    printf("[fib] exercise_tasks=%d time=%.6f s | %s\n",
           result, t_task, ok ? "PASS" : "FAIL");
    if (t_task > 0.0) {
        printf("[fib] speedup_serial_vs_tasks=%.2fx\n", t_serial / t_task);
        printf("[fib] speedup_omp_vs_tasks=%.2fx\n", t_omp / t_task);
    }

    return ok ? 0 : 1;
}
