#include <stdio.h>
#include <omp.h>

// TODO: Implement a DOALL phase (independent iterations) and a DOACROSS phase
// (loop-carried dependency) with correct synchronization.

// DOALL: output[i] depends only on input[i].
static int doall_compute_last(const int *input, int n) {
    (void)input;
    (void)n;
    // TODO: Implement using #pragma omp parallel for
    // Example transform: output[i] = input[i] * 2 + i
    return 0;
}

// DOACROSS: output[i] depends on output[i-1].
static int doacross_compute_last(const int *input, int n) {
    (void)input;
    (void)n;
    // TODO: Implement using a dependent prefix sum.
    // Hint: run the dependent prefix loop inside a single thread
    // (e.g., #pragma omp parallel + #pragma omp single).
    // Example transform: output[i] = output[i-1] + input[i] * 3
    return 0;
}

int main(void) {
    const int input[] = {2, 7, 1, 8, 2, 8};
    const int n = (int)(sizeof(input) / sizeof(input[0]));

    const double t0 = omp_get_wtime();
    const int doall_last = doall_compute_last(input, n);
    const int doacross_last = doacross_compute_last(input, n);
    const double t1 = omp_get_wtime();

    // Reference answers (sequential)
    const int doall_out_last = input[n - 1] * 2 + (n - 1);
    int doacross_out_last = input[0] * 3;
    for (int i = 1; i < n; i++) {
        doacross_out_last = doacross_out_last + input[i] * 3;
    }

    const int ok = (doall_last == doall_out_last) && (doacross_last == doacross_out_last);

    printf("[exercise] doall_last=%d doacross_last=%d expected=%d | %s\n",
           doall_last, doacross_last, doacross_out_last, ok ? "PASS" : "FAIL");
    printf("[exercise] time=%f s\n", t1 - t0);
    return ok ? 0 : 1;
}
