#include <stdio.h>
#include <omp.h>

// DOALL: output[i] depends only on input[i].
static int doall_compute_last(const int *input, int n) {
    int output[64];
    if (n > 64) {
        return -1;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * 2 + i;
    }

    return output[n - 1];
}

// DOACROSS: output[i] depends on output[i-1].
static int doacross_compute_last(const int *input, int n) {
    int output[64];
    if (n > 64) {
        return -1;
    }

    #pragma omp parallel
    {
        // The prefix loop has a loop-carried dependency, so only one thread
        // should execute it.
        #pragma omp single
        {
            output[0] = input[0] * 3;
            for (int i = 1; i < n; i++) {
                output[i] = output[i - 1] + input[i] * 3;
            }
        }
    }

    return output[n - 1];
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
