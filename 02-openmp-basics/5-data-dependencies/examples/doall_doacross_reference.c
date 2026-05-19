#include <stdio.h>
#include <omp.h>

static void doall_scale(const int *input, int *output, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * 2;
    }
}

static void doacross_prefix_sum(const int *input, int *output, int n) {
    if (n <= 0) {
        return;
    }

    output[0] = input[0];
    for (int i = 1; i < n; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

int main(void) {
    const int input[5] = {1, 2, 3, 4, 5};
    int doall_out[5] = {0};
    int doacross_out[5] = {0};

    doall_scale(input, doall_out, 5);
    doacross_prefix_sum(input, doacross_out, 5);

    const int ok = (doall_out[4] == 10) && (doacross_out[4] == 15);
    printf("[example] doall_last=%d doacross_last=%d | %s\n",
           doall_out[4], doacross_out[4], ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
