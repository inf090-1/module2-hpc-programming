#include <stdio.h>
#include <omp.h>

// Solution: Add a loop schedule clause that fits irregular work.
// Hint: schedule(guided, 4) is a good starting point.
void simulate_irregular_work(int *owner, int n) {
    #pragma omp parallel for schedule(guided, 4)
    for (int i = 0; i < n; ++i) {
        owner[i] = omp_get_thread_num();
        for (volatile int spin = 0; spin < (i % 6 + 1) * 7000; ++spin) {
        }
    }
}

static void print_owner(const int *owner, int n) {
    printf("[exercise] owner:");
    for (int i = 0; i < n; ++i) {
        printf(" %d", owner[i]);
    }
    printf("\n");
}

int main(void) {
    const int n = 16;
    int owner[16];
    for (int i = 0; i < n; i++) owner[i] = -1;

    const double t0 = omp_get_wtime();
    simulate_irregular_work(owner, n);
    const double t1 = omp_get_wtime();
    print_owner(owner, n);

    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (owner[i] < 0) ok = 0;
    }

    printf("[exercise] guided schedule demo | %s\n", ok ? "PASS" : "FAIL");
    printf("[exercise] time=%f s\n", t1 - t0);
    return 0;
}
