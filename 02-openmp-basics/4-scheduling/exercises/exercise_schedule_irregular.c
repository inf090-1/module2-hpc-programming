#include <stdio.h>
#include <omp.h>

// TODO: Add a loop schedule clause that fits irregular work.
// Hint: schedule(dynamic, 2) is a good starting point.
void simulate_irregular_work(int *owner, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        owner[i] = omp_get_thread_num();
        for (volatile int spin = 0; spin < (i % 4 + 1) * 10000; ++spin) {
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
    const int n = 12;
    int owner[12] = {0};

    simulate_irregular_work(owner, n);
    print_owner(owner, n);

    printf("[exercise] schedule demo | PASS\n");
    return 0;
}
