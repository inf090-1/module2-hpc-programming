#include <stdio.h>
#include <omp.h>

static void print_owner(const char *label, const int *owner, int n) {
    printf("[example] %s owner:", label);
    for (int i = 0; i < n; ++i) {
        printf(" %d", owner[i]);
    }
    printf("\n");
}

int main(void) {
    const int n = 12;
    int static_owner[12] = {0};
    int dynamic_owner[12] = {0};

    #pragma omp parallel for schedule(static, 3)
    for (int i = 0; i < n; ++i) {
        static_owner[i] = omp_get_thread_num();
        for (volatile int spin = 0; spin < (i % 4 + 1) * 10000; ++spin) {
        }
    }

    #pragma omp parallel for schedule(dynamic, 3)
    for (int i = 0; i < n; ++i) {
        dynamic_owner[i] = omp_get_thread_num();
        for (volatile int spin = 0; spin < (i % 4 + 1) * 10000; ++spin) {
        }
    }

    print_owner("static", static_owner, n);
    print_owner("dynamic", dynamic_owner, n);
    printf("[example] scheduling demo | PASS\n");
    return 0;
}
