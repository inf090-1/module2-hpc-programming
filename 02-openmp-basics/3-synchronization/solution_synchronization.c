#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void serial_histogram(const int *data, int n, int *hist, int bins) {
    for (int i = 0; i < bins; i++) {
        hist[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        const int v = data[i];
        hist[v] += 1;
    }
}

void parallel_histogram(const int *data, int n, int *hist, int bins) {
    for (int i = 0; i < bins; i++) {
        hist[i] = 0;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        const int v = data[i];
        // Alternative: #pragma omp atomic or omp_lock_t can also protect updates.
        #pragma omp critical
        {
            hist[v] += 1;
        }
    }
}

static int compare_hist(const int *a, const int *b, int bins) {
    for (int i = 0; i < bins; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

int main(void) {
    const int n = 12000000;
    const int bins = 64;
    int *data = (int *)malloc((size_t)n * sizeof(int));
    int *hist_serial = (int *)malloc((size_t)bins * sizeof(int));
    int *hist_parallel = (int *)malloc((size_t)bins * sizeof(int));
    if (!data || !hist_serial || !hist_parallel) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        data[i] = (i * 17 + 13) & (bins - 1);
    }

    const double t0 = omp_get_wtime();
    serial_histogram(data, n, hist_serial, bins);
    const double t1 = omp_get_wtime();
    parallel_histogram(data, n, hist_parallel, bins);
    const double t2 = omp_get_wtime();

    const int ok = compare_hist(hist_serial, hist_parallel, bins);
    const double t_serial = t1 - t0;
    const double t_parallel = t2 - t1;
    const double speedup = t_parallel > 0.0 ? t_serial / t_parallel : 0.0;

    printf("[exercise] histogram | %s\n", ok ? "PASS" : "FAIL");
    printf("[exercise] serial=%f s parallel=%f s speedup=%.2f\n", t_serial, t_parallel, speedup);

    free(data);
    free(hist_serial);
    free(hist_parallel);
    return ok ? 0 : 1;
}
