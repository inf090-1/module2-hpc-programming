#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static void fill_random(float *data, int n) {
    unsigned int seed = 12345u;
    for (int i = 0; i < n; i++) {
        seed = 1103515245u * seed + 12345u;
        data[i] = (float)(seed & 0xFFFFu) * 0.001f;
    }
}

static double serial_max_abs(const float *data, int n) {
    double maxv = 0.0;
    for (int i = 0; i < n; i++) {
        const double v = fabs((double)data[i]);
        if (v > maxv) maxv = v;
    }
    return maxv;
}

static double serial_l2(const float *data, int n, double maxv) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const double v = (double)data[i] / maxv;
        sum += v * v;
    }
    return sum;
}

double parallel_l2(const float *data, int n, double maxv) {
    // TODO: Use #pragma omp parallel for with reduction(+:sum).
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const double v = (double)data[i] / maxv;
        sum += v * v;
    }
    return sum;
}

int main(void) {
    const int n = 64000000;
    float *data = (float *)malloc((size_t)n * sizeof(float));
    if (!data) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    fill_random(data, n);
    const double maxv = serial_max_abs(data, n);

    const double t0 = omp_get_wtime();
    const double serial = serial_l2(data, n, maxv);
    const double t1 = omp_get_wtime();
    const double parallel = parallel_l2(data, n, maxv);
    const double t2 = omp_get_wtime();

    const double diff = fabs(serial - parallel);
    const double rel = diff / (fabs(serial) + 1e-12);
    const int ok = (diff < 1e-5) || (rel < 1e-10);
    const double t_serial = t1 - t0;
    const double t_parallel = t2 - t1;
    const double speedup = t_parallel > 0.0 ? t_serial / t_parallel : 0.0;

    printf("[exercise] l2 diff=%.3e rel=%.3e | %s\n", diff, rel, ok ? "PASS" : "FAIL");
    printf("[exercise] serial=%f s parallel=%f s speedup=%.2f\n", t_serial, t_parallel, speedup);

    free(data);
    return ok ? 0 : 1;
}
