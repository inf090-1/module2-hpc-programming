#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Number of compute iterations per element to make kernel compute-bound.
#ifndef COMPUTE_ITERS
#define COMPUTE_ITERS 50
#endif

static float compute_busy(float x, float y) {
    float z;
    for (int k = 0; k < COMPUTE_ITERS; k++) {
        x = x * y + 1.0f;
        y = y * x + 1.0f;
        z = x + y;
    }
    return z;
}

// TODO: Implement GPU offload using OpenMP target directives.
// Hint: Use compute_busy(a[i], b[i]) as the computation per element.
// Hint: Try adding 'num_teams(512) thread_limit(256)' clauses.
void vector_add_gpu(float* a, float* b, float* c, int n) {
    // TODO: Add the OpenMP target directive with map(a[0:n], b[0:n], c[0:n]).

    // TODO: Compute c[i] = compute_busy(a[i], b[i]) inside the loop.

    // TODO: After adding the target directive, compute on the device.
    // For now (incorrect stub), set output to 0 so the test fails until
    // you implement the offload.
    for (int i = 0; i < n; i++) {
        c[i] = 0.0f;
    }
}

static void vector_add_cpu_serial(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = compute_busy(a[i], b[i]);
    }
    volatile float sink = c[n - 1];
    (void)sink;
}

static void vector_add_cpu_omp_parallel_for(const float* a, const float* b, float* c, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        c[i] = compute_busy(a[i], b[i]);
    }
}

int main(void) {
    const int n = 1<<24;
    float *a = malloc(n * sizeof(float));
    float *b = malloc(n * sizeof(float));
    float *c = malloc(n * sizeof(float));
    float *c_serial = malloc(n * sizeof(float));
    float *c_omp = malloc(n * sizeof(float));

    if (!a || !b || !c || !c_serial || !c_omp) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    // Baselines (validate to prevent compiler from optimizing away).
    double t_serial0 = omp_get_wtime();
    vector_add_cpu_serial(a, b, c_serial, n);
    double t_serial1 = omp_get_wtime();
    double t_serial = t_serial1 - t_serial0;
    int serial_ok = 1;
    for (int i = 0; i < n; i++) {
        if (c_serial[i] != compute_busy(a[i], b[i])) { serial_ok = 0; break; }
    }

    double t_omp0 = omp_get_wtime();
    vector_add_cpu_omp_parallel_for(a, b, c_omp, n);
    double t_omp1 = omp_get_wtime();
    double t_omp = t_omp1 - t_omp0;
    int omp_ok = 1;
    for (int i = 0; i < n; i++) {
        if (c_omp[i] != compute_busy(a[i], b[i])) { omp_ok = 0; break; }
    }

    // Exercise: OpenMP target offload.
    double t_gpu0 = omp_get_wtime();
    // TODO: call the OpenMP target vector add function.
    double t_gpu1 = omp_get_wtime();
    double t_gpu = t_gpu1 - t_gpu0;

    // Validate target result.
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (c[i] != compute_busy(a[i], b[i])) ok = 0;
    }

    printf("[vector-add] n=%d iters=%d\n", n, COMPUTE_ITERS);
    printf("[vector-add] cpu_serial_time=%.6f s\n", t_serial);
    printf("[vector-add] cpu_omp_parallel_for_time=%.6f s\n", t_omp);
    printf("[vector-add] exercise_target_offload_time=%.6f s\n", t_gpu);
    printf("[vector-add] exercise_result=%s | %s\n", ok ? "correct" : "incorrect", ok ? "PASS" : "FAIL");
    printf("[vector-add] serial_ok=%d omp_ok=%d\n", serial_ok, omp_ok);
    if (t_gpu > 0.0) {
        printf("[vector-add] speedup_serial_vs_target=%.2fx\n", t_serial / t_gpu);
        printf("[vector-add] speedup_omp_vs_target=%.2fx\n", t_omp / t_gpu);
    }

    free(a); free(b); free(c);
    free(c_serial); free(c_omp);
    return ok ? 0 : 1;
}
