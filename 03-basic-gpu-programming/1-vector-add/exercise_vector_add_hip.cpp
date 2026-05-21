#include <hip/hip_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void cpu_vector_add_omp(const float *a, const float *b, float *c, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_hip(const float *a, const float *b, float *c, int n) {
    // TODO: Compute global thread index.
    // TODO: Bounds check and compute c[idx] = a[idx] + b[idx].
}

int main(void) {
    const int n = 1 << 24;
    const size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_cpu = (float *)malloc(bytes);
    float *h_gpu = (float *)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        h_a[i] = sinf((float)i) * 0.5f;
        h_b[i] = cosf((float)i) * 2.0f;
    }

    double cpu_t0 = omp_get_wtime();
    cpu_vector_add_omp(h_a, h_b, h_cpu, n);
    double cpu_t1 = omp_get_wtime();
    double cpu_seconds = cpu_t1 - cpu_t0;

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    // TODO: Allocate memory on the device for d_a, d_b, and d_c.

    double gpu_mem_t0 = omp_get_wtime();
    // TODO: Copy data from host arrays (h_a, h_b) to device arrays (d_a, d_b).
    double gpu_mem_t1 = omp_get_wtime();

    // TODO: Determine the optimal block and grid sizes.
    // Ensure that you launch enough threads to cover 'n' elements.
    
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    // TODO: Call the vector_add_hip kernel.
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, start, stop);

    double gpu_mem_back_t0 = omp_get_wtime();
    // TODO: Copy result from device array (d_c) to host array (h_gpu).
    double gpu_mem_back_t1 = omp_get_wtime();

    const double gpu_h2d_time = (gpu_mem_t1 - gpu_mem_t0);
    const double gpu_d2h_time = (gpu_mem_back_t1 - gpu_mem_back_t0);
    double gpu_mem_time = gpu_h2d_time + gpu_d2h_time;

    float max_abs_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(h_cpu[i] - h_gpu[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    const int ok = (max_abs_diff < 1e-5f);
    const double gpu_seconds = (double)gpu_ms / 1000.0;

    printf("[vec-add] n=%d\n", n);
    printf("[vec-add] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[vec-add] gpu_hip_time=%.6f s (%.3f ms)\n", gpu_seconds, gpu_ms);
    printf("[vec-add] gpu_mem_h2d_time=%.6f s\n", gpu_h2d_time);
    printf("[vec-add] gpu_mem_d2h_time=%.6f s\n", gpu_d2h_time);
    printf("[vec-add] gpu_mem_total_time=%.6f s\n", gpu_mem_time);
    printf("[vec-add] gpu_hip_total_time=%.6f s\n", gpu_seconds + gpu_mem_time);
    if (gpu_seconds > 0.0) {
        printf("[vec-add] speedup_kernel_only=%.2fx\n", cpu_seconds / gpu_seconds);
        printf("[vec-add] speedup_with_mem=%.2fx\n", cpu_seconds / (gpu_seconds + gpu_mem_time));
    }
    printf("[vec-add] max_abs_diff=%g | %s\n", max_abs_diff, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);
    // TODO: Free device memory.
    free(h_a);
    free(h_b);
    free(h_cpu);
    free(h_gpu);

    return ok ? 0 : 1;
}
