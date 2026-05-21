#include <hip/hip_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE 16

static void matmul_cpu_omp(const float *a, const float *b, float *c, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

__global__ void matmul_naive_hip(const float *a, const float *b, float *c, int n) {
    // TODO: Write naive matrix multiplication kernel without shared memory.
    // Use threadIdx and blockIdx to compute row and col.
}

int main(void) {
    const int n = 512;
    const size_t bytes = (size_t)n * n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_cpu = (float *)malloc(bytes);
    float *h_gpu = (float *)malloc(bytes);

    for (int i = 0; i < n * n; ++i) {
        h_a[i] = (float)((i % 17) - 8) * 0.125f;
        h_b[i] = (float)((i % 13) - 6) * 0.25f;
    }

    double cpu_t0 = omp_get_wtime();
    matmul_cpu_omp(h_a, h_b, h_cpu, n);
    double cpu_t1 = omp_get_wtime();
    double cpu_seconds = cpu_t1 - cpu_t0;

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    // TODO: Allocate memory on the device for d_a, d_b, and d_c.

    double gpu_mem_t0 = omp_get_wtime();
    // TODO: Copy data from host arrays (h_a, h_b) to device arrays (d_a, d_b).
    // TODO: Set d_c to zero on the device.
    double gpu_mem_t1 = omp_get_wtime();

    // TODO: Determine optimal dim3 threads and dim3 blocks for the 2D grid.

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    // TODO: Call your kernel
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
    for (int i = 0; i < n * n; ++i) {
        float diff = fabsf(h_cpu[i] - h_gpu[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    const int ok = (max_abs_diff < 5e-2f);
    const double gpu_seconds = (double)gpu_ms / 1000.0;

    printf("[matmul-naive] matmul_n=%d\n", n);
    printf("[matmul-naive] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[matmul-naive] gpu_hip_time=%.6f s (%.3f ms)\n", gpu_seconds, gpu_ms);
    printf("[matmul-naive] gpu_mem_h2d_time=%.6f s\n", gpu_h2d_time);
    printf("[matmul-naive] gpu_mem_d2h_time=%.6f s\n", gpu_d2h_time);
    printf("[matmul-naive] gpu_mem_total_time=%.6f s\n", gpu_mem_time);
    printf("[matmul-naive] gpu_hip_total_time=%.6f s\n", gpu_seconds + gpu_mem_time);
    if (gpu_seconds > 0.0) {
        printf("[matmul-naive] speedup_kernel_only=%.2fx\n", cpu_seconds / gpu_seconds);
        printf("[matmul-naive] speedup_with_mem=%.2fx\n", cpu_seconds / (gpu_seconds + gpu_mem_time));
    }
    printf("[matmul-naive] max_abs_diff=%g | %s\n", max_abs_diff, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);
    // TODO: Free device memory.
    free(h_a);
    free(h_b);
    free(h_cpu);
    free(h_gpu);

    return ok ? 0 : 1;
}
