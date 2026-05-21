#include <hip/hip_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

__host__ __device__ static inline int idx2d(int y, int x, int nx) {
    return y * nx + x;
}

static void stencil_step_cpu(const float *src, float *dst, int ny, int nx) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (y == 0 || y == ny - 1 || x == 0 || x == nx - 1) {
                dst[idx2d(y, x, nx)] = src[idx2d(y, x, nx)];
            } else {
                dst[idx2d(y, x, nx)] =
                    0.25f * (src[idx2d(y - 1, x, nx)] + src[idx2d(y + 1, x, nx)] +
                             src[idx2d(y, x - 1, nx)] + src[idx2d(y, x + 1, nx)]);
            }
        }
    }
}

__global__ void stencil_step_hip(const float *src, float *dst, int ny, int nx) {
    // TODO: Use shared memory (e.g. 18x18 array for 16x16 block + halo) to optimize the stencil.
    // TODO: Compute x/y from block and thread indices.
    // TODO: Load interior and halo cells into shared memory.
    // TODO: Synchronize.
    // TODO: Compute the 4-point stencil using shared memory.
}

int main(void) {
    const int nx = 2048;
    const int ny = 2048;
    const int iters = 100;
    const int n = nx * ny;
    const size_t bytes = (size_t)n * sizeof(float);

    float *h_init = (float *)malloc(bytes);
    float *h_cpu_a = (float *)malloc(bytes);
    float *h_cpu_b = (float *)malloc(bytes);
    float *h_gpu = (float *)malloc(bytes);

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            h_init[idx2d(y, x, nx)] = (float)((x + y) % 97) / 97.0f;
        }
    }

    for (int i = 0; i < n; ++i) {
        h_cpu_a[i] = h_init[i];
        h_cpu_b[i] = 0.0f;
    }

    double cpu_t0 = omp_get_wtime();
    for (int it = 0; it < iters; ++it) {
        stencil_step_cpu(h_cpu_a, h_cpu_b, ny, nx);
        float *tmp = h_cpu_a;
        h_cpu_a = h_cpu_b;
        h_cpu_b = tmp;
    }
    double cpu_t1 = omp_get_wtime();
    double cpu_seconds = cpu_t1 - cpu_t0;

    float *d_a = NULL;
    float *d_b = NULL;
    // TODO: Allocate memory on the device for d_a and d_b.

    double gpu_mem_t0 = omp_get_wtime();
    // TODO: Copy data from h_init to d_a.
    // TODO: Set d_b to zero.
    double gpu_mem_t1 = omp_get_wtime();

    // TODO: Determine optimal dim3 threads and dim3 blocks for the 2D grid.
    
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int it = 0; it < iters; ++it) {
        // TODO: Call your kernel
        
        float *tmp = d_a;
        d_a = d_b;
        d_b = tmp;
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, start, stop);

    double gpu_mem_back_t0 = omp_get_wtime();
    // TODO: Copy result from d_a back to h_gpu.
    double gpu_mem_back_t1 = omp_get_wtime();
    
    const double gpu_h2d_time = (gpu_mem_t1 - gpu_mem_t0);
    const double gpu_d2h_time = (gpu_mem_back_t1 - gpu_mem_back_t0);
    double gpu_mem_time = gpu_h2d_time + gpu_d2h_time;

    float max_abs_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(h_cpu_a[i] - h_gpu[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    const int ok = (max_abs_diff < 1e-4f);
    const double gpu_seconds = (double)gpu_ms / 1000.0;

    printf("[shmem-stencil] grid=%dx%d iters=%d\n", nx, ny, iters);
    printf("[shmem-stencil] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[shmem-stencil] gpu_hip_time=%.6f s (%.3f ms)\n", gpu_seconds, gpu_ms);
    printf("[shmem-stencil] gpu_mem_h2d_time=%.6f s\n", gpu_h2d_time);
    printf("[shmem-stencil] gpu_mem_d2h_time=%.6f s\n", gpu_d2h_time);
    printf("[shmem-stencil] gpu_mem_total_time=%.6f s\n", gpu_mem_time);
    printf("[shmem-stencil] gpu_hip_total_time=%.6f s\n", gpu_seconds + gpu_mem_time);
    if (gpu_seconds > 0.0) {
        printf("[shmem-stencil] speedup_kernel_only=%.2fx\n", cpu_seconds / gpu_seconds);
        printf("[shmem-stencil] speedup_with_mem=%.2fx\n", cpu_seconds / (gpu_seconds + gpu_mem_time));
    }
    printf("[shmem-stencil] max_abs_diff=%g | %s\n", max_abs_diff, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);
    // TODO: Free device memory.
    free(h_init);
    free(h_cpu_a);
    free(h_cpu_b);
    free(h_gpu);

    return ok ? 0 : 1;
}
