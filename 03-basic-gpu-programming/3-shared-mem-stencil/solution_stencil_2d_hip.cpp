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
    // Shared-memory tile: (blockDim.x x blockDim.y) + 1px halo on each side.
    // Current launch config is threads(32,8):
    //   - tx in [0..31] => need tx+2 in [0..33]
    //   - ty in [0..7]  => need ty+2 in [0..9]
    // We allocate a padded shared-memory array to reduce bank conflicts.
    // Shared layout indices:
    //   s_data[ty + 1][tx + 1], s_data[ty + 2][tx + 1], s_data[ty + 1][tx + 2]
    __shared__ float s_data[10][35];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    
    // Load data into shared memory with halo
    if (x < nx && y < ny) {
        s_data[ty + 1][tx + 1] = src[idx2d(y, x, nx)];
        
        // Load halos
        if (tx == 0 && x > 0) s_data[ty + 1][0] = src[idx2d(y, x - 1, nx)];
        if (tx == blockDim.x - 1 && x < nx - 1) s_data[ty + 1][tx + 2] = src[idx2d(y, x + 1, nx)];
        if (ty == 0 && y > 0) s_data[0][tx + 1] = src[idx2d(y - 1, x, nx)];
        if (ty == blockDim.y - 1 && y < ny - 1) s_data[ty + 2][tx + 1] = src[idx2d(y + 1, x, nx)];
    }
    
    __syncthreads();

    if (x < nx && y < ny) {
        if (y == 0 || y == ny - 1 || x == 0 || x == nx - 1) {
            dst[idx2d(y, x, nx)] = s_data[ty + 1][tx + 1];
        } else {
            dst[idx2d(y, x, nx)] =
                0.25f * (s_data[ty][tx + 1] + s_data[ty + 2][tx + 1] +
                         s_data[ty + 1][tx] + s_data[ty + 1][tx + 2]);
        }
    }
}

int main(int argc, char** argv) {
    int nx = 2048;
    int ny = 2048;
    int iters = 100;
    if (argc >= 3) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
    }
    if (argc >= 4) {
        iters = atoi(argv[3]);
    }
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
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);

    // Tuned block shape for shared-memory tiling: 32x8
    // (32x8 = 256 threads/block)
    dim3 threads(32, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x, (ny + threads.y - 1) / threads.y);

    // Warm-up steady-state: do multiple transfer+kernel+transfer loops (not timed)
    const int WARM_REPS = 3;
    const int WARM_ITERS = 1; // keep warm-up cheaper than full 'iters'
    for (int w = 0; w < WARM_REPS; ++w) {
        hipMemcpy(d_a, h_init, bytes, hipMemcpyHostToDevice);
        hipMemset(d_b, 0, bytes);

        for (int it = 0; it < WARM_ITERS; ++it) {
            stencil_step_hip<<<blocks, threads>>>(d_a, d_b, ny, nx);
            float *tmp = d_a;
            d_a = d_b;
            d_b = tmp;
        }
        hipDeviceSynchronize();
        hipMemcpy(h_gpu, d_a, bytes, hipMemcpyDeviceToHost);
    }

    // Timed H2D
    double gpu_mem_t0 = omp_get_wtime();
    hipMemcpy(d_a, h_init, bytes, hipMemcpyHostToDevice);
    hipMemset(d_b, 0, bytes);
    double gpu_mem_t1 = omp_get_wtime();
    const double gpu_h2d_time = gpu_mem_t1 - gpu_mem_t0;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int it = 0; it < iters; ++it) {
        stencil_step_hip<<<blocks, threads>>>(d_a, d_b, ny, nx);
        float *tmp = d_a;
        d_a = d_b;
        d_b = tmp;
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, start, stop);

    double gpu_mem_back_t0 = omp_get_wtime();
    hipMemcpy(h_gpu, d_a, bytes, hipMemcpyDeviceToHost);
    double gpu_mem_back_t1 = omp_get_wtime();
    
    const double gpu_d2h_time = gpu_mem_back_t1 - gpu_mem_back_t0;

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
    hipFree(d_a);
    hipFree(d_b);
    free(h_init);
    free(h_cpu_a);
    free(h_cpu_b);
    free(h_gpu);

    return ok ? 0 : 1;
}
