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
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) {
            acc += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = acc;
    }
}

int main(int argc, char** argv) {
    int n = 512;
    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    const size_t bytes = (size_t)n * n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_cpu = (float *)malloc(bytes);
    float *h_gpu = (float *)malloc(bytes);

    for (int i = 0; i < n * n; ++i) {
        h_a[i] = (float)((i % 17) - 8) * 0.125f;
        h_b[i] = (float)((i % 13) - 6) * 0.25f;
    }

    const char* cpu_max_n_env = getenv("CPU_MAX_N");
    const int cpu_max_n = cpu_max_n_env ? atoi(cpu_max_n_env) : 2048;
    const bool do_cpu = (n <= cpu_max_n);

    double cpu_seconds = 0.0;
    if (do_cpu) {
        double cpu_t0 = omp_get_wtime();
        matmul_cpu_omp(h_a, h_b, h_cpu, n);
        double cpu_t1 = omp_get_wtime();
        cpu_seconds = cpu_t1 - cpu_t0;
    } else {
        cpu_seconds = -1.0;
    }

    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    // Warm-up steady-state: do multiple transfer+kernel+transfer loops (not timed)
    const int WARM_REPS = 3;
    for (int w = 0; w < WARM_REPS; ++w) {
        hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
        hipMemset(d_c, 0, bytes);

        matmul_naive_hip<<<blocks, threads>>>(d_a, d_b, d_c, n);
        hipDeviceSynchronize();
        hipMemcpy(h_gpu, d_c, bytes, hipMemcpyDeviceToHost);
    }

    // Timed H2D
    double gpu_mem_t0 = omp_get_wtime();
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
    hipMemset(d_c, 0, bytes);
    double gpu_mem_t1 = omp_get_wtime();
    const double gpu_h2d_time = gpu_mem_t1 - gpu_mem_t0;

    hipStream_t stream;
    hipStreamCreate(&stream);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start, stream);
    matmul_naive_hip<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    hipStreamSynchronize(stream);

    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, start, stop);

    double gpu_mem_back_t0 = omp_get_wtime();
    hipMemcpy(h_gpu, d_c, bytes, hipMemcpyDeviceToHost);
    double gpu_mem_back_t1 = omp_get_wtime();
    const double gpu_d2h_time = gpu_mem_back_t1 - gpu_mem_back_t0;

    double gpu_mem_time = gpu_h2d_time + gpu_d2h_time;

    float max_abs_diff = 0.0f;
    if (do_cpu) {
        for (int i = 0; i < n * n; ++i) {
            float diff = fabsf(h_cpu[i] - h_gpu[i]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
        }
    } else {
        max_abs_diff = -1.0f;
    }

    const int ok = do_cpu ? (max_abs_diff < 5e-2f) : 1;
    const double gpu_seconds = (double)gpu_ms / 1000.0;

    printf("[matmul-naive] matmul_n=%d\n", n);
    if (cpu_seconds >= 0.0) {
        printf("[matmul-naive] cpu_openmp_time=%.6f s\n", cpu_seconds);
    } else {
        printf("[matmul-naive] cpu_openmp_time=SKIPPED (CPU_MAX_N=%d)\n", cpu_max_n);
    }
    printf("[matmul-naive] gpu_hip_time=%.6f s (%.3f ms)\n", gpu_seconds, gpu_ms);
    printf("[matmul-naive] gpu_mem_h2d_time=%.6f s\n", gpu_h2d_time);
    printf("[matmul-naive] gpu_mem_d2h_time=%.6f s\n", gpu_d2h_time);
    printf("[matmul-naive] gpu_mem_total_time=%.6f s\n", gpu_mem_time);
    printf("[matmul-naive] gpu_hip_total_time=%.6f s\n", gpu_seconds + gpu_mem_time);
    if (gpu_seconds > 0.0 && cpu_seconds >= 0.0) {
        printf("[matmul-naive] speedup_kernel_only=%.2fx\n", cpu_seconds / gpu_seconds);
        printf("[matmul-naive] speedup_with_mem=%.2fx\n", cpu_seconds / (gpu_seconds + gpu_mem_time));
    }
    printf("[matmul-naive] max_abs_diff=%g | %s\n", max_abs_diff, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_cpu);
    free(h_gpu);

    return ok ? 0 : 1;
}
