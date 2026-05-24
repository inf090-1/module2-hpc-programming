#include <hip/hip_runtime.h>

#include <math.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#ifndef TILE
#define TILE 16
#endif

// Exercise: Implement tiled matrix multiplication with coalesced global memory reads.
__global__ void matmul_tiled_shmem(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // TODO: Compute row/col indices.

    // TODO: For each tile, load A and B into shared memory (with bounds checks), __syncthreads().

    // TODO: Compute the partial dot product for the current tile.

    // TODO: Write C[row, col] from the accumulated result (with bounds checks).
}

static void cpu_reference_gemm_omp(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (double)A[m * K + k] * (double)B[k * N + n];
            }
            C[m * N + n] = (float)acc;
        }
    }
}

int main(void) {
    const int M = 256;
    const int N = 256;
    const int K = 256;

    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    float* h_A = (float*)std::malloc(bytesA);
    float* h_B = (float*)std::malloc(bytesB);
    float* h_cpu = (float*)std::malloc(bytesC);
    float* h_gpu = (float*)std::malloc(bytesC);

    if (!h_A || !h_B || !h_cpu || !h_gpu) {
        std::fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < M * K; ++i) h_A[i] = sinf(0.001f * i);
    for (int i = 0; i < K * N; ++i) h_B[i] = cosf(0.001f * i);

    double cpu_t0 = omp_get_wtime();
    cpu_reference_gemm_omp(h_A, h_B, h_cpu, M, N, K);
    double cpu_t1 = omp_get_wtime();
    double cpu_seconds = cpu_t1 - cpu_t0;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    // TODO: Allocate memory on the device for d_A, d_B, and d_C.

    double gpu_mem_t0 = omp_get_wtime();
    // TODO: Copy h_A -> d_A and h_B -> d_B.
    double gpu_mem_t1 = omp_get_wtime();

    // TODO: Determine optimal dim3 threads and dim3 blocks.
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    // TODO: Launch matmul_tiled_shmem kernel.
    // hipLaunchKernelGGL(matmul_tiled_shmem, blocks, threads, 0, 0, d_A, d_B, d_C, M, N, K);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, start, stop);
    double gpu_seconds = (double)gpu_ms / 1000.0;

    double gpu_mem_back_t0 = omp_get_wtime();
    // TODO: Copy d_C -> h_gpu.
    double gpu_mem_back_t1 = omp_get_wtime();

    const double gpu_h2d_time = (gpu_mem_t1 - gpu_mem_t0);
    const double gpu_d2h_time = (gpu_mem_back_t1 - gpu_mem_back_t0);
    double gpu_mem_time = gpu_h2d_time + gpu_d2h_time;

    double max_abs_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs((double)h_cpu[i] - (double)h_gpu[i]));
    }

    const double eps = 1e-2;
    const int ok = (max_abs_err <= eps);

    printf("[matmul-coalescing] M=%d N=%d K=%d\n", M, N, K);
    printf("[matmul-coalescing] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[matmul-coalescing] gpu_hip_time=%.6f s (%.3f ms)\n", gpu_seconds, gpu_ms);
    printf("[matmul-coalescing] gpu_mem_h2d_time=%.6f s\n", gpu_h2d_time);
    printf("[matmul-coalescing] gpu_mem_d2h_time=%.6f s\n", gpu_d2h_time);
    printf("[matmul-coalescing] gpu_mem_total_time=%.6f s\n", gpu_mem_time);
    printf("[matmul-coalescing] gpu_hip_total_time=%.6f s\n", gpu_seconds + gpu_mem_time);
    if (gpu_seconds > 0.0) {
        printf("[matmul-coalescing] speedup_kernel_only=%.2fx\n", cpu_seconds / gpu_seconds);
        printf("[matmul-coalescing] speedup_with_mem=%.2fx\n", cpu_seconds / (gpu_seconds + gpu_mem_time));
    }
    printf("[matmul-coalescing] max_abs_err=%g (eps=%g) | %s\n",
           max_abs_err, eps, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);

    // TODO: Free device memory.
    std::free(h_A);
    std::free(h_B);
    std::free(h_cpu);
    std::free(h_gpu);

    return ok ? 0 : 1;
}
