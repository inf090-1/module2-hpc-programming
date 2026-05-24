#include <hip/hip_runtime.h>

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#ifndef TILE
#define TILE 16
#endif

__global__ void matmul_tiled_shmem(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    const int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        const int aRow = row;
        const int aCol = t * TILE + threadIdx.x;
        const int bRow = t * TILE + threadIdx.y;
        const int bCol = col;

        As[threadIdx.y][threadIdx.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
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

int main(int argc, char** argv) {
    int n = 256;
    if (argc > 1) n = std::atoi(argv[1]);
    if (n <= 0) n = 256;

    const int M = n;
    const int N = n;
    const int K = n;

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

    for (int i = 0; i < M * K; ++i) h_A[i] = std::sin(0.001f * i);
    for (int i = 0; i < K * N; ++i) h_B[i] = std::cos(0.001f * i);

    double cpu_t0 = omp_get_wtime();
    cpu_reference_gemm_omp(h_A, h_B, h_cpu, M, N, K);
    double cpu_t1 = omp_get_wtime();
    const double cpu_seconds = cpu_t1 - cpu_t0;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    hipMalloc(&d_A, bytesA);
    hipMalloc(&d_B, bytesB);
    hipMalloc(&d_C, bytesC);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // Warm-up: includes transfer + kernel + output copy once.
    hipMemcpy(d_A, h_A, bytesA, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytesB, hipMemcpyHostToDevice);
    matmul_tiled_shmem<<<blocks, threads, 0, 0>>>(d_A, d_B, d_C, M, N, K);
    hipDeviceSynchronize();
    hipMemcpy(h_gpu, d_C, bytesC, hipMemcpyDeviceToHost);

    const int warmup_iters = 1;
    const int iters = 5;

    (void)warmup_iters; // warm-up already performed above; keep variable for future extensions.

    double sum_h2d = 0.0;
    double sum_d2h = 0.0;
    double sum_kernel = 0.0;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    for (int rep = 0; rep < iters; ++rep) {
        const double t_h2d0 = omp_get_wtime();
        hipMemcpy(d_A, h_A, bytesA, hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B, bytesB, hipMemcpyHostToDevice);
        const double t_h2d1 = omp_get_wtime();

        hipEventRecord(start);
        hipLaunchKernelGGL(matmul_tiled_shmem, blocks, threads, 0, 0, d_A, d_B, d_C, M, N, K);
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float gpu_ms = 0.0f;
        hipEventElapsedTime(&gpu_ms, start, stop);
        const double kernel_seconds = (double)gpu_ms / 1000.0;

        const double t_d2h0 = omp_get_wtime();
        hipMemcpy(h_gpu, d_C, bytesC, hipMemcpyDeviceToHost);
        const double t_d2h1 = omp_get_wtime();

        sum_h2d += (t_h2d1 - t_h2d0);
        sum_d2h += (t_d2h1 - t_d2h0);
        sum_kernel += kernel_seconds;
    }

    const double avg_h2d = sum_h2d / iters;
    const double avg_d2h = sum_d2h / iters;
    const double avg_kernel = sum_kernel / iters;

    const double avg_mem_time = avg_h2d + avg_d2h;
    const double avg_gpu_total_kernel = avg_kernel + avg_mem_time;

    // correctness using last h_gpu from final repetition
    double max_abs_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs((double)h_cpu[i] - (double)h_gpu[i]));
    }

    const double eps = 1e-2;
    const int ok = (max_abs_err <= eps);

    // Reuse avg_kernel to compute speedup, but also print avg_kernel ms.
    const double avg_kernel_ms = avg_kernel * 1000.0;

    printf("[matmul-coalescing] M=%d N=%d K=%d\n", M, N, K);
    printf("[matmul-coalescing] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[matmul-coalescing] gpu_hip_time=%.6f s (%.3f ms)\n", avg_kernel, avg_kernel_ms);
    printf("[matmul-coalescing] gpu_mem_h2d_time=%.6f s\n", avg_h2d);
    printf("[matmul-coalescing] gpu_mem_d2h_time=%.6f s\n", avg_d2h);
    printf("[matmul-coalescing] gpu_mem_total_time=%.6f s\n", avg_mem_time);
    printf("[matmul-coalescing] gpu_hip_total_time=%.6f s\n", avg_gpu_total_kernel);

    if (avg_kernel > 0.0) {
        printf("[matmul-coalescing] speedup_kernel_only=%.2fx\n", cpu_seconds / avg_kernel);
        printf("[matmul-coalescing] speedup_with_mem=%.2fx\n", cpu_seconds / avg_gpu_total_kernel);
    }

    printf("[matmul-coalescing] max_abs_err=%g (eps=%g) | %s\n",
           max_abs_err, eps, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    std::free(h_A);
    std::free(h_B);
    std::free(h_cpu);
    std::free(h_gpu);

    return ok ? 0 : 1;
}
