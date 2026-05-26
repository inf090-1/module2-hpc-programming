#include <hip/hip_runtime.h>

#include <omp.h>

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#ifndef MAT_DIM
#define MAT_DIM 128
#endif

#ifndef TILE
#define TILE 16
#endif

__global__ void matmul_hip_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        const int a_col = k0 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        const int b_row = k0 + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            acc += As[threadIdx.y][t] * Bs[t][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

static void matmul_hip(const float* A, const float* B, float* C,
                       int M, int N, int K,
                       double* out_alloc_transfer_seconds,
                       double* out_kernel_seconds)
{
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    double t0 = omp_get_wtime();

    hipMalloc((void**)&d_A, (size_t)M * K * sizeof(float));
    hipMalloc((void**)&d_B, (size_t)K * N * sizeof(float));
    hipMalloc((void**)&d_C, (size_t)M * N * sizeof(float));

    hipMemcpy(d_A, A, (size_t)M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, (size_t)K * N * sizeof(float), hipMemcpyHostToDevice);

    double t_after_h2d = omp_get_wtime();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    hipEventRecord(start);
    hipLaunchKernelGGL(matmul_hip_kernel, blocks, threads, 0, 0, d_A, d_B, d_C, M, N, K);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);
    *out_kernel_seconds = (double)ms / 1000.0;

    double t_before_d2h = omp_get_wtime();
    hipMemcpy(C, d_C, (size_t)M * N * sizeof(float), hipMemcpyDeviceToHost);
    double t_after_d2h = omp_get_wtime();

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    hipEventDestroy(start);
    hipEventDestroy(stop);

    *out_alloc_transfer_seconds = (t_after_h2d - t0) + (t_after_d2h - t_before_d2h);
}

static void init_matrix(float* X, int rows, int cols, float scale) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            X[idx] = scale * (sinf(0.001f * idx) + cosf(0.002f * idx));
        }
    }
}

static void matmul_cpu_serial(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (double)A[i * K + k] * (double)B[k * N + j];
            }
            C[i * N + j] = (float)acc;
        }
    }
}

int main(int argc, char** argv) {
    int dim = MAT_DIM;
    if (argc >= 2) {
        dim = std::max(1, atoi(argv[1]));
    }

    const int M = dim;
    const int N = dim;
    const int K = dim;

    std::vector<float> A((size_t)M * K);
    std::vector<float> B((size_t)K * N);
    std::vector<float> C_hip((size_t)M * N);
    std::vector<float> C_serial;

    init_matrix(A.data(), M, K, 1.0f);
    init_matrix(B.data(), K, N, 1.0f);

    double t_cpu_serial = 0.0;
    if (dim <= 256) {
        C_serial.resize((size_t)M * N);
        double t_cpu0 = omp_get_wtime();
        matmul_cpu_serial(A.data(), B.data(), C_serial.data(), M, N, K);
        double t_cpu1 = omp_get_wtime();
        t_cpu_serial = t_cpu1 - t_cpu0;
    }

    double hip_alloc_transfer_seconds = 0.0;
    double hip_kernel_seconds = 0.0;
    matmul_hip(A.data(), B.data(), C_hip.data(), M, N, K,
               &hip_alloc_transfer_seconds, &hip_kernel_seconds);
    double hip_total_seconds = hip_alloc_transfer_seconds + hip_kernel_seconds;

    double checksum = 0.0;
    for (int idx = 0; idx < M * N; ++idx) checksum += (double)C_hip[idx];

    printf("[hip-matmul] MAT_DIM=%d\n", dim);
    printf("[hip-matmul] hip_alloc_transfer_time=%.6f s\n", hip_alloc_transfer_seconds);
    printf("[hip-matmul] hip_kernel_time=%.6f s\n", hip_kernel_seconds);
    printf("[hip-matmul] hip_total_time=%.6f s\n", hip_total_seconds);
    printf("[hip-matmul] checksum=%g\n", checksum);
    if (t_cpu_serial > 0.0 && hip_total_seconds > 0.0) {
        printf("[hip-matmul] cpu_serial_time=%.6f s\n", t_cpu_serial);
        printf("[hip-matmul] speedup_serial_vs_hip=%.2fx\n", t_cpu_serial / hip_total_seconds);
    }

    return 0;
}
