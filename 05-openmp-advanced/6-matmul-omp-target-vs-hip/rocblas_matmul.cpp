#include <hip/hip_runtime.h>

#include <rocblas/rocblas.h>

#include <omp.h>

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

#ifndef MAT_DIM
#define MAT_DIM 128
#endif

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
    std::vector<float> C((size_t)M * N);
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

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    hipMalloc((void**)&d_A, bytesA);
    hipMalloc((void**)&d_B, bytesB);
    hipMalloc((void**)&d_C, bytesC);

    hipMemcpy(d_A, A.data(), bytesA, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), bytesB, hipMemcpyHostToDevice);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    double t_rocblas0 = omp_get_wtime();
    rocblas_status st = rocblas_sgemm(
        handle,
        rocblas_operation_none, rocblas_operation_none,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        /*A=*/d_B, /*lda=*/N,
        /*B=*/d_A, /*ldb=*/K,
        &beta,
        /*C=*/d_C, /*ldc=*/N);

    if (st != rocblas_status_success) {
        fprintf(stderr, "rocblas_sgemm failed: %d\n", (int)st);
        return 2;
    }

    hipDeviceSynchronize();
    double t_rocblas1 = omp_get_wtime();
    double t_rocblas = t_rocblas1 - t_rocblas0;
    hipMemcpy(C.data(), d_C, bytesC, hipMemcpyDeviceToHost);

    rocblas_destroy_handle(handle);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    double checksum = 0.0;
    for (int idx = 0; idx < M * N; ++idx) checksum += (double)C[idx];

    printf("[rocblas-matmul] MAT_DIM=%d\n", dim);
    printf("[rocblas-matmul] rocblas_time=%.6f s\n", t_rocblas);
    printf("[rocblas-matmul] checksum=%g\n", checksum);
    if (t_cpu_serial > 0.0) {
        printf("[rocblas-matmul] cpu_serial_time=%.6f s\n", t_cpu_serial);
    }

    return 0;
}
