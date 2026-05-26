#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

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

static void matmul_omp_target(const float* A, const float* B_T, float* C,
                              int M, int N, int K,
                              double* out_seconds) {
    double t0 = omp_get_wtime();

    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: A[0:M * K], B_T[0:N * K]) \
        map(from: C[0:M * N])
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const int a_row = i * K;
            const int b_row = j * K;
            for (int k = 0; k < K; ++k) {
                acc += A[a_row + k] * B_T[b_row + k];
            }
            C[i * N + j] = acc;
        }
    }

    double t1 = omp_get_wtime();
    *out_seconds = t1 - t0;
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
    std::vector<float> B_T((size_t)N * K);
    std::vector<float> C_omp((size_t)M * N);
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

    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            B_T[j * K + k] = B[k * N + j];
        }
    }

    double omp_target_seconds = 0.0;
    matmul_omp_target(A.data(), B_T.data(), C_omp.data(), M, N, K, &omp_target_seconds);

    double checksum = 0.0;
    for (int idx = 0; idx < M * N; ++idx) checksum += (double)C_omp[idx];

    printf("[omp-target-matmul-solution] MAT_DIM=%d\n", dim);
    printf("[omp-target-matmul-solution] omp_target_time=%.6f s\n", omp_target_seconds);
    printf("[omp-target-matmul-solution] checksum=%g\n", checksum);
    if (t_cpu_serial > 0.0 && omp_target_seconds > 0.0) {
        printf("[omp-target-matmul-solution] cpu_serial_time=%.6f s\n", t_cpu_serial);
        printf("[omp-target-matmul-solution] speedup_serial_vs_target=%.2fx\n",
               t_cpu_serial / omp_target_seconds);
    }

    return 0;
}
