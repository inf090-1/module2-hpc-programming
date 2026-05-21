#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define ROCBLAS_CHECK(cmd)                                                      \
    do {                                                                        \
        rocblas_status status__ = (cmd);                                        \
        if (status__ != rocblas_status_success) {                               \
            fprintf(stderr, "rocBLAS error at %s:%d: %s failed\n",              \
                    __FILE__, __LINE__, #cmd);                                  \
            return 1;                                                           \
        }                                                                       \
    } while (0)

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

int main(int argc, char** argv) {
    // We will test multiple matrix sizes
    std::vector<int> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            sizes.push_back(atoi(argv[i]));
        }
    } else {
        sizes = {512, 1024, 2048, 4096};
    }

    for (int s = 0; s < (int)sizes.size(); s++) {
        const int n = sizes[s];
        const size_t bytes = (size_t)n * n * sizeof(float);

        float *h_a = (float *)malloc(bytes);
        float *h_b = (float *)malloc(bytes);
        float *h_cpu = (float *)malloc(bytes);
        float *h_gpu = (float *)malloc(bytes);

        // rocBLAS expects column-major by default, but we'll use NT or TN tricks to emulate row-major
        // To compute C = A * B in row major, it's equivalent to C^T = B^T * A^T in col major
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


        // Setup rocBLAS
        rocblas_handle handle;
        ROCBLAS_CHECK(rocblas_create_handle(&handle));

        // Use a dedicated HIP stream and bind rocBLAS to it so our hip events
        // measure the correct work.
        hipStream_t stream;
        hipStreamCreate(&stream);
        ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Warm-up steady-state: multiple transfer+rocBLAS+transfer loops (not timed)
        const int WARM_REPS = 3;
        for (int w = 0; w < WARM_REPS; ++w) {
            hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
            hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
            hipMemset(d_c, 0, bytes);

            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n));

            hipDeviceSynchronize();
            hipMemcpy(h_gpu, d_c, bytes, hipMemcpyDeviceToHost);
        }

        // Timed H2D
        double mem_h2d_t0 = omp_get_wtime();
        hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
        hipMemset(d_c, 0, bytes);
        double mem_h2d_t1 = omp_get_wtime();
        const double gpu_h2d_time = mem_h2d_t1 - mem_h2d_t0;

        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        // Ensure previous iterations are finished before timing.
        hipDeviceSynchronize();
        hipEventRecord(start, stream);
        
        // rocblas_sgemm computes C = alpha * op(A) * op(B) + beta * C
        // Since rocblas is column-major and our data is row-major:
        // C_row = A_row * B_row
        // C_col^T = B_col^T * A_col^T
        // B^T = rocblas_operation_none with lda=n, A^T = rocblas_operation_none with ldb=n
        ROCBLAS_CHECK(rocblas_sgemm(
            handle, 
            rocblas_operation_none, rocblas_operation_none,
            n, n, n, 
            &alpha, 
            d_b, n, 
            d_a, n, 
            &beta, 
            d_c, n));
            
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float gpu_ms = 0.0f;
        hipEventElapsedTime(&gpu_ms, start, stop);
        const double gpu_seconds = (double)gpu_ms / 1000.0;

        double mem_d2h_t0 = omp_get_wtime();
        hipMemcpy(h_gpu, d_c, bytes, hipMemcpyDeviceToHost);
        double mem_d2h_t1 = omp_get_wtime();
        const double gpu_d2h_time = mem_d2h_t1 - mem_d2h_t0;

        const double gpu_mem_time = gpu_h2d_time + gpu_d2h_time;

        float max_abs_diff = 0.0f;
        if (do_cpu && n <= 2048) {
            for (int i = 0; i < n * n; ++i) {
                float diff = fabsf(h_cpu[i] - h_gpu[i]);
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                }
            }
        } else {
            max_abs_diff = -1.0f;
        }

        printf("----------------------------------------\n");
        printf("[rocblas-matmul] rocBLAS matmul_n=%d\n", n);
        if (cpu_seconds >= 0.0) {
            printf("[rocblas-matmul] cpu_openmp_time=%.6f s\n", cpu_seconds);
        } else {
            printf("[rocblas-matmul] cpu_openmp_time=SKIPPED (CPU_MAX_N=%d)\n", cpu_max_n);
        }
        printf("[rocblas-matmul] gpu_rocblas_time=%.6f s (%.3f ms)\n", gpu_seconds, gpu_ms);
        printf("[rocblas-matmul] gpu_mem_h2d_time=%.6f s\n", gpu_h2d_time);
        printf("[rocblas-matmul] gpu_mem_d2h_time=%.6f s\n", gpu_d2h_time);
        printf("[rocblas-matmul] gpu_mem_total_time=%.6f s\n", gpu_mem_time);
        printf("[rocblas-matmul] gpu_total_time(rocblas+mem)=%.6f s\n", gpu_seconds + gpu_mem_time);

        // calculate tflops
        double gflops = (2.0 * n * n * n) / (gpu_seconds * 1e9);
        printf("[rocblas-matmul] performance=%.2f GFLOPS\n", gflops);
        
        if (gpu_seconds > 0.0 && cpu_seconds >= 0.0) {
            printf("[rocblas-matmul] speedup=%.2fx\n", cpu_seconds / gpu_seconds);
        }
        if (do_cpu && n <= 2048) {
            const int ok = (max_abs_diff < 5e-2f);
            printf("[rocblas-matmul] max_abs_diff=%g | %s\n", max_abs_diff, ok ? "PASS" : "FAIL");
        } else {
            printf("[rocblas-matmul] max_abs_diff=SKIPPED\n");
        }

        rocblas_destroy_handle(handle);
        hipStreamDestroy(stream);
        hipEventDestroy(start);
        hipEventDestroy(stop);
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
        free(h_a);
        free(h_b);
        free(h_cpu);
        free(h_gpu);
    }

    return 0;
}
