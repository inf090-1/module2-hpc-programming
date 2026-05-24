#include <hip/hip_runtime.h>

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#ifndef SEQ_LEN
#define SEQ_LEN 8
#endif

#ifndef EMBED_DIM
#define EMBED_DIM 16
#endif

// Scores: S[i,j] = dot(Q[i,*], K[j,*]) / sqrt(EMBED_DIM)
__global__ void compute_scores(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 float* __restrict__ S) {
    int i = blockIdx.y;   // query index
    int j = threadIdx.x; // key index

    if (j >= SEQ_LEN) return;

    // TODO: Compute dot product for (i,j) and scale by 1/sqrt(EMBED_DIM).
    // TODO: Write S[i*SEQ_LEN + j].
}

// Softmax in-place over each row i of S.
__global__ void softmax_rows(float* __restrict__ S) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (j >= SEQ_LEN) return;

    // TODO: Compute softmax for row i.
    // TODO: Write normalized S[i*SEQ_LEN + j].
}

// Output O[i,d] = sum_j S[i,j] * V[j,d]
__global__ void compute_output(const float* __restrict__ S,
                                 const float* __restrict__ V,
                                 float* __restrict__ O) {
    int i = blockIdx.y;
    int d = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= SEQ_LEN || d >= EMBED_DIM) return;

    // TODO: Compute weighted sum for (i,d) and write O[i*EMBED_DIM + d].
}

static void cpu_attention_omp(const float* Q, const float* K, const float* V, float* O) {
    const float scale = 1.0f / std::sqrt((float)EMBED_DIM);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < SEQ_LEN; ++i) {
        float scores[SEQ_LEN];

        float m = -1e30f;
        for (int j = 0; j < SEQ_LEN; ++j) {
            float acc = 0.0f;
            for (int dd = 0; dd < EMBED_DIM; ++dd) {
                acc += Q[i * EMBED_DIM + dd] * K[j * EMBED_DIM + dd];
            }
            acc *= scale;
            scores[j] = acc;
            m = fmaxf(m, acc);
        }

        float sum = 0.0f;
        for (int j = 0; j < SEQ_LEN; ++j) {
            scores[j] = expf(scores[j] - m);
            sum += scores[j];
        }

        for (int dd = 0; dd < EMBED_DIM; ++dd) {
            float acc = 0.0f;
            for (int j = 0; j < SEQ_LEN; ++j) {
                float p = scores[j] / sum;
                acc += p * V[j * EMBED_DIM + dd];
            }
            O[i * EMBED_DIM + dd] = acc;
        }
    }
}

int main(void) {
    const size_t n = (size_t)SEQ_LEN * EMBED_DIM;
    const size_t nS = (size_t)SEQ_LEN * SEQ_LEN;

    float* h_Q = (float*)std::malloc(n * sizeof(float));
    float* h_K = (float*)std::malloc(n * sizeof(float));
    float* h_V = (float*)std::malloc(n * sizeof(float));
    float* h_O = (float*)std::malloc(n * sizeof(float));
    float* h_ref = (float*)std::malloc(n * sizeof(float));

    if (!h_Q || !h_K || !h_V || !h_O || !h_ref) {
        std::fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < n; ++i) {
        h_Q[i] = sinf(0.01f * (float)i);
        h_K[i] = cosf(0.01f * (float)i);
        h_V[i] = 0.1f * (float)(i % 17);
    }

    double cpu_t0 = omp_get_wtime();
    cpu_attention_omp(h_Q, h_K, h_V, h_ref);
    double cpu_t1 = omp_get_wtime();
    double cpu_seconds = cpu_t1 - cpu_t0;

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_S = nullptr, *d_O = nullptr;
    // TODO: Allocate device memory for d_Q, d_K, d_V, d_S, d_O.

    double gpu_mem_t0 = omp_get_wtime();
    // TODO: Copy d_Q <- h_Q, d_K <- h_K, d_V <- h_V.
    double gpu_mem_t1 = omp_get_wtime();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    // TODO: Launch compute_scores, softmax_rows, compute_output in the correct order.
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, start, stop);
    double gpu_seconds = (double)gpu_ms / 1000.0;

    double gpu_mem_back_t0 = omp_get_wtime();
    // TODO: Copy h_O <- d_O.
    double gpu_mem_back_t1 = omp_get_wtime();

    const double gpu_h2d_time = (gpu_mem_t1 - gpu_mem_t0);
    const double gpu_d2h_time = (gpu_mem_back_t1 - gpu_mem_back_t0);
    double gpu_mem_time = gpu_h2d_time + gpu_d2h_time;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    double max_abs_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs((double)h_O[i] - (double)h_ref[i]));
    }

    const double eps = 1e-2;
    const int ok = (max_abs_err <= eps);

    printf("[attention] SEQ_LEN=%d EMBED_DIM=%d\n", SEQ_LEN, EMBED_DIM);
    printf("[attention] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[attention] gpu_hip_time=%.6f s (%.3f ms)\n", gpu_seconds, gpu_ms);
    printf("[attention] gpu_mem_h2d_time=%.6f s\n", gpu_h2d_time);
    printf("[attention] gpu_mem_d2h_time=%.6f s\n", gpu_d2h_time);
    printf("[attention] gpu_mem_total_time=%.6f s\n", gpu_mem_time);
    printf("[attention] gpu_hip_total_time=%.6f s\n", gpu_seconds + gpu_mem_time);
    if (gpu_seconds > 0.0) {
        printf("[attention] speedup_kernel_only=%.2fx\n", cpu_seconds / gpu_seconds);
        printf("[attention] speedup_with_mem=%.2fx\n", cpu_seconds / (gpu_seconds + gpu_mem_time));
    }
    printf("[attention] max_abs_err=%g (eps=%g) | %s\n", max_abs_err, eps, ok ? "PASS" : "FAIL");

    // TODO: Free device memory.
    std::free(h_Q);
    std::free(h_K);
    std::free(h_V);
    std::free(h_O);
    std::free(h_ref);

    return ok ? 0 : 1;
}
