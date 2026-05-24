#include <hip/hip_runtime.h>

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

#ifndef SEQ_LEN
#define SEQ_LEN 8
#endif

#ifndef EMBED_DIM
#define EMBED_DIM 16
#endif

// Scores: S[i,j] = dot(Q[i,*], K[j,*]) / sqrt(embed_dim)
__global__ void compute_scores(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 float* __restrict__ S,
                                 int embed_dim) {
    int i = blockIdx.y;
    int j = threadIdx.x;

    if (j >= SEQ_LEN) return;

    float acc = 0.0f;
    for (int d = 0; d < embed_dim; ++d) {
        acc += Q[i * embed_dim + d] * K[j * embed_dim + d];
    }

    const float scale = 1.0f / sqrtf((float)embed_dim);
    S[i * SEQ_LEN + j] = acc * scale;
}

// Softmax in-place over each row i of S.
__global__ void softmax_rows(float* __restrict__ S) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (j >= SEQ_LEN) return;

    __shared__ float rowMax;
    __shared__ float rowSum;
    __shared__ float tmp[SEQ_LEN];

    float x = S[i * SEQ_LEN + j];
    tmp[j] = x;

    if (j == 0) {
        float m = tmp[0];
        for (int t = 1; t < SEQ_LEN; ++t) m = fmaxf(m, tmp[t]);
        rowMax = m;
    }
    __syncthreads();

    float e = expf(x - rowMax);
    tmp[j] = e;

    if (j == 0) {
        float s = 0.0f;
        for (int t = 0; t < SEQ_LEN; ++t) s += tmp[t];
        rowSum = s;
    }
    __syncthreads();

    S[i * SEQ_LEN + j] = e / rowSum;
}

// Output O[i,d] = sum_j S[i,j] * V[j,d]
__global__ void compute_output(const float* __restrict__ S,
                                 const float* __restrict__ V,
                                 float* __restrict__ O,
                                 int embed_dim) {
    int i = blockIdx.y;
    int d = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= SEQ_LEN || d >= embed_dim) return;

    float acc = 0.0f;
    for (int j = 0; j < SEQ_LEN; ++j) {
        float p = S[i * SEQ_LEN + j];
        acc += p * V[j * embed_dim + d];
    }

    O[i * embed_dim + d] = acc;
}

static void cpu_attention_omp(const float* Q, const float* K, const float* V, float* O,
                               int embed_dim) {
    const float scale = 1.0f / std::sqrt((float)embed_dim);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < SEQ_LEN; ++i) {
        float scores[SEQ_LEN];

        float m = -1e30f;
        for (int j = 0; j < SEQ_LEN; ++j) {
            float acc = 0.0f;
            for (int dd = 0; dd < embed_dim; ++dd) {
                acc += Q[i * embed_dim + dd] * K[j * embed_dim + dd];
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

        for (int dd = 0; dd < embed_dim; ++dd) {
            float acc = 0.0f;
            for (int j = 0; j < SEQ_LEN; ++j) {
                float p = scores[j] / sum;
                acc += p * V[j * embed_dim + dd];
            }
            O[i * embed_dim + dd] = acc;
        }
    }
}

int main(int argc, char** argv) {
    int embed_dim = EMBED_DIM;
    if (argc > 1) embed_dim = std::atoi(argv[1]);
    if (embed_dim <= 0) embed_dim = EMBED_DIM;

    const size_t n = (size_t)SEQ_LEN * (size_t)embed_dim;
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
    cpu_attention_omp(h_Q, h_K, h_V, h_ref, embed_dim);
    double cpu_t1 = omp_get_wtime();
    const double cpu_seconds = cpu_t1 - cpu_t0;

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_S = nullptr, *d_O = nullptr;

    hipMalloc(&d_Q, n * sizeof(float));
    hipMalloc(&d_K, n * sizeof(float));
    hipMalloc(&d_V, n * sizeof(float));
    hipMalloc(&d_S, nS * sizeof(float));
    hipMalloc(&d_O, n * sizeof(float));

    dim3 block_scores(SEQ_LEN, 1, 1);
    dim3 block_out(64, 1, 1);
    dim3 grid_out((embed_dim + block_out.x - 1) / block_out.x, SEQ_LEN, 1);

    // Warm-up: includes H2D + kernels + D2H once.
    hipMemcpy(d_Q, h_Q, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, h_K, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, h_V, n * sizeof(float), hipMemcpyHostToDevice);

    compute_scores<<<dim3(1, SEQ_LEN, 1), SEQ_LEN>>>(d_Q, d_K, d_S, embed_dim);
    softmax_rows<<<SEQ_LEN, SEQ_LEN>>>(d_S);
    compute_output<<<grid_out, block_out>>>(d_S, d_V, d_O, embed_dim);
    hipDeviceSynchronize();
    hipMemcpy(h_O, d_O, n * sizeof(float), hipMemcpyDeviceToHost);

    const int iters = 5;

    double sum_h2d = 0.0;
    double sum_d2h = 0.0;
    double sum_kernel = 0.0;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    for (int rep = 0; rep < iters; ++rep) {
        const double t_h2d0 = omp_get_wtime();
        hipMemcpy(d_Q, h_Q, n * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_K, h_K, n * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_V, h_V, n * sizeof(float), hipMemcpyHostToDevice);
        const double t_h2d1 = omp_get_wtime();

        hipEventRecord(start);
        compute_scores<<<dim3(1, SEQ_LEN, 1), SEQ_LEN>>>(d_Q, d_K, d_S, embed_dim);
        softmax_rows<<<SEQ_LEN, SEQ_LEN>>>(d_S);
        compute_output<<<grid_out, block_out>>>(d_S, d_V, d_O, embed_dim);
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float gpu_ms = 0.0f;
        hipEventElapsedTime(&gpu_ms, start, stop);
        const double kernel_seconds = (double)gpu_ms / 1000.0;

        const double t_d2h0 = omp_get_wtime();
        hipMemcpy(h_O, d_O, n * sizeof(float), hipMemcpyDeviceToHost);
        const double t_d2h1 = omp_get_wtime();

        sum_h2d += (t_h2d1 - t_h2d0);
        sum_d2h += (t_d2h1 - t_d2h0);
        sum_kernel += kernel_seconds;
    }

    const double avg_h2d = sum_h2d / iters;
    const double avg_d2h = sum_d2h / iters;
    const double avg_kernel = sum_kernel / iters;

    const double avg_mem_time = avg_h2d + avg_d2h;
    const double avg_gpu_total = avg_kernel + avg_mem_time;

    double max_abs_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs((double)h_O[i] - (double)h_ref[i]));
    }

    const double eps = 1e-2;
    const int ok = (max_abs_err <= eps);

    const double avg_kernel_ms = avg_kernel * 1000.0;

    printf("[attention] SEQ_LEN=%d EMBED_DIM=%d\n", SEQ_LEN, embed_dim);
    printf("[attention] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[attention] gpu_hip_time=%.6f s (%.3f ms)\n", avg_kernel, avg_kernel_ms);
    printf("[attention] gpu_mem_h2d_time=%.6f s\n", avg_h2d);
    printf("[attention] gpu_mem_d2h_time=%.6f s\n", avg_d2h);
    printf("[attention] gpu_mem_total_time=%.6f s\n", avg_mem_time);
    printf("[attention] gpu_hip_total_time=%.6f s\n", avg_gpu_total);
    if (avg_kernel > 0.0) {
        printf("[attention] speedup_kernel_only=%.2fx\n", cpu_seconds / avg_kernel);
        printf("[attention] speedup_with_mem=%.2fx\n", cpu_seconds / avg_gpu_total);
    }
    printf("[attention] max_abs_err=%g (eps=%g) | %s\n", max_abs_err, eps, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_S);
    hipFree(d_O);

    std::free(h_Q);
    std::free(h_K);
    std::free(h_V);
    std::free(h_O);
    std::free(h_ref);

    return ok ? 0 : 1;
}
