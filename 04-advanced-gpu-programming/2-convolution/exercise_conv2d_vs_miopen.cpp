#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// NCHW layout for batch=1, C_in=1, C_out=1.
// Input:  [H*W]
// Kernel: [3*3]
// Output: [H*W]

__global__ void conv2d_naive_3x3(const float* __restrict__ input,
                                 const float* __restrict__ kernel,
                                 float* __restrict__ output,
                                 int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    // TODO: Compute pad=1 3x3 convolution for output[y, x].
}

static void cpu_conv2d_naive_3x3_omp(const float* input, const float* kernel,
                                       float* output, int H, int W) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int ky = 0; ky < 3; ++ky) {
                const int in_y = y + ky - 1;
                if (in_y < 0 || in_y >= H) continue;
                for (int kx = 0; kx < 3; ++kx) {
                    const int in_x = x + kx - 1;
                    if (in_x < 0 || in_x >= W) continue;
                    acc += input[in_y * W + in_x] * kernel[ky * 3 + kx];
                }
            }
            output[y * W + x] = acc;
        }
    }
}

static int miopen_convolution_1x1_3x3(const float* d_input, float* d_output,
                                       const float* h_kernel,
                                       int H, int W) {
    const int N = 1;
    const int C = 1;
    const int K = 3;

    miopenHandle_t handle;
    miopenCreate(&handle);

    miopenTensorDescriptor_t inputDesc, outputDesc;
    miopenCreateTensorDescriptor(&inputDesc);
    miopenCreateTensorDescriptor(&outputDesc);

    miopenSet4dTensorDescriptor(inputDesc, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(outputDesc, miopenFloat, N, C, H, W);

    miopenTensorDescriptor_t filterDesc;
    miopenCreateTensorDescriptor(&filterDesc);
    miopenSet4dTensorDescriptor(filterDesc, miopenFloat, C, C, K, K);

    miopenConvolutionDescriptor_t convDesc;
    miopenCreateConvolutionDescriptor(&convDesc);

    miopenInitConvolutionDescriptor(convDesc, miopenConvolution,
                                    /*pad_h=*/1, /*pad_w=*/1,
                                    /*stride_h=*/1, /*stride_w=*/1,
                                    /*dilation_h=*/1, /*dilation_w=*/1);

    float* d_filter = nullptr;
    hipMalloc(&d_filter, (size_t)C * C * K * K * sizeof(float));
    hipMemcpy(d_filter, h_kernel, (size_t)C * C * K * K * sizeof(float), hipMemcpyHostToDevice);

    size_t solutionCount = 0;
    miopenConvolutionForwardGetSolutionCount(handle,
                                              filterDesc,
                                              inputDesc,
                                              convDesc,
                                              outputDesc,
                                              &solutionCount);

    miopenConvSolution_t* solutions = nullptr;
    if (solutionCount > 0) {
        solutions = (miopenConvSolution_t*)std::malloc(solutionCount * sizeof(miopenConvSolution_t));
    }

    size_t returnedCount = 0;
    if (solutions) {
        miopenConvolutionForwardGetSolution(handle,
                                             filterDesc,
                                             inputDesc,
                                             convDesc,
                                             outputDesc,
                                             solutionCount,
                                             &returnedCount,
                                             solutions);
    }

    uint64_t solution_id = (solutions && returnedCount > 0) ? solutions[0].solution_id : 0;
    size_t workSpaceSize = (solutions && returnedCount > 0) ? solutions[0].workspace_size : 0;

    void* d_workSpace = nullptr;
    if (workSpaceSize > 0) {
        hipMalloc(&d_workSpace, workSpaceSize);
    }

    miopenStatus_t status = miopenConvolutionForwardImmediate(
        handle,
        filterDesc, d_filter,
        inputDesc, d_input,
        convDesc,
        outputDesc, d_output,
        d_workSpace, workSpaceSize,
        solution_id);

    if (d_workSpace) hipFree(d_workSpace);
    if (solutions) std::free(solutions);
    hipFree(d_filter);

    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroyTensorDescriptor(inputDesc);
    miopenDestroyTensorDescriptor(outputDesc);
    miopenDestroyTensorDescriptor(filterDesc);
    miopenDestroy(handle);

    return (status == miopenStatusSuccess) ? 0 : 1;
}

int main(void) {
    const int H = 8, W = 8;

    const size_t nInput = (size_t)H * W;
    const size_t nKernel = 9;
    const size_t nOutput = (size_t)H * W;

    float* h_input = (float*)std::malloc(nInput * sizeof(float));
    float* h_kernel = (float*)std::malloc(nKernel * sizeof(float));
    float* h_out_naive = (float*)std::malloc(nOutput * sizeof(float));
    float* h_out_miopen = (float*)std::malloc(nOutput * sizeof(float));
    float* h_out_cpu = (float*)std::malloc(nOutput * sizeof(float));

    if (!h_input || !h_kernel || !h_out_naive || !h_out_miopen || !h_out_cpu) {
        std::fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < nInput; ++i) h_input[i] = sinf(0.01f * (float)i) * 0.5f + 1.0f;
    for (size_t i = 0; i < nKernel; ++i) h_kernel[i] = 0.1f * (float)(i + 1);

    double cpu_t0 = omp_get_wtime();
    cpu_conv2d_naive_3x3_omp(h_input, h_kernel, h_out_cpu, H, W);
    double cpu_t1 = omp_get_wtime();
    double cpu_seconds = cpu_t1 - cpu_t0;

    float *d_input = nullptr, *d_out_naive = nullptr, *d_out_miopen = nullptr;
    // TODO: Allocate device memory for d_input, d_out_naive, and d_out_miopen.

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    float* d_kernel = nullptr;
    // TODO: Allocate device memory for d_kernel.

    double gpu_mem_t0 = omp_get_wtime();
    // TODO: Copy d_input <- h_input and d_kernel <- h_kernel.
    double gpu_mem_t1 = omp_get_wtime();

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    // TODO: Launch conv2d_naive_3x3 kernel.
    // hipLaunchKernelGGL(conv2d_naive_3x3, blocks, threads, 0, 0, d_input, d_kernel, d_out_naive, H, W);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float gpu_ms = 0.0f;
    hipEventElapsedTime(&gpu_ms, start, stop);
    double gpu_seconds = (double)gpu_ms / 1000.0;

    double gpu_mem_back_t0 = omp_get_wtime();
    // TODO: Copy h_out_naive <- d_out_naive.
    double gpu_mem_back_t1 = omp_get_wtime();

    const double gpu_h2d_time = (gpu_mem_t1 - gpu_mem_t0);
    const double gpu_d2h_time = (gpu_mem_back_t1 - gpu_mem_back_t0);
    double gpu_mem_time = gpu_h2d_time + gpu_d2h_time;

    // Run MIOpen for correctness checking (timed separately).
    float miopen_ms = 0.0f;
    hipEvent_t miopen_start, miopen_stop;
    hipEventCreate(&miopen_start);
    hipEventCreate(&miopen_stop);

    hipEventRecord(miopen_start);
    int miopen_ok = miopen_convolution_1x1_3x3(d_input, d_out_miopen, h_kernel, H, W);
    hipEventRecord(miopen_stop);
    hipEventSynchronize(miopen_stop);

    hipEventElapsedTime(&miopen_ms, miopen_start, miopen_stop);
    const double miopen_seconds = (double)miopen_ms / 1000.0;

    hipEventDestroy(miopen_start);
    hipEventDestroy(miopen_stop);

    // TODO: Copy h_out_miopen <- d_out_miopen.

    double max_abs_err = 0.0;
    for (size_t i = 0; i < nOutput; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs((double)h_out_naive[i] - (double)h_out_miopen[i]));
    }

    const double eps = 1e-3;
    const int ok = (miopen_ok == 0) && (max_abs_err <= eps);

    printf("[conv2d] H=%d W=%d\n", H, W);
    printf("[conv2d] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[conv2d] gpu_hip_time=%.6f s (%.3f ms)\n", gpu_seconds, gpu_ms);
    printf("[conv2d] gpu_mem_h2d_time=%.6f s\n", gpu_h2d_time);
    printf("[conv2d] gpu_mem_d2h_time=%.6f s\n", gpu_d2h_time);
    printf("[conv2d] gpu_mem_total_time=%.6f s\n", gpu_mem_time);
    printf("[conv2d] gpu_hip_total_time=%.6f s\n", gpu_seconds + gpu_mem_time);
    if (gpu_seconds > 0.0) {
        printf("[conv2d] speedup_kernel_only=%.2fx\n", cpu_seconds / gpu_seconds);
        printf("[conv2d] speedup_with_mem=%.2fx\n", cpu_seconds / (gpu_seconds + gpu_mem_time));
    }
    printf("[conv2d] miopen_time=%.6f s (%.3f ms)\n", miopen_seconds, miopen_ms);
    if (miopen_seconds > 0.0) {
        printf("[conv2d] speedup_miopen=%.2fx\n", cpu_seconds / miopen_seconds);
    }
    printf("[conv2d] miopen_status=%d max_abs_err=%g (eps=%g) | %s\n",
           miopen_ok, max_abs_err, eps, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);

    // TODO: Free device memory.
    std::free(h_input);
    std::free(h_kernel);
    std::free(h_out_naive);
    std::free(h_out_miopen);
    std::free(h_out_cpu);

    return ok ? 0 : 1;
}
