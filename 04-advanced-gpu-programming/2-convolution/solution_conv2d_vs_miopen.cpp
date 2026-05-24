#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

// NCHW layout for batch=1, C_in=1, C_out=1.
// Input:  [H*W]
// Kernel: [3*3] (row-major: ky*3 + kx)
// Output: [H*W]

__global__ void conv2d_naive_3x3(const float* __restrict__ input,
                                 const float* __restrict__ kernel,
                                 float* __restrict__ output,
                                 int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    float acc = 0.0f;

    for (int ky = 0; ky < 3; ++ky) {
        int in_y = y + ky - 1;
        if (in_y < 0 || in_y >= H) continue;

        for (int kx = 0; kx < 3; ++kx) {
            int in_x = x + kx - 1;
            if (in_x < 0 || in_x >= W) continue;

            acc += input[in_y * W + in_x] * kernel[ky * 3 + kx];
        }
    }

    output[y * W + x] = acc;
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

static void miopen_select_solution_1x1_3x3(miopenHandle_t handle,
                                             miopenTensorDescriptor_t inputDesc,
                                             miopenTensorDescriptor_t outputDesc,
                                             miopenTensorDescriptor_t filterDesc,
                                             miopenConvolutionDescriptor_t convDesc,
                                             uint64_t& solution_id,
                                             size_t& workspace_size) {
    size_t solutionCount = 0;
    miopenConvolutionForwardGetSolutionCount(handle,
                                              filterDesc,
                                              inputDesc,
                                              convDesc,
                                              outputDesc,
                                              &solutionCount);

    std::vector<miopenConvSolution_t> solutions;
    solutions.resize(solutionCount);

    size_t returnedCount = 0;
    if (solutionCount > 0) {
        miopenConvolutionForwardGetSolution(handle,
                                             filterDesc,
                                             inputDesc,
                                             convDesc,
                                             outputDesc,
                                             solutionCount,
                                             &returnedCount,
                                             solutions.data());
    }

    if (returnedCount == 0) {
        solution_id = 0;
        workspace_size = 0;
        return;
    }

    solution_id = solutions[0].solution_id;
    workspace_size = solutions[0].workspace_size;
}

int main(int argc, char** argv) {
    int H = 8, W = 8;
    if (argc > 1) H = std::atoi(argv[1]);
    if (argc > 2) W = std::atoi(argv[2]);
    if (H <= 0) H = 8;
    if (W <= 0) W = 8;

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

    // CPU reference (OpenMP)
    double cpu_t0 = omp_get_wtime();
    cpu_conv2d_naive_3x3_omp(h_input, h_kernel, h_out_cpu, H, W);
    double cpu_t1 = omp_get_wtime();
    const double cpu_seconds = cpu_t1 - cpu_t0;

    // Device allocations
    float *d_input = nullptr, *d_out_naive = nullptr, *d_out_miopen = nullptr;
    float* d_kernel = nullptr;

    hipMalloc(&d_input, nInput * sizeof(float));
    hipMalloc(&d_out_naive, nOutput * sizeof(float));
    hipMalloc(&d_out_miopen, nOutput * sizeof(float));
    hipMalloc(&d_kernel, nKernel * sizeof(float));

    // Copy kernel once (not included in timed H2D)
    hipMemcpy(d_kernel, h_kernel, nKernel * sizeof(float), hipMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((W + threads.x - 1) / threads.x, (H + threads.y - 1) / threads.y);

    // --- MIOpen setup (cached) ---
    miopenHandle_t handle;
    miopenCreate(&handle);

    miopenTensorDescriptor_t inputDesc, outputDesc;
    miopenCreateTensorDescriptor(&inputDesc);
    miopenCreateTensorDescriptor(&outputDesc);

    miopenSet4dTensorDescriptor(inputDesc, miopenFloat, 1, 1, H, W);
    miopenSet4dTensorDescriptor(outputDesc, miopenFloat, 1, 1, H, W);

    miopenTensorDescriptor_t filterDesc;
    miopenCreateTensorDescriptor(&filterDesc);
    miopenSet4dTensorDescriptor(filterDesc, miopenFloat, 1, 1, 3, 3);

    miopenConvolutionDescriptor_t convDesc;
    miopenCreateConvolutionDescriptor(&convDesc);
    miopenInitConvolutionDescriptor(convDesc, miopenConvolution,
                                     /*pad_h=*/1, /*pad_w=*/1,
                                     /*stride_h=*/1, /*stride_w=*/1,
                                     /*dilation_h=*/1, /*dilation_w=*/1);

    uint64_t solution_id = 0;
    size_t workSpaceSize = 0;
    miopen_select_solution_1x1_3x3(handle, inputDesc, outputDesc, filterDesc, convDesc,
                                     solution_id, workSpaceSize);

    void* d_workSpace = nullptr;
    if (workSpaceSize > 0) {
        hipMalloc(&d_workSpace, workSpaceSize);
    }

    // Warm-up
    hipMemcpy(d_input, h_input, nInput * sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(conv2d_naive_3x3, blocks, threads, 0, 0,
                       d_input, d_kernel, d_out_naive, H, W);
    hipDeviceSynchronize();

    miopenConvolutionForwardImmediate(handle,
                                       filterDesc, d_kernel,
                                       inputDesc, d_input,
                                       convDesc,
                                       outputDesc, d_out_miopen,
                                       d_workSpace, workSpaceSize,
                                       solution_id);
    hipDeviceSynchronize();

    // --- Timed repetitions ---
    const int warmup_iters = 1;
    const int iters = 5;

    double sum_h2d = 0.0;
    double sum_d2h = 0.0;
    double sum_kernel = 0.0;
    double sum_miopen = 0.0;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEvent_t mi_start, mi_stop;
    hipEventCreate(&mi_start);
    hipEventCreate(&mi_stop);

    for (int rep = 0; rep < warmup_iters + iters; ++rep) {
        // H2D input timing (kernel not copied again)
        const double t_h2d0 = omp_get_wtime();
        hipMemcpy(d_input, h_input, nInput * sizeof(float), hipMemcpyHostToDevice);
        const double t_h2d1 = omp_get_wtime();

        // Kernel timing
        hipEventRecord(start);
        hipLaunchKernelGGL(conv2d_naive_3x3, blocks, threads, 0, 0,
                           d_input, d_kernel, d_out_naive, H, W);
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float kernel_ms = 0.0f;
        hipEventElapsedTime(&kernel_ms, start, stop);

        // D2H output timing
        const double t_d2h0 = omp_get_wtime();
        hipMemcpy(h_out_naive, d_out_naive, nOutput * sizeof(float), hipMemcpyDeviceToHost);
        const double t_d2h1 = omp_get_wtime();

        // MIOpen timing (compute only, no D2H inside timing)
        hipEventRecord(mi_start);
        miopenConvolutionForwardImmediate(handle,
                                           filterDesc, d_kernel,
                                           inputDesc, d_input,
                                           convDesc,
                                           outputDesc, d_out_miopen,
                                           d_workSpace, workSpaceSize,
                                           solution_id);
        hipEventRecord(mi_stop);
        hipEventSynchronize(mi_stop);

        float mi_ms = 0.0f;
        hipEventElapsedTime(&mi_ms, mi_start, mi_stop);

        if (rep >= warmup_iters) {
            sum_h2d += (t_h2d1 - t_h2d0);
            sum_d2h += (t_d2h1 - t_d2h0);
            sum_kernel += (double)kernel_ms / 1000.0;
            sum_miopen += (double)mi_ms / 1000.0;
        }
    }

    const double avg_h2d = sum_h2d / iters;
    const double avg_d2h = sum_d2h / iters;
    const double avg_kernel = sum_kernel / iters;
    const double avg_miopen = sum_miopen / iters;

    const double avg_mem_time = avg_h2d + avg_d2h;
    const double avg_gpu_total_kernel = avg_kernel + avg_mem_time;

    // Copy MIOpen output once for correctness
    hipMemcpy(h_out_miopen, d_out_miopen, nOutput * sizeof(float), hipMemcpyDeviceToHost);

    double max_abs_err = 0.0;
    for (size_t i = 0; i < nOutput; ++i) {
        max_abs_err = std::max(max_abs_err, std::abs((double)h_out_naive[i] - (double)h_out_cpu[i]));
    }
    // also compare miopen output
    double max_abs_err_miopen = 0.0;
    for (size_t i = 0; i < nOutput; ++i) {
        max_abs_err_miopen = std::max(max_abs_err_miopen,
                                      std::abs((double)h_out_naive[i] - (double)h_out_miopen[i]));
    }

    const double eps = 1e-3;
    const int ok = (max_abs_err <= eps) && (max_abs_err_miopen <= eps);

    // Speedups
    const double speedup_kernel_only = (avg_kernel > 0.0) ? (cpu_seconds / avg_kernel) : 0.0;
    const double speedup_with_mem = (avg_gpu_total_kernel > 0.0) ? (cpu_seconds / avg_gpu_total_kernel) : 0.0;
    const double speedup_miopen = (avg_miopen > 0.0) ? (cpu_seconds / avg_miopen) : 0.0;

    printf("[conv2d] H=%d W=%d\n", H, W);
    printf("[conv2d] cpu_openmp_time=%.6f s\n", cpu_seconds);
    printf("[conv2d] gpu_hip_time=%.6f s\n", avg_kernel);
    printf("[conv2d] gpu_mem_h2d_time=%.6f s\n", avg_h2d);
    printf("[conv2d] gpu_mem_d2h_time=%.6f s\n", avg_d2h);
    printf("[conv2d] gpu_mem_total_time=%.6f s\n", avg_mem_time);
    printf("[conv2d] gpu_hip_total_time=%.6f s\n", avg_gpu_total_kernel);
    printf("[conv2d] speedup_kernel_only=%.2fx\n", speedup_kernel_only);
    printf("[conv2d] speedup_with_mem=%.2fx\n", speedup_with_mem);
    printf("[conv2d] miopen_time=%.6f s\n", avg_miopen);
    printf("[conv2d] speedup_miopen=%.2fx\n", speedup_miopen);
    printf("[conv2d] max_abs_err_kernel_vs_cpu=%g (eps=%g), max_abs_err_naive_vs_miopen=%g | %s\n",
           max_abs_err, eps, max_abs_err_miopen, ok ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipEventDestroy(mi_start);
    hipEventDestroy(mi_stop);

    if (d_workSpace) hipFree(d_workSpace);

    hipFree(d_input);
    hipFree(d_out_naive);
    hipFree(d_out_miopen);
    hipFree(d_kernel);

    miopenDestroyConvolutionDescriptor(convDesc);
    miopenDestroyTensorDescriptor(inputDesc);
    miopenDestroyTensorDescriptor(outputDesc);
    miopenDestroyTensorDescriptor(filterDesc);
    miopenDestroy(handle);

    std::free(h_input);
    std::free(h_kernel);
    std::free(h_out_naive);
    std::free(h_out_miopen);
    std::free(h_out_cpu);

    return ok ? 0 : 1;
}
