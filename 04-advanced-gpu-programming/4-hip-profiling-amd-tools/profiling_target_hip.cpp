#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

// Simple compute-heavy kernel (vector transform) to generate GPU activity.
__global__ void transform_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = x[idx];
    // Some math so the kernel is not totally trivial.
    y[idx] = alpha * (sinf(v) + cosf(v) + v * v);
}

int main() {
    const int n = 1 << 20; // ~1M floats
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const float alpha = 0.5f;

    float* h_x = (float*)std::malloc(n * sizeof(float));
    float* h_y = (float*)std::malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i) h_x[i] = 0.001f * (float)i;

    float *d_x = nullptr, *d_y = nullptr;
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, n * sizeof(float));

    hipMemcpy(d_x, h_x, n * sizeof(float), hipMemcpyHostToDevice);

    // Warm-up
    hipLaunchKernelGGL(transform_kernel, dim3(blocks), dim3(threads), 0, 0, d_x, d_y, n, alpha);
    hipDeviceSynchronize();

    // Main work: launch multiple times to show activity in the profiler.
    for (int iter = 0; iter < 50; ++iter) {
        hipLaunchKernelGGL(transform_kernel, dim3(blocks), dim3(threads), 0, 0, d_x, d_y, n, alpha);
    }

    hipDeviceSynchronize();

    hipMemcpy(h_y, d_y, n * sizeof(float), hipMemcpyDeviceToHost);

    // Light validation
    double s = 0.0;
    for (int i = 0; i < 10; ++i) s += h_y[i];
    std::printf("[profiling_target] checksum=%f\n", (float)s);

    hipFree(d_x);
    hipFree(d_y);
    std::free(h_x);
    std::free(h_y);

    return 0;
}
