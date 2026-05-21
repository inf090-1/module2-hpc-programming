#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello_gpu(void) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU block=%d thread=%d (idx=%d)\n", blockIdx.x, threadIdx.x, idx);
}

int main(void) {
    const int blocks = 2;
    const int threads = 8;

    hello_gpu<<<blocks, threads>>>();
    hipDeviceSynchronize();

    printf("[hello] kernel launched with %d blocks x %d threads | PASS\n", blocks, threads);
    return 0;
}
