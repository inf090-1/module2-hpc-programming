#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello_gpu(void) {
    // TODO: Print a hello line from each GPU thread.
}

int main(void) {
    // TODO: Launch the kernel using 2 blocks and 8 threads per block.
    // TODO: Synchronize the device.

    printf("[hello] kernel launched | PASS\n");
    return 0;
}
