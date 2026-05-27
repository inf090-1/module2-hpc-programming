#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime_api.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int count = 1024;
    float *d_send, *d_recv;
    hipMalloc((void**)&d_send, count * sizeof(float));
    hipMalloc((void**)&d_recv, count * sizeof(float));

    float val = (float)(rank + 1);
    for (int i = 0; i < count; i++)
        hipMemcpy(&d_send[i], &val, sizeof(float), hipMemcpyHostToDevice);

    hipMemcpy(d_recv, d_send, count * sizeof(float), hipMemcpyDeviceToDevice);

    MPI_Allreduce(d_send, d_recv, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    float *h_data = (float*)malloc(count * sizeof(float));
    hipMemcpy(h_data, d_recv, count * sizeof(float), hipMemcpyDeviceToHost);

    float expected = (float)(size * (size + 1) / 2);
    int ok = 1;
    for (int i = 0; i < count; i++) {
        if (h_data[i] != expected) {
            ok = 0;
            break;
        }
    }
    printf("[solution] rank=%d MPI-GPU allreduce result=%f | %s\n",
           rank, h_data[0], ok ? "PASS" : "FAIL");

    free(h_data);
    hipFree(d_send);
    hipFree(d_recv);
    MPI_Finalize();
    return ok ? 0 : 1;
}
