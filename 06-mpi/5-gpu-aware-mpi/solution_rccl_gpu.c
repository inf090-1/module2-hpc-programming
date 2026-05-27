#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int count = 1024;

    hipSetDevice(rank);

    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t rccl_comm;
    ncclCommInitRank(&rccl_comm, size, id, rank);

    float *d_data;
    hipMalloc((void**)&d_data, count * sizeof(float));

    float val = (float)(rank + 1);
    for (int i = 0; i < count; i++)
        hipMemcpy(&d_data[i], &val, sizeof(float), hipMemcpyHostToDevice);

    ncclAllReduce(d_data, d_data, count, ncclFloat, ncclSum, rccl_comm, 0);

    float *h_data = (float*)malloc(count * sizeof(float));
    hipMemcpy(h_data, d_data, count * sizeof(float), hipMemcpyDeviceToHost);

    float expected = (float)(size * (size + 1) / 2);
    int ok = 1;
    for (int i = 0; i < count; i++) {
        if (h_data[i] != expected) {
            ok = 0;
            break;
        }
    }
    printf("[solution] rank=%d RCCL allreduce result=%f | %s\n",
           rank, h_data[0], ok ? "PASS" : "FAIL");

    free(h_data);
    hipFree(d_data);
    ncclCommDestroy(rccl_comm);
    MPI_Finalize();
    return ok ? 0 : 1;
}
