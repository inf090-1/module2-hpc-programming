#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime_api.h>
#include <rccl/rccl.h>

int main(int argc, char *argv[]) {
    // TODO: MPI_Init, MPI_Comm_rank, MPI_Comm_size

    int count = 1024;

    float *h_data = (float*)malloc(count * sizeof(float));
    size_t mem_bytes = count * sizeof(float);
    float expected = (float)(size * (size + 1) / 2);
    int ok = 1;

    double t_start = MPI_Wtime();

    // TODO: hipSetDevice(rank) to bind each rank to a different GPU
    // TODO: rank 0: ncclGetUniqueId, then MPI_Bcast id
    // TODO: ncclCommInitRank
    // TODO: hipMalloc d_data (count * sizeof(float))
    // TODO: hipMemcpy rank+1 into d_data
    // TODO: ncclAllReduce
    // TODO: hipMemcpy result to h_data

    for (int i = 0; i < count; i++) {
        if (h_data[i] != expected) { ok = 0; break; }
    }

    double elapsed = MPI_Wtime() - t_start;

    printf("[exercise] rank=%d RCCL allreduce time=%fs result=%f mem=%zu bytes | %s\n",
           rank, elapsed, h_data[0], mem_bytes, ok ? "PASS" : "FAIL");

    free(h_data);
    // TODO: hipFree d_data, ncclCommDestroy
    // TODO: MPI_Finalize
    return ok ? 0 : 1;
}
