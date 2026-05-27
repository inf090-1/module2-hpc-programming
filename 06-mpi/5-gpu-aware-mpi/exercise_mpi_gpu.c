#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime_api.h>

int main(int argc, char *argv[]) {
    // TODO: MPI_Init, MPI_Comm_rank, MPI_Comm_size

    int count = 1024;

    float *h_data = (float*)malloc(count * sizeof(float));
    size_t mem_bytes = 2 * count * sizeof(float);  // d_send + d_recv
    float expected = (float)(size * (size + 1) / 2);
    int ok = 1;

    double t_start = MPI_Wtime();

    // TODO: hipMalloc d_send, d_recv (count * sizeof(float) each)
    // TODO: hipMemcpy rank+1 into d_send
    // TODO: MPI_Allreduce (GPU-aware — device pointers work directly)
    // TODO: hipMemcpy result back to h_data

    for (int i = 0; i < count; i++) {
        if (h_data[i] != expected) { ok = 0; break; }
    }

    double elapsed = MPI_Wtime() - t_start;

    printf("[exercise] rank=%d MPI-GPU allreduce time=%fs result=%f mem=%zu bytes | %s\n",
           rank, elapsed, h_data[0], mem_bytes, ok ? "PASS" : "FAIL");

    free(h_data);
    // TODO: hipFree d_send, d_recv
    // TODO: MPI_Finalize
    return ok ? 0 : 1;
}
