#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int N = 512, K = 512, M = 512;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        if (strcmp(argv[i], "--K") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) M = atoi(argv[++i]);
    }

    // TODO: MPI_Init, MPI_Comm_rank, MPI_Comm_size
    // TODO: check N % size == 0

    int local_rows = N / size;

    double *A_local = (double*)malloc(local_rows * K * sizeof(double));
    double *B       = (double*)malloc(K * M * sizeof(double));
    double *C_local = (double*)calloc(local_rows * M, sizeof(double));

    size_t mem_bytes = local_rows * K * sizeof(double) +
                       K * M * sizeof(double) +
                       local_rows * M * sizeof(double);
    int ok = 1;

    double t_start = MPI_Wtime();

    // TODO: initialize A_local and B with 1.0
    // TODO: MPI_Bcast B from rank 0
    // TODO: compute C_local = A_local * B (OpenMP parallel for)

    double *C = NULL;
    if (rank == 0)
        C = (double*)malloc(N * M * sizeof(double));
    if (rank == 0) mem_bytes += N * M * sizeof(double);

    // TODO: MPI_Gather C_local blocks into C on rank 0

    if (rank == 0) {
        for (int i = 0; i < N && ok; i++)
            for (int j = 0; j < M && ok; j++)
                if (C[i * M + j] != (double)K) ok = 0;
        free(C);
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t_start;

    printf("[exercise] rank=%d matmul N=%d K=%d M=%d time=%fs mem=%zu bytes | %s\n",
           rank, N, K, M, elapsed, mem_bytes, ok ? "PASS" : "FAIL");

    free(A_local);
    free(B);
    free(C_local);

    // TODO: MPI_Finalize
    return ok ? 0 : 1;
}
