#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int bufsize = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bufsize") == 0 && i + 1 < argc)
            bufsize = atoi(argv[++i]);
    }

    // TODO: MPI_Init
    // TODO: MPI_Comm_rank / MPI_Comm_size
    // TODO: check size == 2

    int *data = (int*)malloc(bufsize * sizeof(int));
    for (int i = 0; i < bufsize; i++) data[i] = 42;
    int ok = 1;
    size_t mem_bytes = bufsize * sizeof(int);

    double t_start = MPI_Wtime();

    if (rank == 0) {
        // TODO: MPI_Send data (bufsize ints) to rank 1
    } else {
        // TODO: MPI_Recv data (bufsize ints) from rank 0
        for (int i = 0; i < bufsize; i++)
            if (data[i] != 42) ok = 0;
    }

    double elapsed = MPI_Wtime() - t_start;

    printf("[exercise] rank=%d send-recv time=%fs bufsize=%d mem=%zu bytes | %s\n",
           rank, elapsed, bufsize, mem_bytes, ok ? "PASS" : "FAIL");

    free(data);
    // TODO: MPI_Finalize
    return ok ? 0 : 1;
}
