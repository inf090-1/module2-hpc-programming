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

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    int *buf = (int*)malloc(bufsize * sizeof(int));
    for (int i = 0; i < bufsize; i++) buf[i] = 0;

    double t0 = MPI_Wtime();

    if (rank == 0) {
        buf[0] = 42;
        MPI_Send(buf, bufsize, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(buf, bufsize, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(buf, bufsize, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buf, bufsize, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    double elapsed = MPI_Wtime() - t0;
    int ok = (buf[0] == 42);

    printf("[solution] rank=%d token=%d bufsize=%d ring_time=%fs | %s\n",
           rank, buf[0], bufsize, elapsed, ok ? "PASS" : "FAIL");

    free(buf);
    MPI_Finalize();
    return ok ? 0 : 1;
}
