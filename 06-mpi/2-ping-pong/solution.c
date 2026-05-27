#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERATIONS 1000

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

    if (size != 2) {
        if (rank == 0)
            printf("Please run with exactly 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    int *buf = (int*)malloc(bufsize * sizeof(int));
    buf[0] = 1;
    for (int i = 1; i < bufsize; i++) buf[i] = 0;
    int other = (rank == 0) ? 1 : 0;

    double t0 = MPI_Wtime();

    for (int i = 0; i < ITERATIONS; i++) {
        if (rank == 0) {
            MPI_Send(buf, bufsize, MPI_INT, other, 0, MPI_COMM_WORLD);
            MPI_Recv(buf, bufsize, MPI_INT, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf, bufsize, MPI_INT, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            buf[0]++;
            MPI_Send(buf, bufsize, MPI_INT, other, 0, MPI_COMM_WORLD);
        }
    }

    double elapsed = MPI_Wtime() - t0;
    int ok = (buf[0] == 1 + ITERATIONS);

    printf("[solution] rank=%d ping-pong iterations=%d bufsize=%d time=%fs token=%d | %s\n",
           rank, ITERATIONS, bufsize, elapsed, buf[0], ok ? "PASS" : "FAIL");

    free(buf);
    MPI_Finalize();
    return ok ? 0 : 1;
}
