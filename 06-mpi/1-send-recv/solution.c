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

    if (size != 2) {
        if (rank == 0)
            printf("Please run with exactly 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    int *data = (int*)malloc(bufsize * sizeof(int));
    for (int i = 0; i < bufsize; i++) data[i] = 42;
    int ok = 1;

    if (rank == 0) {
        MPI_Send(data, bufsize, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("[solution] rank=%d sent bufsize=%d value=%d\n", rank, bufsize, data[0]);
    } else if (rank == 1) {
        MPI_Recv(data, bufsize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < bufsize; i++)
            if (data[i] != 42) ok = 0;
        printf("[solution] rank=%d received bufsize=%d value=%d | %s\n",
               rank, bufsize, data[0], ok ? "PASS" : "FAIL");
    }

    free(data);
    MPI_Finalize();
    return ok ? 0 : 1;
}
