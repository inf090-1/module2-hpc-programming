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

    int *data   = (int*)malloc(bufsize * sizeof(int));
    int *result = (int*)malloc(bufsize * sizeof(int));
    for (int i = 0; i < bufsize; i++) data[i] = rank + 1;

    int expected = size * (size + 1) / 2;
    int ok = 1;

    double t_bcast = MPI_Wtime();
    MPI_Bcast(data, bufsize, MPI_INT, 0, MPI_COMM_WORLD);
    t_bcast = MPI_Wtime() - t_bcast;

    int bcast_ok = 1;
    for (int i = 0; i < bufsize; i++)
        if (data[i] != 1) bcast_ok = 0;
    if (!bcast_ok) ok = 0;
    if (rank == 0)
        printf("[solution] rank=%d Bcast time=%fs bufsize=%d data=%d | %s\n",
               rank, t_bcast, bufsize, data[0], bcast_ok ? "PASS" : "FAIL");

    for (int i = 0; i < bufsize; i++) data[i] = rank + 1;

    double t_reduce = MPI_Wtime();
    MPI_Reduce(data, result, bufsize, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    t_reduce = MPI_Wtime() - t_reduce;

    if (rank == 0) {
        int reduce_ok = 1;
        for (int i = 0; i < bufsize; i++)
            if (result[i] != expected) reduce_ok = 0;
        if (!reduce_ok) ok = 0;
        printf("[solution] rank=%d Reduce time=%fs sum=%d expected=%d | %s\n",
               rank, t_reduce, result[0], expected, reduce_ok ? "PASS" : "FAIL");
    }

    double t_allreduce = MPI_Wtime();
    MPI_Allreduce(data, result, bufsize, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    t_allreduce = MPI_Wtime() - t_allreduce;

    int allreduce_ok = 1;
    for (int i = 0; i < bufsize; i++)
        if (result[i] != expected) allreduce_ok = 0;
    if (!allreduce_ok) ok = 0;
    printf("[solution] rank=%d Allreduce time=%fs sum=%d expected=%d bufsize=%d | %s\n",
           rank, t_allreduce, result[0], expected, bufsize, allreduce_ok ? "PASS" : "FAIL");

    free(data);
    free(result);
    MPI_Finalize();
    return ok ? 0 : 1;
}
