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

    // TODO: MPI_Init, MPI_Comm_rank, MPI_Comm_size

    int *data   = (int*)malloc(bufsize * sizeof(int));
    int *result = (int*)malloc(bufsize * sizeof(int));
    for (int i = 0; i < bufsize; i++) data[i] = rank + 1;

    int expected = size * (size + 1) / 2;
    int ok = 1;
    size_t mem_bytes = 2 * bufsize * sizeof(int);

    double t_bcast = MPI_Wtime();
    // TODO: MPI_Bcast data (bufsize ints) from rank 0
    int bcast_ok = 1;
    for (int i = 0; i < bufsize; i++)
        if (data[i] != 1) bcast_ok = 0;
    if (!bcast_ok) ok = 0;
    t_bcast = MPI_Wtime() - t_bcast;
    printf("[exercise] rank=%d Bcast time=%fs bufsize=%d data=%d | %s\n",
           rank, t_bcast, bufsize, data[0], bcast_ok ? "PASS" : "FAIL");

    for (int i = 0; i < bufsize; i++) data[i] = rank + 1;

    double t_reduce = MPI_Wtime();
    // TODO: MPI_Reduce data (sum) to rank 0
    if (rank == 0) {
        int reduce_ok = 1;
        for (int i = 0; i < bufsize; i++)
            if (result[i] != expected) reduce_ok = 0;
        if (!reduce_ok) ok = 0;
        printf("[exercise] rank=%d Reduce time=%fs sum=%d expected=%d bufsize=%d | %s\n",
               rank, t_reduce, result[0], expected, bufsize, reduce_ok ? "PASS" : "FAIL");
    }
    t_reduce = MPI_Wtime() - t_reduce;

    double t_allreduce = MPI_Wtime();
    // TODO: MPI_Allreduce data (sum)
    int allreduce_ok = 1;
    for (int i = 0; i < bufsize; i++)
        if (result[i] != expected) allreduce_ok = 0;
    if (!allreduce_ok) ok = 0;
    t_allreduce = MPI_Wtime() - t_allreduce;
    mem_bytes += 2 * bufsize * sizeof(int);
    printf("[exercise] rank=%d Allreduce time=%fs sum=%d expected=%d bufsize=%d | %s\n",
           rank, t_allreduce, result[0], expected, bufsize, allreduce_ok ? "PASS" : "FAIL");

    printf("[exercise] rank=%d total mem=%zu bytes | overall: %s\n",
           rank, mem_bytes, ok ? "PASS" : "FAIL");

    free(data);
    free(result);
    // TODO: MPI_Finalize
    return ok ? 0 : 1;
}
