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

    int next = (rank + 1) % size;
    int prev = (rank + size - 1) % size;

    int *buf = (int*)malloc(bufsize * sizeof(int));
    for (int i = 0; i < bufsize; i++) buf[i] = 0;
    size_t mem_bytes = bufsize * sizeof(int);

    double t0 = MPI_Wtime();

    if (rank == 0) {
        buf[0] = 42;
        // TODO: MPI_Send buf (bufsize ints) to next, MPI_Recv from prev
    } else {
        // TODO: MPI_Recv buf (bufsize ints) from prev, MPI_Send to next
    }

    int ok = (buf[0] == 42);
    double elapsed = MPI_Wtime() - t0;

    printf("[exercise] rank=%d token=%d bufsize=%d ring_time=%fs mem=%zu bytes | %s\n",
           rank, buf[0], bufsize, elapsed, mem_bytes, ok ? "PASS" : "FAIL");

    free(buf);
    // TODO: MPI_Finalize
    return ok ? 0 : 1;
}
