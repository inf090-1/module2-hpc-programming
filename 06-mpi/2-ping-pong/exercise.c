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

    // TODO: MPI_Init, MPI_Comm_rank, MPI_Comm_size
    // TODO: check size == 2

    int *buf = (int*)malloc(bufsize * sizeof(int));
    buf[0] = 1;
    for (int i = 1; i < bufsize; i++) buf[i] = 0;
    int other = (rank == 0) ? 1 : 0;
    size_t mem_bytes = bufsize * sizeof(int);

    double t0 = MPI_Wtime();

    for (int i = 0; i < ITERATIONS; i++) {
        if (rank == 0) {
            // TODO: MPI_Send buf (bufsize ints) to other, then MPI_Recv back
        } else {
            // TODO: MPI_Recv buf from other, increment buf[0], then MPI_Send to other
        }
    }

    int ok = (buf[0] == 1 + ITERATIONS);
    double elapsed = MPI_Wtime() - t0;

    printf("[exercise] rank=%d ping-pong time=%fs bufsize=%d token=%d mem=%zu bytes | %s\n",
           rank, elapsed, bufsize, buf[0], mem_bytes, ok ? "PASS" : "FAIL");

    free(buf);
    // TODO: MPI_Finalize
    return ok ? 0 : 1;
}
