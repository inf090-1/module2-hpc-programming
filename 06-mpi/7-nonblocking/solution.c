#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void reapply_bc(double *u, int rank, int N, int local_rows) {
    if (rank == 0)
        for (int j = 0; j < N; j++)
            u[1 * N + j] = 1.0;
}

static int verify(double *u, int rank, int size, int N, int local_rows) {
    double local_sum = 0.0;
    int start = (rank == 0) ? 2 : 1;
    for (int i = start; i <= local_rows; i++)
        for (int j = 1; j < N - 1; j++)
            local_sum += u[i * N + j];
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return (global_sum > 1e-6) ? 1 : 0;
}

static double run_blocking(int rank, int size, int N, int local_rows, int ITER, int up, int down, int *ok) {
    double *u_alloc  = (double*)calloc((local_rows + 2) * N, sizeof(double));
    double *un_alloc = (double*)calloc((local_rows + 2) * N, sizeof(double));
    double *u = u_alloc, *un = un_alloc;

    reapply_bc(u, rank, N, local_rows);

    double t0 = MPI_Wtime();
    for (int iter = 0; iter < ITER; iter++) {
        MPI_Sendrecv(&u[local_rows * N], N, MPI_DOUBLE, down, 0,
                     &u[0],                N, MPI_DOUBLE, up,   0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[1 * N],            N, MPI_DOUBLE, up,   1,
                     &u[(local_rows + 1) * N], N, MPI_DOUBLE, down, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 1; i <= local_rows; i++)
            for (int j = 1; j < N - 1; j++)
                un[i * N + j] = 0.25 * (
                    u[(i-1)*N + j] + u[(i+1)*N + j] +
                    u[i*N + j-1]   + u[i*N + j+1]);

        double *tmp = u; u = un; un = tmp;
        reapply_bc(u, rank, N, local_rows);
    }
    double elapsed = MPI_Wtime() - t0;
    *ok = verify(u, rank, size, N, local_rows);
    free(u_alloc); free(un_alloc);
    return elapsed;
}

static double run_nonblocking(int rank, int size, int N, int local_rows, int ITER, int up, int down, int *ok) {
    double *u_alloc  = (double*)calloc((local_rows + 2) * N, sizeof(double));
    double *un_alloc = (double*)calloc((local_rows + 2) * N, sizeof(double));
    double *u = u_alloc, *un = un_alloc;

    reapply_bc(u, rank, N, local_rows);

    double t0 = MPI_Wtime();
    for (int iter = 0; iter < ITER; iter++) {
        MPI_Request reqs[4];
        MPI_Irecv(&u[0],                N, MPI_DOUBLE, up,   0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&u[(local_rows + 1) * N], N, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&u[1 * N],            N, MPI_DOUBLE, up,   1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(&u[local_rows * N],   N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[3]);

        for (int i = 2; i <= local_rows - 1; i++)
            for (int j = 1; j < N - 1; j++)
                un[i * N + j] = 0.25 * (
                    u[(i-1)*N + j] + u[(i+1)*N + j] +
                    u[i*N + j-1]   + u[i*N + j+1]);

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        for (int j = 1; j < N - 1; j++)
            un[1 * N + j] = 0.25 * (
                u[0*N + j] + u[2*N + j] +
                u[1*N + j-1] + u[1*N + j+1]);

        for (int j = 1; j < N - 1; j++)
            un[local_rows * N + j] = 0.25 * (
                u[(local_rows-1)*N + j] + u[(local_rows+1)*N + j] +
                u[local_rows*N + j-1] + u[local_rows*N + j+1]);

        double *tmp = u; u = un; un = tmp;
        reapply_bc(u, rank, N, local_rows);
    }
    double elapsed = MPI_Wtime() - t0;
    *ok = verify(u, rank, size, N, local_rows);
    free(u_alloc); free(un_alloc);
    return elapsed;
}

int main(int argc, char *argv[]) {
    int N = 256, ITER = 8192;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        if (strcmp(argv[i], "--iter") == 0 && i + 1 < argc) ITER = atoi(argv[++i]);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "N must be divisible by number of processes\n");
        MPI_Finalize();
        return 1;
    }

    int local_rows = N / size;
    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    size_t mem_bytes = 2 * (local_rows + 2) * N * sizeof(double);

    int ok_block, ok_nb;
    double t_block = run_blocking(rank, size, N, local_rows, ITER, up, down, &ok_block);
    double t_nb    = run_nonblocking(rank, size, N, local_rows, ITER, up, down, &ok_nb);

    if (rank == 0) {
        printf("=== Non-blocking MPI: Performance Comparison ===\n");
        printf("N=%d ITER=%d blocking: %.4fs  non-blocking: %.4fs  speedup: %.2fx\n",
               N, ITER, t_block, t_nb, t_block / t_nb);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int ok = ok_block && ok_nb;
    printf("[nonblocking] rank=%d N=%d local_rows=%d time=%.4fs mem=%zu bytes | %s\n",
           rank, N, local_rows, t_nb, mem_bytes, ok ? "PASS" : "FAIL");

    MPI_Finalize();
    return ok ? 0 : 1;
}
