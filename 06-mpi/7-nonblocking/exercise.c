#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int N = 896, ITER = 2340;
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
    int up   = (rank == 0)          ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1)   ? MPI_PROC_NULL : rank + 1;

    // Grid layout: (local_rows + 2) rows x N columns
    //   row 0: ghost from above neighbor
    //   rows 1 .. local_rows: owned data
    //   row local_rows+1: ghost from below neighbor
    double *u     = (double*)calloc((local_rows + 2) * N, sizeof(double));
    double *u_new = (double*)calloc((local_rows + 2) * N, sizeof(double));

    size_t mem_bytes = 2 * (local_rows + 2) * N * sizeof(double);

    // Set top boundary condition
    if (rank == 0)
        for (int j = 0; j < N; j++)
            u[1 * N + j] = 1.0;

    // ============================================================
    // BLOCKING VERSION (baseline)
    //
    // At each iteration:
    //   1. Exchange halo rows with MPI_Sendrecv
    //   2. Compute stencil on all owned rows
    //
    // TODO: Convert this loop to use non-blocking MPI:
    //   1. Replace MPI_Sendrecv with MPI_Isend / MPI_Irecv
    //   2. Compute interior rows (2 .. local_rows-1) BEFORE Waitall
    //   3. MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE)
    //   4. Compute boundary rows (1 and local_rows) AFTER Waitall
    // ============================================================
    double t_start = MPI_Wtime();

    for (int iter = 0; iter < ITER; iter++) {
        // TODO (step 1): post non-blocking receives and sends
        MPI_Sendrecv(&u[local_rows * N], N, MPI_DOUBLE, down, 0,
                     &u[0],                N, MPI_DOUBLE, up,   0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[1 * N],            N, MPI_DOUBLE, up,   1,
                     &u[(local_rows + 1) * N], N, MPI_DOUBLE, down, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // TODO (step 2): compute interior rows HERE (before Waitall)
        // TODO (step 3): MPI_Waitall(...)

        // Full stencil on all owned rows (blocking version)
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j < N - 1; j++) {
                u_new[i * N + j] = 0.25 * (
                    u[(i-1) * N + j] + u[(i+1) * N + j] +
                    u[i * N + (j-1)] + u[i * N + (j+1)]);
            }
        }

        // TODO (step 4): compute boundary rows AFTER Waitall

        double *tmp = u; u = u_new; u_new = tmp;

        // Re-apply top boundary condition (Dirichlet)
        if (rank == 0)
            for (int j = 0; j < N; j++)
                u[1 * N + j] = 1.0;
    }

    // Verify heat propagated: sum of all cells below top boundary
    double local_sum = 0.0;
    int start = (rank == 0) ? 2 : 1;  // skip top boundary on rank 0
    for (int i = start; i <= local_rows; i++)
        for (int j = 1; j < N - 1; j++)
            local_sum += u[i * N + j];
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int ok = (global_sum > 1e-6) ? 1 : 0;

    double elapsed = MPI_Wtime() - t_start;

    printf("[exercise] rank=%d N=%d local_rows=%d time=%.4fs mem=%zu bytes iters=%d | %s\n",
           rank, N, local_rows, elapsed, mem_bytes, ITER, ok ? "PASS" : "FAIL");

    free(u);
    free(u_new);

    MPI_Finalize();
    return ok ? 0 : 1;
}
