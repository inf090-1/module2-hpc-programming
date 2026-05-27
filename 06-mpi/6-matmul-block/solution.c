#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int N = 512, K = 512, M = 512;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        if (strcmp(argv[i], "--K") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) M = atoi(argv[++i]);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            printf("N must be divisible by number of processes\n");
        MPI_Finalize();
        return 1;
    }

    int local_rows = N / size;

    double *A_local = (double*)malloc(local_rows * K * sizeof(double));
    double *B       = (double*)malloc(K * M * sizeof(double));
    double *C_local = (double*)calloc(local_rows * M, sizeof(double));

    double t_init = MPI_Wtime();

    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < K; j++)
            A_local[i * K + j] = 1.0;

    if (rank == 0)
        for (int i = 0; i < K; i++)
            for (int j = 0; j < M; j++)
                B[i * M + j] = 1.0;

    double t_bcast = MPI_Wtime();
    MPI_Bcast(B, K * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    t_bcast = MPI_Wtime() - t_bcast;

    double t_comp = MPI_Wtime();
    #pragma omp parallel for
    for (int i = 0; i < local_rows; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < M; j++) {
                C_local[i * M + j] += A_local[i * K + k] * B[k * M + j];
            }
        }
    }
    t_comp = MPI_Wtime() - t_comp;

    double *C = NULL;
    if (rank == 0)
        C = (double*)malloc(N * M * sizeof(double));

    double t_gather = MPI_Wtime();
    MPI_Gather(C_local, local_rows * M, MPI_DOUBLE,
               C,       local_rows * M, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    t_gather = MPI_Wtime() - t_gather;

    double total = MPI_Wtime() - t_init;

    int ok = 1;
    if (rank == 0) {
        for (int i = 0; i < N && ok; i++) {
            for (int j = 0; j < M && ok; j++) {
                if (C[i * M + j] != (double)K) {
                    ok = 0;
                    printf("[solution] FAIL at C[%d][%d] = %f (expected %d)\n",
                           i, j, C[i * M + j], K);
                }
            }
        }
        free(C);
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("[solution] rank=%d matmul N=%d K=%d M=%d bcast_time=%fs comp_time=%fs "
           "gather_time=%fs total=%fs | %s\n",
           rank, N, K, M, t_bcast, t_comp, t_gather, total, ok ? "PASS" : "FAIL");

    free(A_local);
    free(B);
    free(C_local);

    MPI_Finalize();
    return ok ? 0 : 1;
}
