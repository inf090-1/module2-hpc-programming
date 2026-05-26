#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// TreeReduce: sum a list of matrices using a task-based binary reduction tree.
//
// We store matrices as flat row-major arrays of length (M*N).

#ifndef M
#define M 256
#endif
#ifndef N
#define N 256
#endif
#ifndef NUM_MATS
#define NUM_MATS 64
#endif

static void tree_reduce_sum_range(const float* const* mats,
                                  int left, int right, // [left, right)
                                  float* out,
                                  int size)
{
    // Base case: only one matrix.
    if (right - left == 1) {
        memcpy(out, mats[left], (size_t)size * sizeof(float));
        return;
    }

    // Recursive case: split the range and reduce left/right in parallel tasks.
    int mid = left + (right - left) / 2;

    // Allocate temporaries for partial sums.
    float* tmpL = (float*)calloc((size_t)size, sizeof(float));
    float* tmpR = (float*)calloc((size_t)size, sizeof(float));
    if (!tmpL || !tmpR) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    // TODO: Create a taskgroup and spawn two tasks:
    //   - one task computes tmpL = sum(mats[left..mid))
    //   - one task computes tmpR = sum(mats[mid..right))

    // TODO: Replace the stubs below with the proper task-based recursion.
    // For now (incorrectly sequential), do it sequentially so the code compiles.
    tree_reduce_sum_range(mats, left, mid, tmpL, size);
    tree_reduce_sum_range(mats, mid, right, tmpR, size);

    // Combine the two partial sums (use double to reduce rounding error).
    for (int i = 0; i < size; ++i) {
        out[i] = (float)((double)tmpL[i] + (double)tmpR[i]);
    }

    free(tmpL);
    free(tmpR);
}

static void tree_reduce_sum(const float* const* mats, int num_mats, float* out, int size) {
    tree_reduce_sum_range(mats, 0, num_mats, out, size);
}

static void reference_sum_serial(const float* const* mats, int num_mats, float* out, int size) {
    for (int i = 0; i < size; ++i) {
        double acc = 0.0;
        for (int m = 0; m < num_mats; ++m) {
            acc += (double)mats[m][i];
        }
        out[i] = (float)acc;
    }
}

static void reference_sum_omp_parallel_for(const float* const* mats, int num_mats, float* out, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        double acc = 0.0;
        for (int m = 0; m < num_mats; ++m) {
            acc += (double)mats[m][i];
        }
        out[i] = (float)acc;
    }
}

int main(void) {
    const int size = M * N;

    float* mats[NUM_MATS];
    for (int i = 0; i < NUM_MATS; ++i) {
        mats[i] = (float*)malloc((size_t)size * sizeof(float));
        if (!mats[i]) {
            fprintf(stderr, "Allocation failed\n");
            return 1;
        }
        for (int j = 0; j < size; ++j) {
            mats[i][j] = (float)(0.001f * (i + 1) * (j + 1));
        }
    }

    float* out_ref = (float*)malloc((size_t)size * sizeof(float));
    float* out_omp = (float*)malloc((size_t)size * sizeof(float));
    float* out_tasks = (float*)malloc((size_t)size * sizeof(float));
    if (!out_ref || !out_omp || !out_tasks) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Baseline 1: CPU serial.
    double t_serial0 = omp_get_wtime();
    reference_sum_serial((const float* const*)mats, NUM_MATS, out_ref, size);
    double t_serial1 = omp_get_wtime();
    double t_serial = t_serial1 - t_serial0;

    // Baseline 2: CPU OpenMP parallel-for.
    double t_omp0 = omp_get_wtime();
    reference_sum_omp_parallel_for((const float* const*)mats, NUM_MATS, out_omp, size);
    double t_omp1 = omp_get_wtime();
    double t_omp = t_omp1 - t_omp0;

    // Exercise: task-based tree reduction.
    double t_task0 = omp_get_wtime();
    // TODO: Create the parallel region and call the parallel function.
    double t_task1 = omp_get_wtime();
    double t_task = t_task1 - t_task0;

    // Validate.
    double max_abs_err = 0.0;
    for (int i = 0; i < size; ++i) {
        double err = (double)out_tasks[i] - (double)out_ref[i];
        if (err < 0) err = -err;
        if (err > max_abs_err) max_abs_err = err;
    }

    const double eps = 1e-1;
    const int ok = (max_abs_err <= eps);

    // validate OpenMP baseline too.
    double max_abs_err_omp = 0.0;
    for (int i = 0; i < size; ++i) {
        double err = (double)out_omp[i] - (double)out_ref[i];
        if (err < 0) err = -err;
        if (err > max_abs_err_omp) max_abs_err_omp = err;
    }

    printf("[tree-reduce-matrix] M=%d N=%d NUM_MATS=%d size=%d\n", M, N, NUM_MATS, size);
    printf("[tree-reduce-matrix] cpu_serial_time=%.6f s\n", t_serial);
    printf("[tree-reduce-matrix] cpu_omp_parallel_for_time=%.6f s\n", t_omp);
    printf("[tree-reduce-matrix] exercise_tasks_time=%.6f s\n", t_task);
    printf("[tree-reduce-matrix] max_abs_err_tasks=%g (eps=%g) | %s\n",
           max_abs_err, eps, ok ? "PASS" : "FAIL");
    printf("[tree-reduce-matrix] max_abs_err_omp_baseline=%g\n", max_abs_err_omp);
    if (t_task > 0.0) {
        printf("[tree-reduce-matrix] speedup_serial_vs_tasks=%.2fx\n", t_serial / t_task);
        printf("[tree-reduce-matrix] speedup_omp_vs_tasks=%.2fx\n", t_omp / t_task);
    }

    for (int i = 0; i < NUM_MATS; ++i) free(mats[i]);
    free(out_ref);
    free(out_omp);
    free(out_tasks);

    return ok ? 0 : 1;
}
