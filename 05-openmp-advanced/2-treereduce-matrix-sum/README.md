# 2. TreeReduce: matrix sum using OpenMP tasks

In this lesson you will implement a **tree-based reduction**.
Instead of summing matrices linearly, you will build a binary reduction tree using **OpenMP tasks**.

## Learning Objectives
- Use tasks to implement a recursive reduction.
- Use `taskgroup` / `taskwait` to enforce correct synchronization.
- Avoid data races by ensuring each task writes only to its own temporary buffer.
- Understand how floating-point accumulation order affects numerical precision (double vs float accumulation).

## Parameters
| Parameter  | Default | Description                         |
|------------|---------|-------------------------------------|
| `M`        | 256     | Matrix rows                         |
| `N`        | 256     | Matrix columns                      |
| `NUM_MATS` | 64      | Number of matrices to sum           |

Override at compile time, e.g. `gcc -DM=128 -DN=128 -DNUM_MATS=32 ...`.

## Exercise
Implement the TODOs in:
- `exercise_tree_reduce.c`

The program builds `NUM_MATS` input matrices (flat row-major arrays), sums them with a task-based binary tree reduction, and validates against a sequential reference that uses double-precision accumulation.

**Tolerance**: The validation uses `eps = 0.1` because the task-based tree sums matrices in a different order than the serial reference, leading to small floating-point differences.

## Compilation and Execution on the INF0090 Cluster

### Using the root Makefile (recommended)
```bash
cd 05-openmp-advanced
make 2-treereduce-matrix-sum/solution_tree_reduce
```

### Manual compilation
```bash
gcc -O3 -fopenmp exercise_tree_reduce.c -o exercise_tree_reduce
```

### Running directly with `srun`
```bash
cd 2-treereduce-matrix-sum
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 \
  bash -c 'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}; ./exercise_tree_reduce'
```

### Running via Batch Script (`sbatch`)
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

cd 2-treereduce-matrix-sum
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
./exercise_tree_reduce
```
Submit:
```bash
sbatch job.slurm
```

## Questions
- Why do we accumulate with `double` inside each reduction node? What happens with pure `float`?
- Which part dominates when `NUM_MATS` grows large: task overhead, memory bandwidth, or arithmetic work?
- How does the error tolerance of 0.1 relate to the number of matrices and the input values?
