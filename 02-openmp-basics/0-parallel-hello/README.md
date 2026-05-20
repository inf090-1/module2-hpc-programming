# 0. OpenMP Parallel Hello

This intro lesson focuses on the basic OpenMP parallel region (`#pragma omp parallel`).

## Learning Objectives
- Create a simple OpenMP parallel region.
- Print each thread identifier with `omp_get_thread_num()`.
- Observe that multiple threads execute the same block.

## Exercise
- `exercise_parallel_hello.c`: Add an OpenMP parallel region so each thread prints `Hello` and its thread id.
- `solution_parallel_hello.c`: Reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile your code on the login node:
```bash
gcc -O3 -fopenmp exercise_parallel_hello.c -o hello_omp
```

### Running directly with `srun`
You can execute the compiled program directly on a compute node using `srun`:
```bash
# Set the number of threads you want to use
export OMP_NUM_THREADS=4

# Run on the cpu partition
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 ./hello_omp
```

### Running via Batch Script (`sbatch`)
Alternatively, for longer runs, you can submit a batch job. Create a script named `job.slurm`:

```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./hello_omp
```

Submit the job using:
```bash
sbatch job.slurm
```

## Questions
- Does the thread print order stay the same between runs?
