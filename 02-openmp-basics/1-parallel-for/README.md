# 1. Parallel For Loops

This lesson introduces OpenMP parallel regions and worksharing loop constructs (`#pragma omp parallel for`).

## Learning Objectives
- Understand how to create a team of threads.
- Learn how to distribute loop iterations among threads.

## Exercise
- `exercise_parallel_for.c`: Parallelize a data-parallel transform and compare speedup.
- `solution_parallel_for.c`: Reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile your code on the login node:
```bash
gcc -O3 -fopenmp exercise_parallel_for.c -o transform
```

### Running directly with `srun`
You can execute the compiled program directly on a compute node using `srun`:
```bash
# Set the number of threads you want to use
export OMP_NUM_THREADS=4

# Run on the cpu partition
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 ./transform
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

# Set the number of threads to the requested CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the executable
./transform
```

Submit the job using:
```bash
sbatch job.slurm
```
You can view the output in the generated `slurm-<job_id>.out` file.
## Questions
- What happens if the number of loop iterations is not divisible by the number of threads?
- Does OpenMP guarantee the order in which iterations are executed?
