# 2. Synchronization Constructs

This lesson covers critical sections, atomics, locks, and barriers in OpenMP. When multiple threads access shared resources, synchronization is needed to prevent data races and ensure correctness.

## Learning Objectives
- Differentiate between `critical` and `atomic` operations.
- Understand how to use OpenMP locks (`omp_lock_t`).
- Learn how to synchronize threads at a specific point using `#pragma omp barrier`.

## Exercise
- `exercise_synchronization.c`: Build a histogram with `critical` and compare speedup. You can also explore `atomic` or `omp_lock_t` to protect updates.
- `solution_synchronization.c`: Reference solution.

### Sync Alternatives
- `atomic`: best for a single shared increment; lowest overhead when contention is moderate.
- `omp_lock_t`: useful for more complex critical regions or when you want to lock per-bin.
- `critical`: simplest correctness baseline; can be slower under heavy contention.

## Compilation and Execution on the INF0090 Cluster

First, compile your code on the login node:
```bash
gcc -O3 -fopenmp exercise_synchronization.c -o synchronization
```

### Running directly with `srun`
You can execute the compiled program directly on a compute node using `srun`:
```bash
# Set the number of threads you want to use
export OMP_NUM_THREADS=4

# Run on the cpu partition
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 ./synchronization
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
./synchronization
```

Submit the job using:
```bash
sbatch job.slurm
```
You can view the output in the generated `slurm-<job_id>.out` file.
## Questions
- Why are `atomic` operations generally preferred over `critical` sections when applicable?
- In what scenarios would you need explicit locks rather than compiler directives?
- What would happen in the barrier exercise if some threads never hit the barrier?
