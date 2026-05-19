# 2. Synchronization Constructs

This lesson covers critical sections, atomics, locks, and barriers in OpenMP. When multiple threads access shared resources, synchronization is needed to prevent data races and ensure correctness.

## Learning Objectives
- Differentiate between `critical` and `atomic` operations.
- Understand how to use OpenMP locks (`omp_lock_t`).
- Learn how to synchronize threads at a specific point using `#pragma omp barrier`.

## Exercises & Examples
- `examples/critical_section_reference.c` / `exercises/exercise_critical_max.c`
- `examples/atomic_counter_reference.c` / `exercises/exercise_atomic_counter.c`
- `examples/lock_counter_reference.c` / `exercises/exercise_lock_counter.c`
- `examples/barrier_phases_reference.c` / `exercises/exercise_barrier_phases.c`

## Compilation and Execution on the INFO090 Cluster

First, compile your code on the login node:
```bash
gcc -fopenmp examples/critical_section_reference.c -o critical_section
```

### Running directly with `srun`
You can execute the compiled program directly on a compute node using `srun`:
```bash
# Set the number of threads you want to use
export OMP_NUM_THREADS=4

# Run on the cpu partition
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 ./critical_section
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
./critical_section
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
