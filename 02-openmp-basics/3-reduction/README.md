# 3. Reduction Clause

This lesson introduces the OpenMP `reduction` clause, used to efficiently aggregate results from multiple threads without explicit synchronization.

## Learning Objectives
- Understand how OpenMP handles private copies and aggregation for reductions.
- Learn to safely compute sums, products, max, and min in parallel loops.

## Exercises & Examples
- `examples/reduction_reference.c`: Example of reduction operations.
- `exercises/exercise_reduction.c`: Practice implementing reductions.
- `solutions/exercise_reduction.c`: Reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile your code on the login node:
```bash
gcc -fopenmp examples/reduction_reference.c -o reduction
```

### Running directly with `srun`
You can execute the compiled program directly on a compute node using `srun`:
```bash
# Set the number of threads you want to use
export OMP_NUM_THREADS=4

# Run on the cpu partition
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 ./reduction
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
./reduction
```

Submit the job using:
```bash
sbatch job.slurm
```
You can view the output in the generated `slurm-<job_id>.out` file.
## Questions
- What mathematical properties must an operator have to be used safely in an OpenMP reduction?
- How is memory handled internally by OpenMP for a reduction variable?
