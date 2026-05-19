# 4. Scheduling Strategies

This lesson explores how OpenMP schedules loop iterations among threads using different clauses (`static`, `dynamic`, `guided`).

## Learning Objectives
- Learn the difference between static, dynamic, and guided scheduling.
- Understand how chunk sizes affect load balancing and overhead.

## Exercises & Examples
- `examples/scheduling_reference.c`: Example showing different scheduling clauses.
- `exercises/exercise_schedule_irregular.c`: Practice balancing irregular workloads.
- `solutions/exercise_schedule_irregular.c`: Reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile your code on the login node:
```bash
gcc -fopenmp examples/scheduling_reference.c -o scheduling
```

### Running directly with `srun`
You can execute the compiled program directly on a compute node using `srun`:
```bash
# Set the number of threads you want to use
export OMP_NUM_THREADS=4

# Run on the cpu partition
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 ./scheduling
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
./scheduling
```

Submit the job using:
```bash
sbatch job.slurm
```
You can view the output in the generated `slurm-<job_id>.out` file.
## Questions
- If a loop has iterations that take vastly different amounts of time, which schedule type is usually best? Why?
- What is the overhead trade-off when using a small chunk size in dynamic scheduling?
