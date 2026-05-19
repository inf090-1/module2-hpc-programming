# 5. Data Dependencies

This lesson covers loop iterations with dependencies, such as loop-carried dependencies, and how they restrict parallelism (DOALL vs DOACROSS).

## Learning Objectives
- Identify independent (DOALL) vs dependent (DOACROSS) loops.
- Learn to manage or restructure dependencies.

## Exercises & Examples
- `examples/doall_doacross_reference.c`: Example showing different dependency types.

## Compilation and Execution on the INFO090 Cluster

First, compile your code on the login node:
```bash
gcc -fopenmp examples/doall_doacross_reference.c -o dependencies
```

### Running directly with `srun`
You can execute the compiled program directly on a compute node using `srun`:
```bash
# Set the number of threads you want to use
export OMP_NUM_THREADS=4

# Run on the cpu partition
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 ./dependencies
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
./dependencies
```

Submit the job using:
```bash
sbatch job.slurm
```
You can view the output in the generated `slurm-<job_id>.out` file.
## Questions
- What makes a loop "DOALL"?
- How can loop-carried dependencies prevent safe parallelization with a basic `#pragma omp parallel for`?
