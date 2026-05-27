# Lesson 4: Collective Operations

Use MPI collective operations to communicate efficiently among all processes.

## Learning Objectives

- Understand `MPI_Bcast`, `MPI_Reduce`, `MPI_Allreduce`
- Measure and compare collective operation times
- Run with exactly 4 processes as specified

## Compilation

```bash
module load spack
module load cmake
module load openmpi5
mkdir -p build && cd build
cmake ..
make collectives-exercise collectives-solution
```

Or compile manually:

```bash
mpicc -O3 -Wall exercise.c -o collectives-exercise
mpicc -O3 -Wall solution.c -o collectives-solution
```

## Parameters

- `--bufsize <N>`: number of integers per operation (default: 1).  
  Larger buffers reveal how collective communication time scales with message size.

## Execution

Run with exactly 4 processes for comparison:

```bash
mpirun -np 4 ./bin/collectives-exercise
mpirun -np 4 ./bin/collectives-solution --bufsize 10000
```

Try with different process counts and buffer sizes:

```bash
mpirun -np 2 ./bin/collectives-solution --bufsize 1000
mpirun -np 8 ./bin/collectives-solution --bufsize 1000
```

### Running on the INF0090 Cluster (CPU partition)

Interactive with `srun`:
```bash
srun --partition=cpu --nodes=1 --ntasks=4 --mpi=pmix ./bin/collectives-exercise
srun --partition=cpu --nodes=1 --ntasks=4 --mpi=pmix ./bin/collectives-solution
```

Via batch script (`job.slurm`):
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:05:00

srun ./bin/collectives-solution
```
Submit: `sbatch job.slurm`

## Expected Output

```
[exercise] rank=0 Bcast time=0.000023s bufsize=1 data=1 | PASS
[exercise] rank=0 Reduce time=0.000034s sum=10 expected=10 bufsize=1 | PASS
[exercise] rank=0 Allreduce time=0.000041s sum=10 expected=10 bufsize=1 | PASS
[exercise] rank=0 total mem=8 bytes | overall: PASS
```

## Hints

- `MPI_Bcast`: one-to-all (root sends, all receive)
- `MPI_Reduce`: all-to-one (all send, root receives)
- `MPI_Allreduce`: all-to-all (all send, all receive)
- Root is typically rank 0
- Each process contributes `rank + 1` as its local value
