# Lesson 1: Send and Receive

Learn the most basic MPI communication pattern: passing a single piece of data between two processes.

## Learning Objectives

- Understand MPI rank and communicator concepts
- Use `MPI_Send` and `MPI_Recv` for point-to-point communication
- Use `MPI_Status` to inspect received messages

## Compilation

```bash
module load spack
module load cmake
module load openmpi5
mkdir -p build && cd build
cmake ..
make send-recv-exercise send-recv-solution
```

Or compile manually:

```bash
mpicc -O3 -Wall exercise.c -o send-recv-exercise
mpicc -O3 -Wall solution.c -o send-recv-solution
```

## Parameters

- `--bufsize <N>`: number of integers to send (default: 1).  
  Increase to see how message size affects latency.

## Execution

Run with exactly 2 processes:

```bash
mpirun -np 2 ./build/bin/send-recv-exercise
mpirun -np 2 ./build/bin/send-recv-solution --bufsize 10000
```

### Running on the INF0090 Cluster (CPU partition)

Interactive with `srun`:
```bash
srun --partition=cpu --nodes=2 --ntasks=2 --mpi=pmix ./build/bin/send-recv-exercise
srun --partition=cpu --nodes=2 --ntasks=2 --mpi=pmix ./build/bin/send-recv-solution
```

Via batch script (`job.slurm`):
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:05:00

srun --mpi=pmix ./build/bin/send-recv-exercise
```
Submit: `sbatch job.slurm`

## Expected Output

```
[exercise] rank=0 send-recv time=0.000123s bufsize=1 mem=4 bytes | PASS
[exercise] rank=1 send-recv time=0.000123s bufsize=1 mem=4 bytes | PASS
```

With `--bufsize 100000`:
```
[exercise] rank=0 send-recv time=0.000456s bufsize=100000 mem=400000 bytes | PASS
```

## Hints

- Rank 0 sends, rank 1 receives
- Use `MPI_Send(&data, 1, MPI_INT, dest, tag, MPI_COMM_WORLD)`
- Use `MPI_Recv(&data, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status)`
