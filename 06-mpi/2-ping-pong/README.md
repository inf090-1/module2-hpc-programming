# Lesson 2: Ping Pong

Two processes bounce a value back and forth, measuring the round-trip time.

## Learning Objectives

- Implement alternating send/receive pattern
- Use `MPI_Wtime` for accurate timing
- Understand latency of MPI messages

## Compilation

```bash
module load spack
module load cmake
module load openmpi5
mkdir -p build && cd build
cmake ..
make ping-pong-exercise ping-pong-solution
```

Or compile manually:

```bash
mpicc -O3 -Wall exercise.c -o ping-pong-exercise
mpicc -O3 -Wall solution.c -o ping-pong-solution
```

## Parameters

- `--bufsize <N>`: number of integers per message (default: 1).  
  Larger buffers increase per-message time, letting students measure bandwidth vs latency effects.

## Execution

Run with exactly 2 processes:

```bash
mpirun -np 2 ./build/bin/ping-pong-exercise
mpirun -np 2 ./build/bin/ping-pong-solution --bufsize 10000
```

### Running on the INF0090 Cluster (CPU partition)

Interactive with `srun`:
```bash
srun --partition=cpu --nodes=2 --ntasks=2 --mpi=pmix ./build/bin/ping-pong-exercise
srun --partition=cpu --nodes=2 --ntasks=2 --mpi=pmix ./build/bin/ping-pong-solution
```

Via batch script (`job.slurm`):
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=00:05:00

srun --mpi=pmix ./build/bin/ping-pong-solution
```
Submit: `sbatch job.slurm`

## Expected Output

```
[exercise] rank=0 ping-pong time=0.001234s bufsize=1 token=1001 mem=4 bytes | PASS
[exercise] rank=1 ping-pong time=0.001234s bufsize=1 token=1001 mem=4 bytes | PASS
```

With `--bufsize 100000` (larger messages take longer):
```
[exercise] rank=0 ping-pong time=0.045678s bufsize=100000 token=1001 mem=400000 bytes | PASS
```

## Hints

- Start with rank 0 sending to rank 1
- After receiving, send the value back
- Use `MPI_Wtime()` before and after the loop
- Each full round-trip = 2 messages
