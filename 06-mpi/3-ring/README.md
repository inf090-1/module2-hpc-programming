# Lesson 3: Ring Communication

A token circulates around a ring of processes. Each process sends to the next and receives from the previous.

## Learning Objectives

- Implement cyclic (toroidal) communication patterns
- Use modulo arithmetic for neighbor calculation
- Measure collective communication time

## Compilation

```bash
module load spack
module load cmake
module load openmpi5
mkdir -p build && cd build
cmake ..
make ring-exercise ring-solution
```

Or compile manually:

```bash
mpicc -O3 -Wall exercise.c -o ring-exercise
mpicc -O3 -Wall solution.c -o ring-solution
```

## Parameters

- `--bufsize <N>`: number of integers per message (default: 1).  
  Larger buffers increase message transfer time.

## Execution

```bash
mpirun -np 4 ./build/bin/ring-exercise
mpirun -np 4 ./build/bin/ring-solution --bufsize 10000
```

### Running on the INF0090 Cluster (CPU partition)

Interactive with `srun`:
```bash
srun --partition=cpu --nodes=4 --ntasks=4 --mpi=pmix ./build/bin/ring-exercise
srun --partition=cpu --nodes=4 --ntasks=4 --mpi=pmix ./build/bin/ring-solution
```

Via batch script (`job.slurm`):
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --time=00:05:00

srun --mpi=pmix ./build/bin/ring-solution
```
Submit: `sbatch job.slurm`

## Expected Output

```
[exercise] rank=0 token=42 bufsize=1 ring_time=0.000456s mem=4 bytes | PASS
[exercise] rank=1 token=42 bufsize=1 ring_time=0.000456s mem=4 bytes | PASS
[exercise] rank=2 token=42 bufsize=1 ring_time=0.000456s mem=4 bytes | PASS
[exercise] rank=3 token=42 bufsize=1 ring_time=0.000456s mem=4 bytes | PASS
```

## Hints

- Next rank: `(rank + 1) % size`
- Previous rank: `(rank - 1 + size) % size`
- Use tag to distinguish different messages

## Questions

1. How does ring time scale as you increase the number of processes? Is the relationship linear?
2. What would change if the token traversed the ring in the opposite direction (rank → rank-1)?
3. Why do we need the `+ size` in `(rank - 1 + size) % size`? What happens without it?
