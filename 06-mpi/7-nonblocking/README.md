# Lesson 7: Non-blocking MPI Communication

Non-blocking communication with `MPI_Isend`/`MPI_Irecv` enables overlapping communication with computation. This lesson uses a 2D Jacobi stencil (heat equation) that exchanges halo rows at each iteration — a natural fit for overlap.

The **exercise** provides a working blocking-MPI stencil; your task is to convert the halo exchange to non-blocking and split the computation so interior cells are computed while halo data is in flight.

## Learning Objectives

- Use `MPI_Isend`, `MPI_Irecv`, and `MPI_Waitall` for non-blocking halo exchange
- Overlap communication with computation by computing interior cells during halo transfer
- Compare performance of blocking vs non-blocking MPI

## Compilation

```bash
module load spack
module load cmake
module load openmpi5
mkdir -p build && cd build
cmake ..
make nonblocking-exercise nonblocking-solution
```

Or compile manually:

```bash
mpicc -O3 -Wall exercise.c -o nonblocking-exercise
mpicc -O3 -Wall solution.c -o nonblocking-solution
```

## Parameters

- `--N <n>`: grid rows per dimension (default: 896, must be divisible by `nprocs`)
- `--iter <n>`: number of Jacobi iterations (default: 2340)

Change these to see how problem size affects the blocking vs non-blocking speedup.  
Smaller grids (e.g., N=256) have less compute per halo exchange, reducing the overlap benefit.  
Larger grids (e.g., N=1024) have more compute per halo exchange, but communication is a smaller fraction of total time.  
The default N=896 gives the best observed speedup (~1.27x on 4 ranks).

## Execution

```bash
# Compare blocking (exercise) vs non-blocking (solution) with default N=512
mpirun -np 2 ./bin/nonblocking-exercise
mpirun -np 2 ./bin/nonblocking-solution

# Try different grid sizes
mpirun -np 4 ./bin/nonblocking-solution --N 1024 --iter 2048
mpirun -np 4 ./bin/nonblocking-solution --N 256 --iter 8192

# Try with more processes
mpirun -np 4 ./bin/nonblocking-solution
```

### Running on the INF0090 Cluster (CPU partition)

Interactive with `srun`:
```bash
srun --partition=cpu --nodes=1 --ntasks=4 --cpus-per-task=1 --mpi=pmix \
  ./bin/nonblocking-solution
```

Via batch script (`job.slurm`):
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

srun ./bin/nonblocking-solution
```
Submit: `sbatch job.slurm`

## Expected Output

```
[exercise] rank=0 N=896 local_rows=224 time=1.2000s mem=3239936 bytes iters=2340 | PASS
[exercise] rank=1 N=896 local_rows=224 time=1.2000s mem=3239936 bytes iters=2340 | PASS
```

The solution prints a timing comparison (exact times vary by system):

```
=== Non-blocking MPI: Performance Comparison ===
N=896 ITER=2340 blocking: 1.2131s  non-blocking: 0.9646s  speedup: 1.26x
[nonblocking] rank=0 N=896 local_rows=224 time=0.9646s mem=3239936 bytes | PASS
```

## Benchmark Results (4 ranks on info090 CPU partition)

| N | ITER | Blocking | Non-blocking | Speedup | Overlap* |
|---|------|----------|--------------|---------|----------|
| 128 | 16384 | 0.1412s | 0.1319s | 1.07x | 93.8% |
| 256 | 8192 | 0.2155s | 0.2070s | 1.04x | 96.9% |
| 384 | 5461 | 0.2988s | 0.2801s | 1.07x | 97.9% |
| 512 | 4096 | 0.5218s | 0.4521s | 1.15x | 98.4% |
| 640 | 3276 | 0.5932s | 0.5354s | 1.11x | 98.8% |
| 704 | 2978 | 0.7047s | 0.5894s | 1.20x | 98.9% |
| **768** | 2730 | 0.9058s | 0.7486s | **1.21x** | 99.0% |
| **896** | 2340 | 1.2336s | 0.9692s | **1.27x** | 99.1% |
| 1024 | 2048 | 1.7422s | 1.5552s | 1.12x | 99.2% |
| 1280 | 1638 | 2.6769s | 2.2914s | 1.17x | 99.4% |
| 1536 | 1365 | 3.7754s | 4.0748s | 0.93x | 99.5% |
| 2048 | 1024 | 5.8384s | 5.7937s | 1.01x | 99.6% |

\*Overlap = fraction of owned rows that don't need halo data: `(local_rows - 2) / local_rows`.  
N=896 gives the best speedup (1.27x average, up to 1.32x) because:
- Interior rows (can be computed during halo exchange) = 222 out of 224 owned rows
- At smaller N the communication-to-compute ratio is less favorable
- At larger N the total runtime increases but communication is a smaller fraction

## Hints

- Post receives (`MPI_Irecv`) **before** sends (`MPI_Isend`) to avoid buffer deadlocks
- Interior rows (2 to `local_rows - 1`) don't need halo data — compute these after posting but before `MPI_Waitall`
- Boundary rows (1 and `local_rows`) need halo data — compute these after `MPI_Waitall`
- Use `MPI_PROC_NULL` for ranks outside `[0, size-1]` — sends/recvs with `MPI_PROC_NULL` complete immediately with no effect
