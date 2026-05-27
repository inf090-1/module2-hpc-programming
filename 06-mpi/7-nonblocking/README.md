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

- `--N <n>`: grid rows per dimension (default: 256, must be divisible by `nprocs`)
- `--iter <n>`: number of Jacobi iterations (default: 8192)

Change these to see how problem size affects the blocking vs non-blocking speedup.  
Smaller grids have a higher communication-to-compute ratio, making overlap more beneficial — especially across nodes where network latency amplifies the gain.  
Larger grids have more compute per halo exchange, reducing the fraction of time spent in communication.  
The default N=256 gives the best observed multi-node speedup (~2.14x on 4 nodes).

## Execution

```bash
# Compare blocking (exercise) vs non-blocking (solution) with default N=256
mpirun -np 2 ./build/bin/nonblocking-exercise
mpirun -np 2 ./build/bin/nonblocking-solution

# Try different grid sizes (4 processes across 4 nodes)
mpirun -np 4 ./build/bin/nonblocking-solution --N 896 --iter 2340
mpirun -np 4 ./build/bin/nonblocking-solution --N 128 --iter 16384
```

### Running on the INF0090 Cluster (CPU partition)

Interactive with `srun`:
```bash
srun --partition=cpu --nodes=4 --ntasks=4 --cpus-per-task=1 --mpi=pmix \
  ./build/bin/nonblocking-solution
```

Via batch script (`job.slurm`):
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

srun --mpi=pmix ./build/bin/nonblocking-solution
```
Submit: `sbatch job.slurm`

## Expected Output

```
[exercise] rank=0 N=256 local_rows=64 time=2.0000s mem=270336 bytes iters=8192 | PASS
[exercise] rank=1 N=256 local_rows=64 time=2.0000s mem=270336 bytes iters=8192 | PASS
```

The solution prints a timing comparison (exact times vary by system):

```
=== Non-blocking MPI: Performance Comparison ===
N=256 ITER=8192 blocking: 4.5599s  non-blocking: 2.1353s  speedup: 2.14x
[nonblocking] rank=0 N=256 local_rows=64 time=2.1353s mem=270336 bytes | PASS
```

## Benchmark Results (4 nodes, 4 tasks on info090 CPU partition)

| N | ITER | Blocking | Non-blocking | Speedup |
|---|------|----------|--------------|---------|
| **128** | 16384 | 6.4134s | 3.3574s | 1.91x |
| **256** | 8192 | 4.5599s | 2.1353s | **2.14x** |
| 384 | 5461 | 4.6354s | 2.6704s | 1.74x |
| 512 | 4096 | 5.5096s | 3.1067s | 1.77x |
| 640 | 3276 | 5.4223s | 3.5240s | 1.54x |
| 704 | 2978 | 5.3663s | 3.8772s | 1.38x |
| 768 | 2730 | 6.1452s | 4.7822s | 1.29x |
| 896 | 2340 | 6.5778s | 4.8602s | 1.35x |
| 1024 | 2048 | 7.7160s | 6.2627s | 1.23x |
| 1280 | 1638 | 9.1885s | 7.8311s | 1.17x |
| 1536 | 1365 | 10.2779s | 9.3573s | 1.10x |
| 2048 | 1024 | 13.4012s | 11.2438s | 1.19x |

N=256 gives the best multi-node speedup (2.14x) because:
- Smaller grids have a higher communication-to-compute ratio
- Inter-node communication (network) has higher latency than shared memory, so overlap benefits are magnified
- At larger N, compute dominates and the relative improvement from overlap shrinks

## Hints

- Post receives (`MPI_Irecv`) **before** sends (`MPI_Isend`) to avoid buffer deadlocks
- Interior rows (2 to `local_rows - 1`) don't need halo data — compute these after posting but before `MPI_Waitall`
- Boundary rows (1 and `local_rows`) need halo data — compute these after `MPI_Waitall`
- Use `MPI_PROC_NULL` for ranks outside `[0, size-1]` — sends/recvs with `MPI_PROC_NULL` complete immediately with no effect

## Questions

1. Why does non-blocking communication achieve a higher speedup on multi-node (2.14x at N=256) than on single-node (1.27x at N=896)? What factor changes between these two scenarios?
2. What would happen if you posted `MPI_Isend` before `MPI_Irecv`? Would the program still produce correct results?
3. How would the speedup change if you increased the number of processes per node while keeping the total process count constant (e.g., 2 nodes × 2 processes instead of 4 nodes × 1 process)?
