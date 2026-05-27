# Lesson 6: Block Matrix Multiplication with MPI + OpenMP

Distributed matrix multiplication using 1D block decomposition. Each process owns a contiguous block of rows, computes locally with OpenMP, and uses MPI collectives to assemble the result.

## Learning Objectives

- Decompose a matrix into row blocks across MPI processes
- Use `MPI_Bcast` to share common data and `MPI_Gather`/`MPI_Allgather` to assemble results
- Combine OpenMP thread parallelism within each MPI process
- Measure computation vs communication time

## Compilation

```bash
module load spack
module load cmake
module load openmpi5
mkdir -p build && cd build
cmake ..
make matmul-block-exercise matmul-block-solution
```

Or compile manually:

```bash
module load openmpi5
mpicc -O3 -Wall -fopenmp exercise.c -o matmul-block-exercise -lm
mpicc -O3 -Wall -fopenmp solution.c -o matmul-block-solution -lm
```

## Parameters

- `--N <n>`: matrix rows (default: 512)
- `--K <n>`: inner dimension (default: 512)
- `--M <n>`: matrix columns (default: 512)

Increase values to see how compute and communication time scale with matrix size.

## Execution

```bash
# Single node, multiple processes (default: N=512)
mpirun -np 4 ./bin/matmul-block-solution

# Larger matrices
mpirun -np 4 ./bin/matmul-block-solution --N 1024 --K 1024 --M 1024

# With explicit process binding
mpirun -np 4 --map-by socket ./bin/matmul-block-solution
```

### Running on the INF0090 Cluster (CPU partition)

Interactive with `srun` (adjust `--ntasks` and `--cpus-per-task` for OpenMP):
```bash
# 4 MPI processes, 1 OpenMP thread each (fits a 4-core CPU node)
srun --partition=cpu --nodes=1 --ntasks=4 --cpus-per-task=1 --mpi=pmix \
  bash -c 'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}; ./bin/matmul-block-solution'

# Or 2 MPI processes with 2 OpenMP threads each
srun --partition=cpu --nodes=1 --ntasks=2 --cpus-per-task=2 --mpi=pmix \
  bash -c 'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}; ./bin/matmul-block-solution'
```

Via batch script (`job.slurm`):
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
srun ./bin/matmul-block-solution
```
Submit: `sbatch job.slurm`

## Expected Output

```
[solution] rank=0 matmul N=512 K=512 M=512 bcast_time=0.001234s comp_time=0.045678s gather_time=0.001234s total=0.046912s | PASS
[solution] rank=1 matmul N=512 K=512 M=512 bcast_time=0.001234s comp_time=0.045678s gather_time=0.001234s total=0.046912s | PASS
```

## Hints

- Each process owns `local_rows = N / size` rows (assuming N divisible by size)
- Matrix A_local: `local_rows × K`, Matrix B: `K × M` (full), Matrix C_local: `local_rows × M`
- Broadcast B from rank 0 to all processes
- Compute C_local = A_local × B using `#pragma omp parallel for`
- Gather all C_local blocks into C on rank 0 (or use Allgather)
- Verify by comparing against a sequential reference on rank 0
