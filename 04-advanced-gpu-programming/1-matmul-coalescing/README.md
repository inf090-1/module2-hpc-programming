# 1. Matmul with Coalesced Memory Access (HIP)

This lesson focuses on writing an efficient **tiled matrix multiplication** (GEMM) in HIP, with the main goal of producing **coalesced global memory loads**.

## Learning Objectives
- Use tiling + `__shared__` memory to reduce redundant global memory traffic.
- Implement global memory loads for `A` and `B` so that accesses are **coalesced**.
- Compute `C = A * B` correctly and validate with the provided CPU reference.

## Exercise
- `exercise_matmul_coalescing.cpp`: implement the tiled loads into `As` and `Bs` and the per-tile compute loop in `matmul_tiled_shmem`.
- `solution_matmul_coalescing.cpp`: reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile on the login node:

```bash
hipcc -O3 --offload-arch=gfx942 exercise_matmul_coalescing.cpp -o exercise_matmul_coalescing
```

### Running directly with `srun`

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  bash -lc 'export OMP_NUM_THREADS=8; export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./exercise_matmul_coalescing'
```

### Running via Batch Script (`sbatch`)

Create `job.slurm`:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

module load rocm || true
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH

./exercise_matmul_coalescing
```

Submit:

```bash
sbatch job.slurm
```

## Question
- How does changing the tile size (`TILE`, via `-DTILE=...`) affect runtime on the GPU?