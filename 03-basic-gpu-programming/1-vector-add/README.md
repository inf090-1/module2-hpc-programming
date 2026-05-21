# 1. Vector Add: OpenMP vs HIP

This lesson compares a CPU OpenMP vector addition with a GPU HIP kernel.

## Learning Objectives
- Measure OpenMP CPU baseline performance.
- Implement HIP vector addition on the GPU.
- Compare correctness and runtime.

## Exercise
- `exercise_vector_add_hip.cpp`: implement `vector_add_hip`.
- `solution_vector_add_hip.cpp`: reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile on the login node:

```bash
hipcc -O3 -fopenmp --offload-arch=gfx942 exercise_vector_add_hip.cpp -o vector_add

```

### Running directly with `srun`

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  bash -lc 'export OMP_NUM_THREADS=8; export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./vector_add'
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

./vector_add
```

Submit:

```bash
sbatch job.slurm
```

## Question
- How does speedup change when you increase vector size?
