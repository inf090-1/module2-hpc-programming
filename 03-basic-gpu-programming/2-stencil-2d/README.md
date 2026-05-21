# 2. 2D Stencil: OpenMP vs HIP

This lesson compares a CPU OpenMP 2D stencil and a GPU HIP stencil.

## Learning Objectives
- Understand nearest-neighbor stencil update patterns.
- Implement a 2D HIP kernel with proper boundary handling.
- Compare OpenMP and HIP execution times.

## Exercise
- `exercise_stencil_2d_hip.cpp`: implement `stencil_step_hip`.
- `solution_stencil_2d_hip.cpp`: reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile on the login node:

```bash
hipcc -O3 -fopenmp --offload-arch=native exercise_stencil_2d_hip.cpp -o stencil_2d
# or explicitly for MI300X:
# hipcc -O3 -fopenmp --offload-arch=gfx942 exercise_stencil_2d_hip.cpp -o stencil_2d
```

### Running directly with `srun`

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  bash -lc 'export OMP_NUM_THREADS=8; export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./stencil_2d'
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

./stencil_2d
```

Submit:

```bash
sbatch job.slurm
```

## Question
- Why must boundary cells be treated differently from interior cells?
