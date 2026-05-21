# 0. Hello GPU

This introductory lesson presents the HIP execution model and device-side `printf`.

## Learning Objectives
- Launch your first HIP kernel.
- Understand how block and thread indices identify each GPU worker.
- Print messages from GPU threads.

## Exercise
- `exercise_hello_gpu.cpp`: implement the GPU hello kernel.
- `solution_hello_gpu.cpp`: reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile on the login node:

```bash
hipcc -O3 --offload-arch=gfx942 exercise_hello_gpu.cpp -o hello_gpu

```

### Running directly with `srun`

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --gpus-per-node=1 \
  bash -lc 'export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./hello_gpu'
```

### Running via Batch Script (`sbatch`)

Create `job.slurm`:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=slurm-%j.out

module load rocm || true
export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH

./hello_gpu
```

Submit:

```bash
sbatch job.slurm
```

## Question
- How many hello lines are printed when you change grid/block size?
