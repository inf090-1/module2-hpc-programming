# 2. Convolution vs MIOpen (HIP + MIOpen)

This exercise compares a **naive 2D 3x3 convolution** implemented in HIP against **MIOpen**.

## Learning Objectives
- Write a basic HIP convolution kernel and handle padding correctly.
- Validate GPU results against a library implementation (MIOpen).
- Interpret correctness checks (`max_abs_err`, PASS/FAIL) and tolerance.

## Exercise
- `exercise_conv2d_vs_miopen.cpp`: implement the 3x3 convolution accumulation in `conv2d_naive_3x3`.
- `solution_conv2d_vs_miopen.cpp`: reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile on the login node:

```bash
hipcc -O3 -fopenmp --offload-arch=gfx942 exercise_conv2d_vs_miopen.cpp -o exercise_conv2d_vs_miopen -lmiopen
```

### Running directly with `srun`

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  bash -lc 'export OMP_NUM_THREADS=8; export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./exercise_conv2d_vs_miopen'
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

./exercise_conv2d_vs_miopen
```

Submit:

```bash
sbatch job.slurm
```

## Question
- How does the naive kernel’s execution time compare to MIOpen, and which parts are likely responsible (kernel math vs. launch overhead vs. library optimizations)?