# 3. Scaled Dot-Product Attention (Basic) on HIP

This exercise implements a small, educational version of **scaled dot-product attention** on the GPU.

## Learning Objectives
- Implement the GPU kernel for attention **scores**: `S = (Q·K^T)/sqrt(D)`.
- Implement per-row **softmax**: `P = softmax(S)`.
- Compute the final attention output: `O = P·V`.
- Validate correctness against the provided CPU reference.

## Exercise
- `exercise_attention_basic.cpp`: implement the TODOs in:
  - `compute_scores`
  - `softmax_rows`
  - `compute_output`
- `solution_attention_basic.cpp`: reference solution.

## Compilation and Execution on the INF0090 Cluster

First, compile on the login node:

```bash
hipcc -O3 --offload-arch=gfx942 exercise_attention_basic.cpp -o exercise_attention_basic
```

### Running directly with `srun`

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  bash -lc 'export OMP_NUM_THREADS=8; export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./exercise_attention_basic'
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

./exercise_attention_basic
```

Submit:

```bash
sbatch job.slurm
```

## Question
- If you increase `SEQ_LEN` (and recompile), what changes in numerical stability and runtime do you expect from the current softmax implementation?