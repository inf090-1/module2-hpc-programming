# 4. OpenMP Target Offload — Compute-Intensive Vector Operation

This lesson shows the simplest useful OpenMP **target offload** pattern:
`#pragma omp target teams distribute parallel for`.

Instead of a trivial `c[i] = a[i] + b[i]` (which is PCIe-bandwidth-bound and shows *no* speedup on GPU), we use an artificially compute-intensive kernel (`compute_busy`) with `COMPUTE_ITERS=50` iterations per element. This makes the kernel compute-bound and demonstrates genuine GPU acceleration.

## Learning Objectives
- Understand the basic structure of an OpenMP target offload.
- Use `teams distribute parallel for` to execute a loop on the GPU.
- Use `num_teams(512) thread_limit(256)` to control GPU resource usage.
- Know (at least conceptually) how `map(...)` controls host/device data movement.
- Understand why a compute-bound kernel is necessary to saturate a GPU.

## Parameters
| Parameter       | Default  | Description                              |
|-----------------|----------|------------------------------------------|
| `COMPUTE_ITERS` | 50       | Compute iterations per element           |

Override at compile time: `amdclang -DCOMPUTE_ITERS=100 ...`.

The vector size is fixed at `n = 1 << 24` (16,777,216 elements).

## Exercise
Implement the TODOs in:
- `exercise_vector_add.c`

The program computes `c[i] = compute_busy(a[i], b[i])` on the GPU and validates against CPU baselines.

## Compilation and Execution on the INF0090 Cluster

### Using the root Makefile (recommended)
```bash
cd /home/cl3t0/module2-hpc-programming/05-openmp-advanced
make 4-omp-target-vector-add/solution_vector_add
```

### Manual compilation
```bash
amdclang -O3 -fopenmp \
  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx942 \
  exercise_vector_add.c -o exercise_vector_add
```

### Running directly with `srun`
```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
  --chdir=/home/cl3t0/module2-hpc-programming/05-openmp-advanced/4-omp-target-vector-add \
  bash -c 'export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH; \
           export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}; \
           ./exercise_vector_add'
```

### Running via Batch Script (`sbatch`)
Create `job.slurm`:
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/cl3t0/module2-hpc-programming/05-openmp-advanced/4-omp-target-vector-add

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
./exercise_vector_add
```
Submit:
```bash
sbatch job.slurm
```

## Questions
- What happens to the speedup if you set `COMPUTE_ITERS=1` (i.e., a trivial `a[i]+b[i]`)?
- Why do we add `volatile float sink = c[n-1]` and validation loops in the CPU baselines?
- How does `num_teams(512) thread_limit(256)` affect performance? Try removing them.
