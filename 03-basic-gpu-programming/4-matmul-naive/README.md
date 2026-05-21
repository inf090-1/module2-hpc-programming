# 4. Matrix Multiplication and rocBLAS with HIP

This lesson focuses on implementing a naive matrix multiplication kernel and comparing it with a highly optimized library (rocBLAS).

## Learning Objectives
- Write a basic matrix multiplication kernel on GPUs.
- Launch kernels and calculate blocks and grids layout.
- Use `rocblas` library for peak performance on AMD GPUs.
- Compare CPU OpenMP, naive GPU HIP, and rocBLAS runtimes.

## Exercise
- `exercise_matmul_hip.cpp`: implement the naive `matmul_naive_hip` kernel and its dispatch.
- `solution_matmul_hip.cpp`: reference solution for the naive approach.
- `solution_rocblas_matmul.cpp`: reference implementation using `rocBLAS` comparing sizes up to 4096.

## Compilation and Execution on the INF0090 Cluster

First, compile on the login node:

```bash
hipcc -O3 -fopenmp --offload-arch=native exercise_matmul_hip.cpp -o exercise_matmul_hip
hipcc -O3 -fopenmp --offload-arch=native -lrocblas solution_rocblas_matmul.cpp -o solution_rocblas_matmul
# or explicitly for MI300X:
# hipcc -O3 -fopenmp --offload-arch=gfx942 exercise_matmul_hip.cpp -o exercise_matmul_hip
```

### Running directly with `srun`

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  bash -lc 'export OMP_NUM_THREADS=8; export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./exercise_matmul_hip; ./solution_rocblas_matmul'
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

./exercise_matmul_hip
./solution_rocblas_matmul
```

Submit:

```bash
sbatch job.slurm
```

## AMD profiling results (solutions, MI300X / gfx942)

**Goal:** compare kernel time of the **naive HIP matmul** vs **rocBLAS**.

**Profiling method:** `rocprofv3` with `--kernel-trace --memory-copy-trace --stats --summary`.

**Run:** matrix size **n=1024**, traced **solutions** (`./matmul_hip 1024` and `./rocblas_matmul 1024`).

### Raw rocprofv3 outputs copied from the cluster
All raw outputs used for the tables below were copied into this folder:

- `amd-profiling-raw/naive_1024_results.db`
- `amd-profiling-raw/naive_1024_summary.txt`
- `amd-profiling-raw/rocblas_1024_results.db`
- `amd-profiling-raw/rocblas_1024_summary.txt`

### Profiling script used
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=profile-matmul-sols-%j.out

set -e
cd /home/cl3t0/03-basic-gpu-programming-agent-test

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH

OUTDIR=/home/cl3t0/03-basic-gpu-programming-agent-test/4-matmul-naive/amd-profiling
mkdir -p "$OUTDIR"

# Naive HIP matmul
rocprofv3 --kernel-trace --memory-copy-trace --stats \
  -S true -D true \
  --summary-output-file "$OUTDIR/naive_n1024_summary.txt" \
  -- ./matmul_hip 1024

# rocBLAS matmul
rocprofv3 --kernel-trace --memory-copy-trace --stats \
  -S true -D true \
  --summary-output-file "$OUTDIR/rocblas_n1024_summary.txt" \
  -- ./rocblas_matmul 1024

echo "profiling done"
```

### Snippet of the rocprofv3 “kernel dispatch” summary
```text
# Naive HIP (avg kernel duration)
matmul_naive_hip(float const*, float const*, float*, int) | CALLS=4 | AVG≈648100 ns

# rocBLAS (avg main kernel duration)
Cijk_Ailk_Bljk_...(rocBLAS) | CALLS=4 | AVG≈23910 ns
```


### Naive HIP (kernel: `matmul_naive_hip(float const*, float const*, float*, int)`)
From `naive_n1024_summary.txt.txt`:

| Kernel | Calls | Total duration (ns) | Avg duration (ns) |
|---|---:|---:|---:|
| `matmul_naive_hip(...)` | 4 | 2,592,576 | 648,100 |

Memory copies (same run, for context):

| Copy | Calls | Avg duration (ns) |
|---|---:|---:|
| Host → Device | 8 | 93,271 |
| Device → Host | 4 | 101,300 |

### rocBLAS (kernel: `Cijk_Ailk_Bljk_..._WGM8`)
From `rocblas_n1024_summary.txt.txt`:

| Kernel | Calls | Total duration (ns) | Avg duration (ns) |
|---|---:|---:|---:|
| `Cijk_Ailk_Bljk_...(rocBLAS)` | 4 | 95,621 | 23,910 |
| `__amd_rocclr_fillBufferAligned` | 4 | 24,117 | 6,029 |

Memory copies (same run, for context):

| Copy | Calls | Avg duration (ns) |
|---|---:|---:|
| Host → Device | 8 | 101,200 |
| Device → Host | 4 | 110,400 |

### Comparison (kernel time)
- **Naive kernel avg:** ~648,100 ns (≈ 0.648 ms)
- **rocBLAS main-kernel avg:** ~23,910 ns (≈ 0.0239 ms)
- **Kernel-only speedup (rough):** 0.648 ms / 0.0239 ms ≈ **27×**

> Why rocBLAS is better: the naive kernel does a straightforward `O(n^3)` loop with no tiling/shared-memory/register blocking and no structure-specific optimization. rocBLAS uses highly-tuned GPU microkernels (tiling like `MT64x64x32`, vectorized math, and optimized instruction/data movement) so the same arithmetic work runs far more efficiently.

### Connection to your measured program timers
The program also reported **much smaller `gpu_rocblas_time`** than `gpu_hip_time` for the same `n=1024`, consistent with the rocprofv3 kernel-dispatch summaries above.

## Question
- How does the naive implementation scale compared to rocBLAS when matrix sizes increase?
