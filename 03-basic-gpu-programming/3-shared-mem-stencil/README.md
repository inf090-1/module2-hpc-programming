# 3. 2D Stencil Optimized with Shared Memory

This lesson expands on the previous stencil example by leveraging shared memory to reduce global memory accesses and improve execution time.

## Learning Objectives
- Understand shared memory tiles and ghost zones (halos).
- Use `__shared__` memory and `__syncthreads()` in HIP.
- Observe performance improvements over the naive global memory implementation.

## Exercise
- `exercise_stencil_2d_hip.cpp`: implement `stencil_step_hip` using shared memory to load the 16x16 tile + 1px halo (18x18 total per block).
- `solution_stencil_2d_hip.cpp`: reference solution using shared memory.

## Compilation and Execution on the INF0090 Cluster

First, compile on the login node:

```bash
hipcc -O3 -fopenmp --offload-arch=gfx942 exercise_stencil_2d_hip.cpp -o stencil_2d_shmem

```

### Running directly with `srun`

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  bash -lc 'export OMP_NUM_THREADS=8; export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./stencil_2d_shmem'
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

./stencil_2d_shmem
```

Submit:

```bash
sbatch job.slurm
```

## AMD profiling comparison: naive vs shared-memory stencil

To understand whether the shared-memory version truly improves the *kernel* execution, we profiled both HIP solutions with **rocprofv3**.

### Profiling run
- **GPU:** MI300X (`gfx942`) on INF0090 GPU partition
- **Configuration:** `nx=4096`, `ny=4096`, `iters=50`
- **Programs:**
  - Naive: `./stencil_2d` (from `2-stencil-2d`)
  - Shared-memory: `./stencil_2d_shmem` (from `3-shared-mem-stencil`)
- **Tool:** `rocprofv3 --kernel-trace --memory-copy-trace --stats -S true -D true --summary-output-file ...`

Raw rocprofv3 outputs (latest tuning) were copied into:
- **Naive:** `amd-profiling-raw/stencil_naive_4096_summary.txt`
- **Shared-memory (tuned):** `amd-profiling-raw/stencil_shmem_4096_summary.txt`

The associated rocprofv3 results DBs were also overwritten with the latest run and kept under `amd-profiling-raw/`:
- `amd-profiling-raw/naive_4096x4096_results.db`
- `amd-profiling-raw/shmem_4096x4096_results.db`

### rocprofv3 commands used (reproducible)
```bash
NX=4096
NY=4096
ITERS=50

OUTDIR=./amd-profiling-raw
mkdir -p "$OUTDIR"

# Naive stencil (2-stencil-2d)
rocprofv3 --kernel-trace --memory-copy-trace --stats -S true -D true \
  --summary-output-file "$OUTDIR/stencil_naive_nx${NX}_ny${NY}_iters${ITERS}_summary.txt" \
  -- ./stencil_2d ${NX} ${NY} ${ITERS}

# Shared-memory stencil (3-shared-mem-stencil)
rocprofv3 --kernel-trace --memory-copy-trace --stats -S true -D true \
  --summary-output-file "$OUTDIR/stencil_shmem_nx${NX}_ny${NY}_iters${ITERS}_summary.txt" \
  -- ./stencil_2d_shmem ${NX} ${NY} ${ITERS}
```

### Kernel-dispatch timing (from rocprofv3 summaries)
Both implementations dispatch the same kernel symbol name `stencil_step_hip`, but with different machine code generated from different source / launch configuration.

> **Best shared-memory overall result (opt2, shared looked better end-to-end):**
> - Shared kernel uses `threads(32,8)` and a padded shared tile (`s_data[10][35]`).

| Version | Kernel avg duration (ns) | Interpretation |
|---|---:|---|
| **Naive (2-stencil-2d)** | **7.033e+04** ns (≈ 70.33 µs) | Baseline |
| **Shared-mem (tuned 3-shared-mem-stencil)** | **7.273e+04** ns (≈ 72.73 µs) | ~3.4% slower kernel avg |

Estimated kernel-only ratio (tuned shared / naive):
- 72.73 / 70.33 ≈ **1.034×** (≈ **3.4% slower**)

### Application-level timing (from program stdout in rocprofv3 compare-opt2)
Although kernel-dispatch stats show the shared kernel slightly slower, the end-to-end timings favored the shared version in this specific job order:

- **Naive:** `gpu_mem_total_time=1.580257 s`, `speedup_with_mem=0.18x`
- **Shared:** `gpu_mem_total_time=0.760146 s`, `speedup_with_mem=0.36x`

So shared improves `speedup_with_mem` by about **2×** (0.36/0.18), mainly because the measured H2D+transfer+kernel total time was much lower for the shared run.

> Note: rocprofv3 runs `./stencil_2d` first and `./stencil_2d_shmem` second, so these transfer timings may be somewhat biased by first-run initialization.

### Why shared memory isn’t faster in the *kernel-dispatch* stats
Shared still does extra **shared-memory traffic** + **halo load logic** + a **`__syncthreads()`** barrier per block. Those costs can outweigh reduced global reads if the naive kernel already benefits from caching/coalescing on MI300X.
