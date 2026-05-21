# Day 3 - Basic GPU Programming with ROCm/HIP

This day follows the same lesson-oriented structure used in `../02-openmp-basics`.
Each exercise is now isolated in its own folder so students can focus on one concept at a time.

## Quick Cheat Sheet

| Task | Command / API |
|---|---|
| Compile HIP code (MI300X-safe) | `hipcc -O3 --offload-arch=native file.cpp -o app` |
| Launch a kernel | `kernel<<<grid, block>>>(...)` |
| Allocate device memory | `hipMalloc` |
| Copy memory | `hipMemcpy` |
| Time GPU kernels | `hipEventRecord` + `hipEventElapsedTime` |

## Learning Objectives

- Implement first HIP kernels, launch configurations, and device memory allocations.
- Compare OpenMP CPU and HIP GPU implementations.
- Measure CPU and GPU execution time for the same workload.
- Use shared memory to optimize a 2D stencil pattern.
- Compare naive GPU kernels with optimized libraries (rocBLAS).

## Course Structure (Lessons)

- **`0-hello-gpu/`**: Intro to GPU kernels and device-side `printf`.
- **`1-vector-add/`**: OpenMP vector add vs HIP vector add.
- **`2-stencil-2d/`**: OpenMP 2D stencil vs HIP 2D naive stencil.
- **`3-shared-mem-stencil/`**: 2D stencil optimized with HIP shared memory.
- **`4-matmul-naive/`**: Naive matrix multiplication with HIP vs rocBLAS performance.

## Build Everything

From this folder:

```bash
make
```

This compiles both exercise and solution files in all lessons.

To force a specific GPU architecture (for MI300X use `gfx942`):

```bash
make HIP_ARCH=gfx942
```

## Cleaning

```bash
make clean
```

## Cluster Run Notes

Each lesson README includes direct commands for compile + `srun` and an `sbatch` script template for the INF0090 cluster (GPU partition with ROCm).
Follow those instructions from inside each lesson directory.

## MI300X Notes (INF0090)

- Use `--offload-arch=native` when compiling directly on a GPU node, or `--offload-arch=gfx942` for explicit MI300X binaries.
- On some nodes, `module load rocm` may be unavailable even though ROCm exists at `/opt/rocm`.
- If execution fails with `libomp.so: cannot open shared object file`, export runtime libraries:

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
```

- When running with Slurm, set environment variables inside the launched shell:

```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  bash -lc 'export OMP_NUM_THREADS=8; export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH; ./your_binary'
```
