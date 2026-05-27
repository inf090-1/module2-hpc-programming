# Lesson 5: GPU-Aware MPI with ROCm

Learn how MPI can communicate data residing in GPU memory using ROCm/HIP and RCCL.

## Learning Objectives

- Allocate GPU memory with `hipMalloc` and transfer data with `hipMemcpy`
- Use MPI collectives directly on GPU buffers (GPU-aware MPI)
- Compare MPI GPU-to-GPU communication vs RCCL direct GPU-to-GPU

## Exercises

Two exercises in this lesson:

| Exercise | File | Description |
|---|---|---|
| MPI GPU | `exercise_mpi_gpu.c` | MPI allreduce on GPU buffers (no RCCL) |
| RCCL GPU | `exercise_rccl_gpu.c` | RCCL allreduce on GPU buffers (bonus) |

## Compilation

With CMake (requires ROCm):

```bash
mkdir -p build && cd build
cmake .. -DBUILD_GPU_LESSONS=ON
make mpi-gpu-solution rccl-gpu-solution
```

Or compile manually:

```bash
# MPI GPU version (compile on head node or GPU node)
# First load an MPI module (e.g., openmpi5 which sets MPI_DIR)
module load openmpi5
hipcc -D__HIP_PLATFORM_AMD__ \
    -I${MPI_DIR}/include \
    solution_mpi_gpu.c \
    -L${MPI_DIR}/lib -lmpi -L/opt/rocm/lib -lamdhip64 \
    -o mpi-gpu-solution -O3

# RCCL version — MUST be compiled on the GPU node (g1)
# ROCm versions differ between head node (7.2.3) and g1 (7.2.0)
# Use srun to compile on g1:
srun -p gpu --gres=gpu:1 -n 1 --mpi=pmix bash -c '
  module load openmpi5
  hipcc -D__HIP_PLATFORM_AMD__ -O3 \
    -I${MPI_DIR}/include -I/opt/rocm/include \
    solution_rccl_gpu.c \
    -L${MPI_DIR}/lib -lmpi -L/opt/rocm/lib -lamdhip64 -lrccl \
    -o rccl-gpu-solution \
    -Wl,-rpath,/opt/rocm-7.2.0/lib'
```

## Execution

### MPI-GPU version (runs anywhere with ROCm)

```bash
srun -p gpu --gres=gpu:2 -n 2 --mpi=pmix ./mpi-gpu-solution
```

### RCCL version (requires build on g1)

```bash
srun -p gpu --gres=gpu:2 -n 2 --mpi=pmix ./rccl-gpu-solution
```

### Running on the INF0090 Cluster (GPU partition)

Interactive with `srun`:

```bash
srun --partition=gpu --gres=gpu:2 --ntasks=2 --mpi=pmix ./mpi-gpu-solution
srun --partition=gpu --gres=gpu:2 --ntasks=2 --mpi=pmix ./rccl-gpu-solution
```

Via batch script (`job.slurm`):

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --time=00:05:00

srun --mpi=pmix ./mpi-gpu-solution
```

Submit: `sbatch job.slurm`

## Expected Output

```
[solution] rank=0 MPI-GPU allreduce result=3.000000 | PASS
[solution] rank=1 MPI-GPU allreduce result=3.000000 | PASS
[solution] rank=0 RCCL allreduce result=3.000000 | PASS
[solution] rank=1 RCCL allreduce result=3.000000 | PASS
```

## Hints (MPI GPU)

- Use `hipMalloc((void**)&d_send, count * sizeof(float))` for GPU allocation
- Fill GPU buffer with `hipMemcpy(d_send, &val, sizeof(float), hipMemcpyHostToDevice)`
- MPI works directly with device pointers on GPU-aware MPI stacks
- Use separate send/recv buffers for reliability

## Hints (RCCL)

- Call `hipSetDevice(rank)` before initializing RCCL to bind each rank to a distinct GPU
- Rank 0 creates unique id with `ncclGetUniqueId` and broadcasts it via `MPI_Bcast`
- Each rank calls `ncclCommInitRank(&comm, size, id, rank)`
- Use `ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum, comm, 0)`
- Stream parameter `0` uses the default stream

## Note

- The MPI-GPU solution does **not** require RCCL and is the primary exercise.
- The RCCL version is a bonus to demonstrate direct GPU-to-GPU communication.
- RCCL must be compiled on the GPU node (g1) because the head node uses ROCm 7.2.3 while g1 uses 7.2.0, causing an ABI mismatch at runtime.
- When building on g1, use `-Wl,-rpath,/opt/rocm-7.2.0/lib` to ensure the correct runtime library is loaded.
