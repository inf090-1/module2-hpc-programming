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
module load spack
module load cmake
module load openmpi5
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=hipcc
make mpi-gpu-solution rccl-gpu-solution
```

Or compile manually:

```bash
module load openmpi5

# MPI GPU version
hipcc -D__HIP_PLATFORM_AMD__ --offload-arch=gfx942 \
    -I${MPI_DIR}/include -I/opt/rocm/include \
    solution_mpi_gpu.c \
    -L${MPI_DIR}/lib -lmpi -L/opt/rocm/lib -lamdhip64 \
    -o mpi-gpu-solution -O3

# RCCL version
hipcc -D__HIP_PLATFORM_AMD__ --offload-arch=gfx942 \
    -I${MPI_DIR}/include -I/opt/rocm/include \
    solution_rccl_gpu.c \
    -L${MPI_DIR}/lib -lmpi -L/opt/rocm/lib -lamdhip64 -lrccl \
    -o rccl-gpu-solution -O3
```

## Execution

### MPI-GPU version (runs anywhere with ROCm)

```bash
srun -p gpu --gres=gpu:2 -n 2 --mpi=pmix ./build/bin/mpi-gpu-solution
```

### RCCL version

```bash
srun -p gpu --gres=gpu:2 -n 2 --mpi=pmix ./build/bin/rccl-gpu-solution
```

### Running on the INRF090 Cluster (GPU partition)

Interactive with `srun`:

```bash
srun --partition=gpu --gres=gpu:2 --ntasks=2 --mpi=pmix ./build/bin/mpi-gpu-solution
srun --partition=gpu --gres=gpu:2 --ntasks=2 --mpi=pmix ./build/bin/rccl-gpu-solution
```

Via batch script (`job.slurm`):

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --time=00:05:00

srun --mpi=pmix ./build/bin/mpi-gpu-solution
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
- Compilation targets `gfx942` (MI300X). Both solutions compile on the head node and run on the GPU partition.
- Use `/opt/rocm` symlinks for linking — no `-Wl,-rpath` needed.

## Questions

1. Why can MPI operate directly on GPU pointers in GPU-aware MPI? What hardware feature enables this?
2. How does RCCL differ from MPI when performing an allreduce on GPU data? Which is likely faster and why?
3. Without GPU-aware MPI, how would you implement the same allreduce using only CPU buffers and explicit GPU copies?
