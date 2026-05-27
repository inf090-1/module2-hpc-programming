# Day 6 — MPI

> **⚠️ GPU Execution Note:** When running on the GPU partition (`srun -p gpu`), use `--mpi=pmix` for correct MPI rank assignment. Without this flag, `srun` launches each process as `rank=0` in its own MPI universe. Example: `srun -p gpu --gres=gpu:N -n N --mpi=pmix ./binary`

This module introduces distributed-memory parallel programming with MPI, from basic point-to-point communication to GPU-aware communication and non-blocking communication.

## Lessons

| # | Lesson | Description | Processes |
|---|--------|-------------|-----------|
| 1 | [Send/Recv](1-send-recv/README.md) | Pass a single datum between 2 processes | 2 |
| 2 | [Ping Pong](2-ping-pong/README.md) | Alternating send/receive with timing | 2 |
| 3 | [Ring](3-ring/README.md) | Token circulation in a process ring | any |
| 4 | [Collectives](4-collectives/README.md) | Bcast, Reduce, Allreduce with timing | 4 |
| 5 | [GPU-Aware MPI](5-gpu-aware-mpi/README.md) | MPI with ROCm/HIP and RCCL | 2+ (GPU) |
| 6 | [Matmul Block](6-matmul-block/README.md) | MPI+OpenMP distributed matmul | any |
| 7 | [Non-blocking](7-nonblocking/README.md) | Isend/Irecv + computation overlap via Jacobi 2D stencil | any |

## Compilation

```bash
# Standard MPI lessons
mkdir -p build && cd build
cmake ..
make -j

# GPU-aware MPI lessons (require ROCm)
cmake .. -DBUILD_GPU_LESSONS=ON
make -j
```

## Quick Reference

| Task | MPI Call |
|---|---|
| Initialize | `MPI_Init(&argc, &argv)` |
| Finalize | `MPI_Finalize()` |
| Rank / Size | `MPI_Comm_rank` / `MPI_Comm_size` |
| Send | `MPI_Send(buf, count, type, dest, tag, comm)` |
| Recv | `MPI_Recv(buf, count, type, src, tag, comm, &status)` |
| Broadcast | `MPI_Bcast(buf, count, type, root, comm)` |
| Reduce | `MPI_Reduce(send, recv, count, type, op, root, comm)` |
| Allreduce | `MPI_Allreduce(send, recv, count, type, op, comm)` |
| Gather | `MPI_Gather(send, scount, stype, recv, rcount, rtype, root, comm)` |
| Isend | `MPI_Isend(buf, count, type, dest, tag, comm, &req)` |
| Irecv | `MPI_Irecv(buf, count, type, src, tag, comm, &req)` |
| Waitall | `MPI_Waitall(count, reqs, statuses)` |
| Timer | `MPI_Wtime()` |

## Notes

- Compiler: use `mpicc` (GCC + OpenMPI) for MPI-only code
- GPU lessons: use `hipcc` (AMD ROCm) with MPI includes/libs
- GPU lessons require `--mpi=pmix` with `srun` for correct MPI rank assignment
- RCCL uses NCCL-compatible API (`ncclComm_t`, `ncclAllReduce`, etc.)
- In multi-GPU RCCL code, call `hipSetDevice(rank)` so each MPI rank binds to a distinct GPU
- RCCL must be compiled on the GPU node (g1) — the head node has ROCm 7.2.3 while g1 has 7.2.0, causing runtime ABI mismatch

### Run Options Comparison

| Context | Command |
|---|---|
| CPU (login node) | `mpirun -np N ./binary` |
| CPU (via srun) | `srun -p cpu -n N --mpi=pmix ./binary` |
| GPU node (MPI-GPU) | `srun -p gpu --gres=gpu:N -n N --mpi=pmix ./binary` |
| GPU node (RCCL) | compile on g1 first, then run as MPI-GPU |
