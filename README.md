# Module 2 - Programming in HPC

This module moves from using the cluster to programming for it. Day 1 focuses on parallelism, hardware limits, and performance models. Day 2 introduces OpenMP multithreading and synchronization. Day 3 covers AMD GPU programming with ROCm/HIP. Day 4 moves into GPU libraries and profiling. Day 5 teaches OpenMP tasks and target offload. Day 6 introduces MPI and distributed-memory parallelism.

**What You Will Learn**

- How to reason about strong and weak scaling, Amdahl's law, Gustafson's law, and Roofline
- How to inspect NUMA, PCIe, memory, and GPU topology with common Linux tools
- How to write OpenMP programs with `parallel`, `for`, synchronization, locks, critical sections, scheduling, tasks, and target offload
- How to write basic HIP programs, use shared memory, and reason about CUDA vs. HIP
- How to use ROCm libraries, tracing tools, and Roofline-style thinking
- How to use MPI for collectives, topologies, and GPU-aware communication

**Lesson Overview**

- [01-parallelism-concepts](01-parallelism-concepts/README.md): scaling concepts, HPC architecture, inspection tools, C/C++, Python, and Roofline.
- [02-openmp-basics](02-openmp-basics/README.md): multithreading, `omp parallel`, `omp for`, synchronization, atomics, locks, critical sections, loop scheduling, and DOALL/DOACROSS.
- [03-basic-gpu-programming](03-basic-gpu-programming/README.md): ROCm/HIP basics, kernels, shared memory, and basic profiling with rocprofv3.
- [04-advanced-gpu-programming](04-advanced-gpu-programming/README.md): coalescing, MIOpen, attention, tracing, and profiling.
- [05-openmp-gpu](05-openmp-gpu/README.md): OpenMP tasks, then OpenMP target offload and profiling.