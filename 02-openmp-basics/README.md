# Day 2 - Programming with OpenMP

This day moves from ideas to code. You will start with OpenMP multithreading, then use `omp parallel` and `omp for` to split work across threads. After that you will look at synchronization, race conditions, critical sections, loop scheduling, and the difference between DOALL and DOACROSS loops.

## Quick Cheat Sheet

| Task | Directive / Command |
|---|---|
| Start a parallel region | `#pragma omp parallel` |
| Split loop iterations | `#pragma omp parallel for` |
| Protect a shared update | `#pragma omp critical` |
| Choose a schedule | `schedule(static)` / `schedule(dynamic)` |
| Describe independent vs dependent loops | DOALL/DOACROSS |

## Learning Objectives

- Understand the OpenMP multithreading model.
- Use `omp parallel` to create a thread team.
- Use `omp for` to distribute loop iterations.
- Use OpenMP locks for simple mutual exclusion.
- Recognize race conditions and fix them with synchronization.
- Use `atomic`, `critical`, `barrier`, and locks when appropriate.
- Compare `critical` sections with `reduction`.
- Choose a loop schedule that matches the workload.
- Distinguish DOALL loops from DOACROSS loops.

## Theory

### Introduction to Multithreading
OpenMP creates a team of threads and lets the runtime split work across them.
The programmer decides where parallelism exists; the runtime decides how to map it to threads.

### OpenMP parallel
`#pragma omp parallel` starts a region that multiple threads enter together.
Code inside the region runs concurrently unless additional directives restrict it.

### OpenMP for
`#pragma omp parallel for` distributes loop iterations among threads.
This is the simplest way to express a DOALL loop, where iterations are independent.

### Synchronization / Race condition
A race condition appears when two or more threads update shared data without coordination.
OpenMP gives you multiple tools to fix it, including `critical`, `atomic`, and `reduction`.

### Critical section
`#pragma omp critical` lets only one thread enter a region at a time.
It is easy to use, but it can become a bottleneck if the protected work is large.

### Loop scheduling
The `schedule` clause controls how loop iterations are assigned to threads.
Use `static` for regular work, `dynamic` for irregular work, and `guided` when the workload changes over time.

### DOALL/DOACROSS
- **DOALL**: every iteration is independent, so `omp parallel for` works well.
- **DOACROSS**: later iterations depend on earlier ones, so you need dependencies, synchronization, or a different algorithm.

## Course Structure (Lessons)

The examples and exercises have been broken down into specific topics (lessons). Navigate to each directory for specific learning objectives, reference examples, exercises, and cluster execution instructions.

- **`1-parallel-for/`**: Multithreading basics, parallelizing loops (`vector_add_parallel`, `exercise_parallel_for`).
- **`2-synchronization/`**: Critical sections, atomics, locks, and barriers (`critical_section`, `atomic_counter`, `lock_counter`, `barrier_phases`).
- **`3-reduction/`**: Race-free accumulation and aggregations (`reduction_reference`, `exercise_reduction`).
- **`4-scheduling/`**: Tuning work distribution for regular and irregular workloads (`scheduling`, `schedule_irregular`).
- **`5-data-dependencies/`**: DOALL vs. DOACROSS dependency management (`doall_doacross`).
- **`6-matmul-optimizations/`**: Compare optimized OpenMP+SIMD vs BLAS for Matrix Multiplication.

Each topic that has `exercises/` now also provides a corresponding `solutions/` folder.

## Common Failure Modes

- **Race conditions**: shared updates without `critical`, `atomic`, `locks`, or `reduction`.
- **Missing barrier**: `nowait` removes the implicit sync, so dependent phases can run too early.
- **Overusing `critical`**: it works, but it can serialize too much code.
- **Bad schedule choice**: `static` is fast for balanced work, `dynamic` helps when work is uneven.
- **Confusing DOALL and DOACROSS**: dependency-free loops are easy to parallelize; dependency-heavy loops are not.

## Further Reading

- [OpenMP specification](https://www.openmp.org/spec-html/)
- [OpenMP examples](https://www.openmp.org/resources/openmp-examples/)
- [Parallel programming patterns](https://patterns.eecs.berkeley.edu/)
