# 1. Simple OpenMP Task examples (Fibonacci, Tree sum)

This lesson gives you a first, concrete feel for **OpenMP tasks**.
You will implement recursive/irregular computations (Fibonacci and a binary tree sum) where `for` loops alone are not a natural fit.

## Learning Objectives
- Understand how `#pragma omp task` creates irregular parallel work.
- Use `#pragma omp taskgroup` (or `taskwait`) so parent computations wait for children.
- Use `#pragma omp parallel` + `#pragma omp single` as the entry point for task creation.
- See how a **cutoff** (`FIB_CUTOFF`, `PER_NODE_WORK`) limits task overhead.

## Parameters
| Parameter       | Default | Description                                    |
|-----------------|---------|------------------------------------------------|
| `FIB_CUTOFF`    | 20      | Only create tasks for `n > CUTOFF` (Fibonacci) |
| `TREE_DEPTH`    | 12      | Depth of the perfect binary tree (tree sum)    |
| `PER_NODE_WORK` | 10000   | Artificial work per tree node (loop iterations)|

Override at compile time, e.g. `gcc -DFIB_CUTOFF=10 ...`.

## Exercises
Implement the marked parts in these files:
- `exercise_fibonacci_tasks.c`: implement `fib_task()` using tasks (Fibonacci n=30).
- `exercise_tree_tasks.c`: implement `tree_sum()` using tasks (depth=12, 4095 nodes).

Each exercise compares the computed result against a known expected value.

## Compilation and Execution on the INF0090 Cluster

### Using the root Makefile (recommended)
```bash
cd /home/cl3t0/module2-hpc-programming/05-openmp-advanced
make 1-simple-openmp-task/solution_fibonacci_tasks
make 1-simple-openmp-task/solution_tree_tasks
```

### Manual compilation
```bash
gcc -O3 -fopenmp exercise_fibonacci_tasks.c -o exercise_fibonacci_tasks
gcc -O3 -fopenmp exercise_tree_tasks.c -o exercise_tree_tasks
```

### Running directly with `srun`
```bash
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 --chdir=/home/cl3t0/module2-hpc-programming/05-openmp-advanced/1-simple-openmp-task \
  bash -c 'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}; ./exercise_fibonacci_tasks'
srun --partition=cpu --nodes=1 --ntasks=1 --cpus-per-task=4 --chdir=/home/cl3t0/module2-hpc-programming/05-openmp-advanced/1-simple-openmp-task \
  bash -c 'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}; ./exercise_tree_tasks'
```

### Running via Batch Script (`sbatch`)
Create `job_fib.slurm`:
```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/cl3t0/module2-hpc-programming/05-openmp-advanced/1-simple-openmp-task

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
./exercise_fibonacci_tasks
```
Submit:
```bash
sbatch job_fib.slurm
```

## Questions
- Fibonacci: How does the runtime change when you adjust `FIB_CUTOFF`? What happens with no cutoff?
- Tree sum: How does `PER_NODE_WORK` affect the speedup? What if you set it to 1?
- Why might the task-based version not always outperform a simple `parallel for` for these small problems?
