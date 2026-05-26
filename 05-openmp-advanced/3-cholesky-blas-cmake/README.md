# 3. Block Cholesky with BLAS kernels + OpenMP tasks (dependency graph)

In this lesson you will implement/understand a **blocked (in-place) Cholesky factorization** with parallelism via **OpenMP tasks**.
The heavy operations use **BLAS** calls:
- `dtrsm` (triangular solve) to compute panel blocks `L(i,k)`
- `dsyrk` to update diagonal blocks of the trailing submatrix
- `dgemm` to update off-diagonal blocks

The diagonal block `L(k,k)` factorization uses a simple loop-based Cholesky to keep the example short; the main parallelism and updates come from tasks + BLAS.

## Files
- `cholesky_blas.c`: complete implementation (OpenMP tasks + BLAS).
- `CMakeLists.txt`: CMake build to find and link a BLAS library and enable OpenMP.

## Build with CMake

```bash
cd 05-openmp-advanced/3-cholesky-blas-cmake
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Executable:
- `build/cholesky_blas`

## Execution
```bash
./build/cholesky_blas 512 64 0
```

- `n` = size of the SPD matrix
- `block` = tile size used to partition the matrix
- `seed` = random generator seed

The program prints a checksum (`diag_sum`) and a `residual_Frob` to validate the factorization.

## Task structure (quick overview)
For each block-column `k`, the algorithm runs in **phases** (separated by `#pragma omp taskwait`):
1. `factor(k,k)`: factor the diagonal block with plain loops.
2. `trsm(i,k)`: for each `i>k`, compute `L(i,k)` with `dtrsm` (parallel tasks).
3. `update(i,i)`: update each diagonal block with `dsyrk` (parallel tasks).
4. `update(i,j)`: update off-diagonal blocks with `dgemm` (for i>j, also parallel).

The `taskwait` phases ensure that the required updates finish before advancing to the next `k`.

## Questions
- How does the `block` size affect:
  - the parallelism exposed through tasks?
  - the size of the BLAS kernel calls (Level-3 efficiency)?
- Does the speedup come mainly from tasks (CPU parallelism) or from the BLAS library's own internal parallelism (depending on the library)?
