# Day 5 - OpenMP Advanced (Tasks + Target Offload)

This day extends OpenMP in two directions:
- **Tasks** for recursive / irregular parallel work.
- **OpenMP target offload** to run parts of your code on the GPU (AMD INF0090).

## Lessons
1. **Simple OpenMP Task examples** (Fibonacci, tree sum)
2. **TreeReduce** (matrix sum using an OpenMP task reduction tree)
3. **Cholesky with BLAS** (bloqueado + CMake)
4. **OpenMP target** (simple GPU vector add)
5. **Mandelbrot** (OpenMP target + quadtree, multiple target regions)
6. **Matmul comparison** (OpenMP target vs HIP)

All lesson content lives in subdirectories `1-...` to `6-...`.
See each lesson’s `README.md` for compilation and execution instructions.
