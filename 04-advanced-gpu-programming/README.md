# Day 4 - Advanced GPU Programming with ROCm (HIP + Libraries)

This day builds on basic GPU kernels and moves into more advanced topics:

## Learning Objectives
- Implement **tiled matrix multiplication** with a strong focus on **coalesced global memory access** (using shared memory for tiling).
- Compare a **naive convolution** against **MIOpen**.
- Implement a basic **scaled dot-product attention** pipeline (scores, softmax, weighted sum).
- Profile **HIP** applications using **AMD tools** (rocprof / rocminfo).

## Exercises (4 parts)
1. `1-matmul-coalescing/`  
   - `exercise_matmul_coalescing.cpp`
   - `solution_matmul_coalescing.cpp`

2. `2-convolution/`  
   - `exercise_conv2d_vs_miopen.cpp`
   - `solution_conv2d_vs_miopen.cpp`

3. `3-attention/`  
   - `exercise_attention_basic.cpp`
   - `solution_attention_basic.cpp`

4. `4-hip-profiling-amd-tools/`
   - `profiling_target_hip.cpp`
   - `profile_hip.sh`

## Running the solutions
All HIP programs print:
- CPU OpenMP time
- GPU kernel time (HIP events)
- GPU H2D/D2H transfer times (host timing)
- speedups: `speedup_kernel_only` and `speedup_with_mem`

Convolution additionally prints `miopen_time` and `speedup_miopen`.

Examples (run on the GPU node):
- Matmul: `./1-matmul-coalescing/solution_matmul_coalescing [n]`
- Convolution: `./2-convolution/solution_conv2d_vs_miopen [H] [W]`
- Attention: `./3-attention/solution_attention_basic [embed_dim]`

Control CPU threading with `OMP_NUM_THREADS` (recommended: 16 on the cluster).
## Common Failure Modes
- Wrong GPU / missing ROcm runtime setup.
- Missing library dependencies (especially for MIOpen).
- Profiling the wrong executable (ensure the binary path matches your rocprof command).

## Further Reading
- MIOpen documentation: https://github.com/ROCmSoftwarePlatform/MIOpen
- rocprof documentation: https://docs.amd.com/ (ROCm profiler / rocprof)
