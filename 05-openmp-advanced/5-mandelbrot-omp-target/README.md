# 5. Mandelbrot with Multi-Device OpenMP Target Offload

This lesson demonstrates **multi-device OpenMP target offload** by computing the Mandelbrot set across multiple GPUs.

## Approach

The image is split into **vertical strips**, one per device. Each device runs a single `target teams distribute parallel for` loop over a **flattened 1D index** spanning its strip. This gives the runtime a large, contiguous iteration space to distribute across all available compute units:

```
#pragma omp parallel num_threads(device_count)
{
    int dev = omp_get_thread_num();
    int y_start = dev * H / device_count;
    int y_end = (dev + 1) * H / device_count;

    #pragma omp target data device(dev) map(tofrom: dev_image[0:W*H])
    #pragma omp target teams distribute parallel for device(dev) \
        num_teams(512) thread_limit(256)
    for (int idx = 0; idx < (y_end - y_start) * W; ++idx) {
        int py = y_start + idx / W;
        int px = idx % W;
        // compute mandelbrot at (px, py)
    }
}
```

Key points:
- **No `#pragma omp task`** is used in the offload path.
- Each device receives a contiguous set of rows (vertical strip).
- A **flattened 1D loop** (instead of `collapse(2)`) ensures the OpenMP runtime sees a single contiguous iteration space for better work distribution.
- A large problem size (`W=4096, H=4096, MAX_ITER=5000`) is essential to saturate the 220 CUs of each MI300X GPU.

## Learning Objectives
- Use `omp_get_num_devices()` and `omp_get_thread_num()` from within a parallel region for multi-device dispatch.
- Use `#pragma omp target data device(dev)` to allocate data on each device.
- Use `#pragma omp target teams distribute parallel for device(dev)` to offload computation.
- Understand how problem size and loop structure affect GPU utilization.

## Parameters
| Parameter  | Default | Description                  |
|------------|---------|------------------------------|
| `WIDTH`    | 4096    | Image width (pixels)         |
| `HEIGHT`   | 4096    | Image height (pixels)        |
| `MAX_ITER` | 5000    | Maximum Mandelbrot iterations|

Override at compile time: `amdclang++ -DWIDTH=2048 -DHEIGHT=2048 ...`.

**Note**: Smaller sizes (e.g. 1024×1024) give the GPU too little work and produce weak speedups. 4096×4096 with 5000 iterations achieves >100× speedup on 2× MI300X.

GPU and CPU floating-point computations can lead to small differences in iteration counts; the code uses `ITER_EPS = MAX_ITER` tolerance when deciding PASS/FAIL.

## Exercise
Implement the multi-device version in:
- `exercise_mandelbrot.cpp`

Correctness is checked by comparing your GPU result against a CPU reference.
The program writes `mandelbrot.ppm` in the current directory.

## Compilation and Execution on the INF0090 Cluster

### Using the root Makefile (recommended)
```bash
cd /home/cl3t0/module2-hpc-programming/05-openmp-advanced
make 5-mandelbrot-omp-target/solution_mandelbrot
```

### Manual compilation
```bash
amdclang++ -O3 -fopenmp \
  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=gfx942 \
  -std=c++17 \
  exercise_mandelbrot.cpp -o exercise_mandelbrot
```

### Running directly with `srun`
```bash
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 \
  --chdir=/home/cl3t0/module2-hpc-programming/05-openmp-advanced/5-mandelbrot-omp-target \
  bash -c 'export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH; \
           export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}; \
           ./exercise_mandelbrot'
```

### Running via Batch Script (`sbatch`)
Create `job.slurm`:
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/cl3t0/module2-hpc-programming/05-openmp-advanced/5-mandelbrot-omp-target

export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
./exercise_mandelbrot
```
Submit:
```bash
sbatch job.slurm
```

## Output image

After running, the program writes `mandelbrot.ppm` in the current directory.
Convert to PNG:
```bash
magick mandelbrot.ppm mandelbrot.png
```

### Example result (4096×4096, 5000 iterations, 2× MI300X)

![Mandelbrot fractal](mandelbrot.png)

## Questions
- Why is a 4096×4096 image necessary to get good GPU speedup? What happens at 1024×1024?
- How does the flattened 1D loop (`idx = 0 ... (y_end-y_start)*W`) differ from `collapse(2)`?
- What changes when you use only 1 GPU (`--gres=gpu:1`)? Does performance scale linearly with device count?
- Why does the solution use `#pragma omp parallel num_threads(device_count)` instead of spawning `device_count` separate target regions?
