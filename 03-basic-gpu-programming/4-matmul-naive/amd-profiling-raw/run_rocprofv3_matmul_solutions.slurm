#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=profile-matmul-sols-%j.out

set -e
cd /home/cl3t0/03-basic-gpu-programming-agent-test

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH

OUTDIR=/home/cl3t0/03-basic-gpu-programming-agent-test/4-matmul-naive/amd-profiling
mkdir -p "$OUTDIR"

# Naive HIP matmul
rocprofv3 --kernel-trace --memory-copy-trace --stats \
  -S true -D true \
  --summary-output-file "$OUTDIR/naive_n1024_summary.txt" \
  -- ./matmul_hip 1024

# rocBLAS matmul
rocprofv3 --kernel-trace --memory-copy-trace --stats \
  -S true -D true \
  --summary-output-file "$OUTDIR/rocblas_n1024_summary.txt" \
  -- ./rocblas_matmul 1024

echo "profiling done"
