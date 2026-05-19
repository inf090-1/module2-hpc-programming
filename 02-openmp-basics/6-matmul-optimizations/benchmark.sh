#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=benchmark.out

export LD_LIBRARY_PATH=/opt/ohpc/pub/libs/gnu14/openblas/0.3.29/lib:$LD_LIBRARY_PATH

gcc -O3 -march=native matmul_serial.c -o matmul_serial
gcc -fopenmp -O3 -march=native matmul_omp_simd.c -o matmul_simd
gcc -fopenmp -O3 -march=native -I/opt/ohpc/pub/libs/gnu14/openblas/0.3.29/include -L/opt/ohpc/pub/libs/gnu14/openblas/0.3.29/lib matmul_omp_blas.c -o matmul_blas -lopenblas

echo "THREADS,TYPE,TIME" > results.csv

serial_time=$(./matmul_serial | grep Time | awk '{print $5}')
echo "1,SERIAL,$serial_time" >> results.csv

for threads in 1 2 4; do
    export OMP_NUM_THREADS=$threads
    export OPENBLAS_NUM_THREADS=$threads
    
    simd_time=$(./matmul_simd | grep Time | awk '{print $7}')
    echo "$threads,SIMD,$simd_time" >> results.csv
    
    blas_time=$(./matmul_blas | grep Time | awk '{print $7}')
    echo "$threads,BLAS,$blas_time" >> results.csv
done
