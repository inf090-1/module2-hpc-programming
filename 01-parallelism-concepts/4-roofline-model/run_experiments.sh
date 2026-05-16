#!/bin/bash
#SBATCH --job-name=roofline_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=roofline_test.out

# Compile the programs
gcc -O3 -fopenmp saxpy.c -o saxpy
gcc -O3 -fopenmp matmul_l1.c -o matmul_l1

# Initialize CSV file
CSV_FILE="roofline_results.csv"
echo "Threads,Algorithm,Time" > $CSV_FILE

THREADS="1 2 4"

echo "Running Roofline comparison experiments..."

for t in $THREADS; do
    # Run SAXPY (Memory Bound)
    OUTPUT=$(srun --exact -N 1 -n 1 --cpus-per-task=$t env OMP_NUM_THREADS=$t ./saxpy)
    TIME=$(echo "$OUTPUT" | sed -n 's/.*Time=\(.*\)/\1/p')
    echo "$t,SAXPY,$TIME" >> $CSV_FILE

    # Run MATMUL (Compute Bound)
    OUTPUT=$(srun --exact -N 1 -n 1 --cpus-per-task=$t env OMP_NUM_THREADS=$t ./matmul_l1)
    TIME=$(echo "$OUTPUT" | sed -n 's/.*Time=\(.*\)/\1/p')
    echo "$t,MATMUL,$TIME" >> $CSV_FILE
done

echo "Experiments completed. Results saved to $CSV_FILE"
