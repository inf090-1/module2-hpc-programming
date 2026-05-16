#!/bin/bash
#SBATCH --job-name=scaling_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=scaling_test.out

# Compile the programs
gcc -O3 -fopenmp compute_heavy.c -o compute_heavy
gcc -O3 -fopenmp memory_bound.c -o memory_bound

# Initialize CSV file
CSV_FILE="scaling_results.csv"
echo "Threads,Program,ScalingType,Time" > $CSV_FILE

THREADS="1 2 4"

echo "Running experiments..."

for t in $THREADS; do
    # --- Compute Heavy ---
    # Strong Scaling (Fixed Size: multiplier 4)
    OUTPUT=$(srun --exact -N 1 -n 1 --cpus-per-task=$t env OMP_NUM_THREADS=$t ./compute_heavy 4)
    TIME=$(echo "$OUTPUT" | sed -n 's/.*Time=\(.*\) seconds/\1/p')
    echo "$t,compute_heavy,strong,$TIME" >> $CSV_FILE

    # Weak Scaling (Scaled Size: multiplier t)
    OUTPUT=$(srun --exact -N 1 -n 1 --cpus-per-task=$t env OMP_NUM_THREADS=$t ./compute_heavy $t)
    TIME=$(echo "$OUTPUT" | sed -n 's/.*Time=\(.*\) seconds/\1/p')
    echo "$t,compute_heavy,weak,$TIME" >> $CSV_FILE

    # --- Memory Bound ---
    # Strong Scaling (Fixed Size: multiplier 1)
    OUTPUT=$(srun --exact -N 1 -n 1 --cpus-per-task=$t env OMP_NUM_THREADS=$t ./memory_bound 1)
    TIME=$(echo "$OUTPUT" | sed -n 's/.*Time=\(.*\) seconds/\1/p')
    echo "$t,memory_bound,strong,$TIME" >> $CSV_FILE

    # Weak Scaling (Scaled Size: multiplier t)
    OUTPUT=$(srun --exact -N 1 -n 1 --cpus-per-task=$t env OMP_NUM_THREADS=$t ./memory_bound $t)
    TIME=$(echo "$OUTPUT" | sed -n 's/.*Time=\(.*\) seconds/\1/p')
    echo "$t,memory_bound,weak,$TIME" >> $CSV_FILE
done

echo "Experiments completed. Results saved to $CSV_FILE"
