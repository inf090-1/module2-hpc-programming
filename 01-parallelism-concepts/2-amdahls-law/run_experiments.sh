#!/bin/bash
#SBATCH --job-name=amdahl_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=amdahl_test.out

# Compile the C++ program
g++ -O3 -fopenmp pi_calc.cpp -o pi_calc

# Initialize CSV file
CSV_FILE="amdahl_results.csv"
echo "Threads,Time" > $CSV_FILE

THREADS="1 2 4"

echo "Running Amdahl's Law experiments..."

for t in $THREADS; do
    OUTPUT=$(srun --exact -N 1 -n 1 --cpus-per-task=$t env OMP_NUM_THREADS=$t ./pi_calc)
    TIME=$(echo "$OUTPUT" | sed -n 's/.*Time: \(.*\) seconds/\1/p')
    echo "$t,$TIME" >> $CSV_FILE
done

echo "Experiments completed. Results saved to $CSV_FILE"
