#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00

gcc -fopenmp 1-parallel-for/examples/vector_add_parallel_reference.c -o vec_add
./vec_add
