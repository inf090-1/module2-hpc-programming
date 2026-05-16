# Day 1: Parallelism Concepts & Cluster Familiarization

Welcome to the first practical session of the HPC Programming course. Today, we bridge the gap between theoretical parallel computing concepts and practical execution on real hardware. 

You will work on a Slurm-managed cluster containing both CPU and GPU nodes. By executing commands and running simple programs, you will inspect the underlying architecture, measure parallel scaling, observe the effects of memory layout, and analyze algorithmic bottlenecks.

## Course Structure for Today

The day is broken down into four practical lessons. Please navigate through the folders in order:

### [Lesson 1: Hardware Recognition](./1-hardware-recognition/README.md)
Get to know the cluster! You will use standard Linux and HPC tools (`lscpu`, `numactl`, `rocm-smi`) via Slurm allocations to inspect core counts, cache sizes, NUMA layouts, and the AMD MI300X GPUs. Understanding the hardware is the first step to optimizing for it.

### [Lesson 2: Amdahl's Law and Scalability](./2-amdahls-law/README.md)
Experience the limits of parallel scaling. You will compile and run a simple OpenMP C++ program calculating $\pi$, manually varying thread counts. You'll plot the execution times to calculate speedup and directly observe Amdahl's Law in action as scaling plateaus.

### [Lesson 3: Strong and Weak Scaling](./3-scaling/README.md)
Understand the difference between evaluating performance for fixed-size problems (Strong Scaling) and growing problems (Weak Scaling). You will run two different programs (a compute-heavy math loop and a memory-bound array addition) to measure execution times, plot scaling curves, and observe how real-world hardware bottlenecks like memory bandwidth saturation affect scalability.

### [Lesson 4: Hardware Limits and Roofline Model](./4-roofline-model/README.md)
Analyze algorithm limitations mathematically. You will calculate the **Arithmetic Intensity** of two different algorithms (SAXPY and a highly-cached Matrix Multiplication) and determine whether they are "Compute Bound" or "Memory Bound" based on the hardware limits you discovered in Lesson 1.
