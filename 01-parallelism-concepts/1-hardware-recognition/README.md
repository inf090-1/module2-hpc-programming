# Lesson 1: Hardware Recognition

In this section, we will inspect the hardware topology of our cluster. This cluster has:
- **1 GPU Node (`g1`)**: 20 effective cores, 257GB RAM, 2x AMD MI300X GPUs. Partition: `gpu`.
- **4 CPU Nodes (`n[1-4]`)**: 4 effective cores each, 7GB RAM per node. Partition: `cpu`.

## Inspecting CPU Architecture

First, let's look at the CPU nodes. Allocate an interactive session on a CPU node and run `lscpu` to inspect its core architecture:

```bash
srun -N 1 -p cpu --pty bash
lscpu
exit
```

**Output Example:**
```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  4
On-line CPU(s) list:                     0-3
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz
...
Thread(s) per core:                      1
Core(s) per socket:                      1
Socket(s):                               4
...
L1d cache:                               128 KiB (4 instances)
L1i cache:                               128 KiB (4 instances)
L2 cache:                                16 MiB (4 instances)
L3 cache:                                64 MiB (4 instances)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-3
```

**Questions to consider:**
- How many cores per socket are there?
- What are the sizes of the L1, L2, and L3 caches?

## Inspecting NUMA Topology

Now, let's move to the powerful GPU node. Modern multi-socket systems have Non-Uniform Memory Access (NUMA) architectures. Run `numactl -H` to visualize the NUMA topology and memory distribution.

```bash
srun -p gpu numactl -H
```

**Output Example:**
```text
available: 1 nodes (0)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
node 0 size: 257216 MB
node 0 free: 231119 MB
node distances:
node   0 
  0:  10 
```

**Questions to consider:**
- How many NUMA nodes are present?
- How is the 257GB of RAM distributed across these nodes?
- What are the distances between different NUMA nodes?

## Inspecting GPU Architecture

Finally, let's inspect the AMD MI300X GPUs on the `g1` node using `rocm-smi`.

```bash
srun -p gpu rocm-smi
```

**Output Example:**
```text
============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                  
==========================================================================================================================
0       1     0x74a1,   58197  42.0°C      135.0W    NPS1, SPX, 0        122Mhz  900Mhz  0%   auto  750.0W  0%     0%    
1       2     0x74a1,   33617  43.0°C      133.0W    NPS1, SPX, 0        122Mhz  900Mhz  0%   auto  750.0W  0%     0%    
==========================================================================================================================
================================================== End of ROCm SMI Log ===================================================
```

**Reflection:**
1. What is the power cap configured for each MI300X GPU?
2. What are the current temperatures of the GPUs?
