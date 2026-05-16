import matplotlib.pyplot as plt
import numpy as np
import csv

threads = []
times_saxpy = {}
times_matmul = {}

# Read data from CSV
with open('roofline_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = int(row['Threads'])
        if t not in threads:
            threads.append(t)
        
        algo = row['Algorithm']
        time_val = float(row['Time'])
        
        if algo == 'SAXPY':
            times_saxpy[t] = time_val
        elif algo == 'MATMUL':
            times_matmul[t] = time_val

threads.sort()

# Calculate Speedup
t1_saxpy = times_saxpy[threads[0]]
t1_matmul = times_matmul[threads[0]]

speedup_saxpy = [t1_saxpy / times_saxpy[t] for t in threads]
speedup_matmul = [t1_matmul / times_matmul[t] for t in threads]

# Plotting Speedup
plt.figure(figsize=(8, 6))

plt.plot(threads, speedup_saxpy, marker='x', color='red', label='SAXPY (Memory Bound)')
plt.plot(threads, speedup_matmul, marker='o', color='blue', label='Matrix Multiplication (Compute Bound)')
plt.plot(threads, threads, linestyle='--', color='gray', label='Ideal Speedup')

plt.title("Scaling Comparison: Memory Bound vs Compute Bound")
plt.xlabel('Threads')
plt.ylabel('Speedup')
plt.xticks(threads)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('roofline_scaling.png')
plt.close()

# -----------------------------------------------------
# Plotting Roofline Model
# -----------------------------------------------------
time_saxpy_4 = times_saxpy[4]
time_matmul_4 = times_matmul[4]

# SAXPY metrics: N=100,000,000
# FLOPs = 2 * N (1 mult, 1 add)
# Bytes = 3 * N * 4 bytes (read X, read Y, write Y)
saxpy_flops = 2.0 * 100_000_000
saxpy_bytes = 12.0 * 100_000_000
saxpy_ai = saxpy_flops / saxpy_bytes
saxpy_gflops = (saxpy_flops / time_saxpy_4) / 1e9

# MATMUL metrics: N=1024
# The i-k-j loop order significantly improves cache locality.
# FLOPs = 2 * N^3
# Bytes = 4 * N^2 (matrices A, B, C total size) + some overhead.
# Since it's heavily cached, memory traffic is mostly compulsory misses.
matmul_flops = 2.0 * (1024**3)
matmul_bytes = 3.0 * (1024**2) * 4 # Theoretical minimum memory traffic (read A, B, write C)
matmul_ai = matmul_flops / matmul_bytes
matmul_gflops = (matmul_flops / time_matmul_4) / 1e9

# Hardware estimated theoretical peaks (Xeon E5-2620 v2 CPU allocation)
PEAK_MEM_BW = 25.0  # GB/s
PEAK_FLOPS = 67.0   # GFLOP/s

plt.figure(figsize=(8, 6))
ai_vals = np.logspace(-2, 4, 500)
perf_vals = np.minimum(PEAK_FLOPS, ai_vals * PEAK_MEM_BW)

# Plot Rooflines
plt.plot(ai_vals, perf_vals, color='black', linewidth=2, label='Hardware Roofline')
plt.plot(ai_vals, [PEAK_FLOPS]*len(ai_vals), linestyle='--', color='gray', label=f'Peak Compute ({PEAK_FLOPS} GF/s)')
plt.plot(ai_vals, ai_vals * PEAK_MEM_BW, linestyle='-.', color='gray', label=f'Peak Mem Bandwidth ({PEAK_MEM_BW} GB/s)')

# Scatter plot the applications
plt.scatter([saxpy_ai], [saxpy_gflops], color='red', s=100, zorder=5, label=f'SAXPY (AI={saxpy_ai:.2f})')
plt.scatter([matmul_ai], [matmul_gflops], color='blue', s=100, zorder=5, label=f'MATMUL (AI={matmul_ai:.2f})')

plt.text(saxpy_ai, saxpy_gflops * 1.5, ' SAXPY', color='red', verticalalignment='bottom')
plt.text(matmul_ai, matmul_gflops * 1.5, ' MATMUL', color='blue', verticalalignment='bottom')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
plt.ylabel('Performance (GFLOP/s)')
plt.title('Roofline Model Analysis (4 Threads)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlim(0.01, 10000)
plt.ylim(0.1, 200)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roofline_model.png')
plt.close()

