import matplotlib.pyplot as plt
import csv

# Initialize data structures
threads = []

times_ch_strong = {}
times_ch_weak = {}
times_mb_strong = {}
times_mb_weak = {}

# Read data from CSV
with open('scaling_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = int(row['Threads'])
        if t not in threads:
            threads.append(t)
        
        time_val = float(row['Time'])
        prog = row['Program']
        stype = row['ScalingType']

        if prog == 'compute_heavy' and stype == 'strong':
            times_ch_strong[t] = time_val
        elif prog == 'compute_heavy' and stype == 'weak':
            times_ch_weak[t] = time_val
        elif prog == 'memory_bound' and stype == 'strong':
            times_mb_strong[t] = time_val
        elif prog == 'memory_bound' and stype == 'weak':
            times_mb_weak[t] = time_val

threads.sort()

# Calculate Speedups and Efficiencies
t1_ch_strong = times_ch_strong[threads[0]]
t1_ch_weak = times_ch_weak[threads[0]]
t1_mb_strong = times_mb_strong[threads[0]]
t1_mb_weak = times_mb_weak[threads[0]]

speedup_ch_strong = [t1_ch_strong / times_ch_strong[t] for t in threads]
speedup_mb_strong = [t1_mb_strong / times_mb_strong[t] for t in threads]

efficiency_ch_weak = [t1_ch_weak / times_ch_weak[t] for t in threads]
efficiency_mb_weak = [t1_mb_weak / times_mb_weak[t] for t in threads]

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Strong Scaling
ax1.plot(threads, speedup_ch_strong, marker='o', color='blue', label='Compute Heavy (Good)')
ax1.plot(threads, speedup_mb_strong, marker='x', color='red', label='Memory Bound (Poor)')
ax1.plot(threads, threads, linestyle='--', color='gray', label='Ideal Speedup')
ax1.set_title('Strong Scaling Comparison')
ax1.set_xlabel('Threads')
ax1.set_ylabel('Speedup')
ax1.set_xticks(threads)
ax1.legend()
ax1.grid(True)

# Plot Weak Scaling
ax2.plot(threads, efficiency_ch_weak, marker='o', color='blue', label='Compute Heavy (Good)')
ax2.plot(threads, efficiency_mb_weak, marker='x', color='red', label='Memory Bound (Poor)')
ax2.axhline(y=1.0, linestyle='--', color='gray', label='Ideal Efficiency')
ax2.set_title('Weak Scaling Comparison')
ax2.set_xlabel('Threads')
ax2.set_ylabel('Efficiency (T_1 / T_N)')
ax2.set_ylim([0.0, 1.2])
ax2.set_xticks(threads)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('scaling_curves.png')
