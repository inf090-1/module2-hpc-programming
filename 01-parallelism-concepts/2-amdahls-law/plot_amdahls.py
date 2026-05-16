import matplotlib.pyplot as plt
import csv

threads = []
times = {}

# Read data from CSV
with open('amdahl_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = int(row['Threads'])
        threads.append(t)
        times[t] = float(row['Time'])

threads.sort()

# Calculate Speedup
t1 = times[threads[0]]
speedup = [t1 / times[t] for t in threads]

# Plotting
plt.figure(figsize=(8, 6))

plt.plot(threads, speedup, marker='o', color='blue', label='Actual Speedup')
plt.plot(threads, threads, linestyle='--', color='gray', label='Ideal Speedup')

plt.title("Amdahl's Law - Pi Calculation Scaling")
plt.xlabel('Threads')
plt.ylabel('Speedup')
plt.xticks(threads)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('amdahls_curve.png')
