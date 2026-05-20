import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('results.csv')

serial_time = df[df['TYPE'] == 'SERIAL']['TIME'].values[0]

simd_data = df[df['TYPE'] == 'SIMD'].copy()
blas_data = df[df['TYPE'] == 'BLAS'].copy()

simd_data['SPEEDUP'] = serial_time / simd_data['TIME']
blas_data['SPEEDUP'] = serial_time / blas_data['TIME']

plt.figure(figsize=(10, 6))

plt.plot(simd_data['THREADS'], simd_data['SPEEDUP'], marker='o', label='OpenMP+SIMD')
plt.plot(blas_data['THREADS'], blas_data['SPEEDUP'], marker='s', label='OpenBLAS')
plt.plot(simd_data['THREADS'], simd_data['THREADS'], linestyle='--', color='gray', label='Ideal Speedup')

plt.xlabel('Number of Threads')
plt.ylabel('Speedup (relative to optimized serial)')
plt.title('Matmul Speedup: OpenMP+SIMD vs OpenBLAS vs Serial (1024x1024)')
plt.legend()
plt.grid(True)
plt.xticks([1, 2, 4])

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/matmul_speedup.png')
