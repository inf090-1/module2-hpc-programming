CC = hipcc
HIP_ARCH ?= gfx942
CFLAGS = -O3 -Wall -fopenmp --offload-arch=$(HIP_ARCH)
ROCBLAS_FLAGS = -lrocblas

all: exercise_matmul_hip solution_matmul_hip solution_rocblas_matmul

exercise_matmul_hip: exercise_matmul_hip.cpp
	$(CC) $(CFLAGS) $< -o $@

solution_matmul_hip: solution_matmul_hip.cpp
	$(CC) $(CFLAGS) $< -o $@

solution_rocblas_matmul: solution_rocblas_matmul.cpp
	$(CC) $(CFLAGS) $(ROCBLAS_FLAGS) $< -o $@

clean:
	rm -f exercise_matmul_hip solution_matmul_hip solution_rocblas_matmul
