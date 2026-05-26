HIPCC = hipcc
HIP_ARCH ?= gfx942

GCC = gcc
AMDCLANG = amdclang
AMDCLANGXX = amdclang++

GCC_CFLAGS = -O3 -Wall -fopenmp
HIP_CFLAGS = -O3 -Wall -fopenmp --offload-arch=$(HIP_ARCH)
HIP_CXXFLAGS = -O3 -Wall -fopenmp --offload-arch=$(HIP_ARCH)
OMP_TARGET_CFLAGS = -O3 -Wall -fopenmp \
	-fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=$(HIP_ARCH)
OMP_TARGET_CXXFLAGS = -O3 -Wall -fopenmp \
	-fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target -march=$(HIP_ARCH) -std=c++17

# Lessons 1-2 run on CPU (gcc)
GCC_SOURCES = $(wildcard 1-simple-openmp-task/*.c) $(wildcard 2-treereduce-matrix-sum/*.c)
GCC_TARGETS = $(GCC_SOURCES:.c=)

# Lessons 3-5 use OpenMP target and/or HIP
OMP_TARGET_C_SOURCES = $(wildcard 4-omp-target-vector-add/*.c)
OMP_TARGET_C_TARGETS = $(OMP_TARGET_C_SOURCES:.c=)

OMP_TARGET_CPP_SOURCES = $(wildcard 5-mandelbrot-omp-target/*.cpp) \
                         $(wildcard 6-matmul-omp-target-vs-hip/omp_target_matmul.cpp)
OMP_TARGET_CPP_TARGETS = $(OMP_TARGET_CPP_SOURCES:.cpp=)

HIP_CPP_SOURCES = $(wildcard 6-matmul-omp-target-vs-hip/hip_matmul.cpp)
HIP_CPP_TARGETS = $(HIP_CPP_SOURCES:.cpp=)

ROCBLAS_SOURCES = $(wildcard 6-matmul-omp-target-vs-hip/rocblas_matmul.cpp)
ROCBLAS_TARGETS = $(ROCBLAS_SOURCES:.cpp=)

.PHONY: all clean

all: $(GCC_TARGETS) $(OMP_TARGET_C_TARGETS) $(OMP_TARGET_CPP_TARGETS) $(HIP_CPP_TARGETS) $(ROCBLAS_TARGETS)

# Compile CPU (gcc)
%: %.c
	$(GCC) $(GCC_CFLAGS) -o $@ $<

# Compile OpenMP target C files with amdclang
$(OMP_TARGET_C_TARGETS): %: %.c
	$(AMDCLANG) $(OMP_TARGET_CFLAGS) -o $@ $<

# Compile OpenMP target C++ files with amdclang++
$(OMP_TARGET_CPP_TARGETS): %: %.cpp
	$(AMDCLANGXX) $(OMP_TARGET_CXXFLAGS) -o $@ $<

# Compile HIP C++ files with hipcc
$(HIP_CPP_TARGETS): %: %.cpp
	$(HIPCC) $(HIP_CXXFLAGS) -std=c++17 -o $@ $<

# Compile rocBLAS runner with hipcc + rocblas
$(ROCBLAS_TARGETS): %: %.cpp
	$(HIPCC) $(HIP_CXXFLAGS) -std=c++17 -lrocblas -o $@ $<

clean:
	rm -f $(GCC_TARGETS) $(OMP_TARGET_C_TARGETS) $(OMP_TARGET_CPP_TARGETS) $(HIP_CPP_TARGETS) $(ROCBLAS_TARGETS)
