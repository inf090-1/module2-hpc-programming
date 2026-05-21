SUBDIRS = 0-hello-gpu 1-vector-add 2-stencil-2d 3-shared-mem-stencil 4-matmul-naive

HIPCC = hipcc
HIP_ARCH ?= native
ROCM_ROOT ?= /opt/rocm
GCC_RUNTIME_LIB ?= /opt/ohpc/pub/compiler/gcc/14.2.0/lib64

CXXFLAGS = -O3 -Wall -fopenmp --offload-arch=$(HIP_ARCH)

# Special rule for rocblas
4-matmul-naive/solution_rocblas_matmul: 4-matmul-naive/solution_rocblas_matmul.cpp
	$(HIPCC) $(CXXFLAGS) -lrocblas -o $@ $<

SOURCES = $(filter-out 4-matmul-naive/solution_rocblas_matmul.cpp, $(wildcard $(addsuffix /*.cpp,$(SUBDIRS))))
TARGETS = $(SOURCES:.cpp=) 4-matmul-naive/solution_rocblas_matmul

.PHONY: all clean print-config

all: $(TARGETS)

%: %.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

print-config:
	@echo "HIPCC=$(HIPCC)"
	@echo "HIP_ARCH=$(HIP_ARCH)"
	@echo "ROCM_ROOT=$(ROCM_ROOT)"
	@echo "GCC_RUNTIME_LIB=$(GCC_RUNTIME_LIB)"

clean:
	rm -f $(TARGETS)
