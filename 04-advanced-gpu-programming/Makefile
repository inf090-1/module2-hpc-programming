SUBDIRS = \
  1-matmul-coalescing \
  2-convolution \
  3-attention \
  4-hip-profiling-amd-tools

HIPCC = hipcc
HIP_ARCH ?= gfx942

CXXFLAGS = -O3 -Wall -fopenmp --offload-arch=$(HIP_ARCH)

# MIOpen is only needed for the convolution exercise/solution.
MIOPEN_LIBS = -lMIOpen

SOURCES = $(wildcard $(addsuffix /*.cpp,$(SUBDIRS)))
TARGETS = $(SOURCES:.cpp=)

.PHONY: all clean

all: $(TARGETS)

%: %.cpp
	EXTRA_LIBS=""; \
	if echo "$<" | grep -q "2-convolution/"; then EXTRA_LIBS="$(MIOPEN_LIBS)"; fi; \
	$(HIPCC) $(CXXFLAGS) -o $@ $< $$EXTRA_LIBS

clean:
	rm -f $(TARGETS)
