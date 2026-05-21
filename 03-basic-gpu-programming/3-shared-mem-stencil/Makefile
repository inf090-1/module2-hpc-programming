CC = hipcc
HIP_ARCH ?= gfx942
CFLAGS = -O3 -Wall -fopenmp --offload-arch=$(HIP_ARCH)

all: exercise_stencil_2d_hip solution_stencil_2d_hip

exercise_stencil_2d_hip: exercise_stencil_2d_hip.cpp
	$(CC) $(CFLAGS) $< -o $@

solution_stencil_2d_hip: solution_stencil_2d_hip.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f exercise_stencil_2d_hip solution_stencil_2d_hip
