cmake_minimum_required(VERSION 3.20)
project(mpi-lessons LANGUAGES C)

set(CMAKE_C_FLAGS "-O3 -Wall")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(MPI REQUIRED)

add_subdirectory(1-send-recv)
add_subdirectory(2-ping-pong)
add_subdirectory(3-ring)
add_subdirectory(4-collectives)
add_subdirectory(6-matmul-block)
add_subdirectory(7-nonblocking)

option(BUILD_GPU_LESSONS "Build GPU-aware MPI lessons (requires ROCm)" OFF)
if(BUILD_GPU_LESSONS)
  add_subdirectory(5-gpu-aware-mpi)
endif()
