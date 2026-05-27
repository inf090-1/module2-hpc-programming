cmake_minimum_required(VERSION 3.20)
project(lesson-5-gpu-aware-mpi LANGUAGES C)

set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation root")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(MPI REQUIRED)

set(OFFLOAD_ARCH gfx942 CACHE STRING "AMD GPU target architecture")

function(add_gpu_target name src)
    add_executable(${name} ${src})
    target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)
    target_compile_options(${name} PRIVATE --offload-arch=${OFFLOAD_ARCH})
    target_include_directories(${name} PRIVATE ${MPI_C_INCLUDE_DIRS} ${ROCM_PATH}/include)
    target_link_libraries(${name} ${MPI_C_LIBRARIES} ${ROCM_PATH}/lib/libamdhip64.so ${ARGN})
endfunction()

add_gpu_target(mpi-gpu-exercise exercise_mpi_gpu.c)
add_gpu_target(mpi-gpu-solution  solution_mpi_gpu.c)
add_gpu_target(rccl-gpu-exercise exercise_rccl_gpu.c ${ROCM_PATH}/lib/librccl.so)
add_gpu_target(rccl-gpu-solution  solution_rccl_gpu.c ${ROCM_PATH}/lib/librccl.so)
