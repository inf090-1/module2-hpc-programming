set(CMAKE_C_COMPILER "hipcc" CACHE STRING "HIP compiler wrapper")
set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "HIP compiler wrapper")
set(CMAKE_C_FLAGS "-O3")
set(CMAKE_CXX_FLAGS "-O3")

set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation root")

add_executable(mpi-gpu-exercise exercise_mpi_gpu.c)
target_compile_definitions(mpi-gpu-exercise PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(mpi-gpu-exercise PRIVATE ${MPI_C_INCLUDE_DIRS} ${ROCM_PATH}/include)
target_link_libraries(mpi-gpu-exercise ${MPI_C_LIBRARIES} ${ROCM_PATH}/lib/libamdhip64.so)

add_executable(mpi-gpu-solution solution_mpi_gpu.c)
target_compile_definitions(mpi-gpu-solution PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(mpi-gpu-solution PRIVATE ${MPI_C_INCLUDE_DIRS} ${ROCM_PATH}/include)
target_link_libraries(mpi-gpu-solution ${MPI_C_LIBRARIES} ${ROCM_PATH}/lib/libamdhip64.so)

add_executable(rccl-gpu-exercise exercise_rccl_gpu.c)
target_compile_definitions(rccl-gpu-exercise PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(rccl-gpu-exercise PRIVATE ${MPI_C_INCLUDE_DIRS} ${ROCM_PATH}/include)
target_link_libraries(rccl-gpu-exercise ${MPI_C_LIBRARIES} ${ROCM_PATH}/lib/libamdhip64.so ${ROCM_PATH}/lib/librccl.so)

add_executable(rccl-gpu-solution solution_rccl_gpu.c)
target_compile_definitions(rccl-gpu-solution PRIVATE __HIP_PLATFORM_AMD__)
target_include_directories(rccl-gpu-solution PRIVATE ${MPI_C_INCLUDE_DIRS} ${ROCM_PATH}/include)
target_link_libraries(rccl-gpu-solution ${MPI_C_LIBRARIES} ${ROCM_PATH}/lib/libamdhip64.so ${ROCM_PATH}/lib/librccl.so)
