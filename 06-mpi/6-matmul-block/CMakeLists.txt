cmake_minimum_required(VERSION 3.20)
project(lesson-6-matmul-block LANGUAGES C)

find_package(MPI REQUIRED)
find_package(OpenMP)

if(OpenMP_C_FOUND)
  add_executable(matmul-block-exercise exercise.c)
  target_link_libraries(matmul-block-exercise MPI::MPI_C OpenMP::OpenMP_C)

  add_executable(matmul-block-solution solution.c)
  target_link_libraries(matmul-block-solution MPI::MPI_C OpenMP::OpenMP_C)
else()
  message(WARNING "OpenMP not found — matmul-block lesson will not be built")
endif()
