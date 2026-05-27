cmake_minimum_required(VERSION 3.20)
project(lesson-7-nonblocking LANGUAGES C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(MPI REQUIRED)

add_executable(nonblocking-exercise exercise.c)
target_link_libraries(nonblocking-exercise MPI::MPI_C)

add_executable(nonblocking-solution solution.c)
target_link_libraries(nonblocking-solution MPI::MPI_C)
