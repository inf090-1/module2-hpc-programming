cmake_minimum_required(VERSION 3.20)
project(lesson-4-collectives LANGUAGES C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(MPI REQUIRED)

add_executable(collectives-exercise exercise.c)
target_link_libraries(collectives-exercise MPI::MPI_C)

add_executable(collectives-solution solution.c)
target_link_libraries(collectives-solution MPI::MPI_C)
