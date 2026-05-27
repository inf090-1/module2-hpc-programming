cmake_minimum_required(VERSION 3.20)
project(lesson-3-ring LANGUAGES C)

find_package(MPI REQUIRED)

add_executable(ring-exercise exercise.c)
target_link_libraries(ring-exercise MPI::MPI_C)

add_executable(ring-solution solution.c)
target_link_libraries(ring-solution MPI::MPI_C)
