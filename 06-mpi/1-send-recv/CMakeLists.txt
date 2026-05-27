cmake_minimum_required(VERSION 3.20)
project(lesson-1-send-recv LANGUAGES C)

find_package(MPI REQUIRED)

add_executable(send-recv-exercise exercise.c)
target_link_libraries(send-recv-exercise MPI::MPI_C)

add_executable(send-recv-solution solution.c)
target_link_libraries(send-recv-solution MPI::MPI_C)
