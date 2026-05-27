cmake_minimum_required(VERSION 3.20)
project(lesson-2-ping-pong LANGUAGES C)

find_package(MPI REQUIRED)

add_executable(ping-pong-exercise exercise.c)
target_link_libraries(ping-pong-exercise MPI::MPI_C)

add_executable(ping-pong-solution solution.c)
target_link_libraries(ping-pong-solution MPI::MPI_C)
