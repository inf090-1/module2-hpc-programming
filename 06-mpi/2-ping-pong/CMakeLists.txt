add_executable(ping-pong-exercise exercise.c)
target_link_libraries(ping-pong-exercise MPI::MPI_C)

add_executable(ping-pong-solution solution.c)
target_link_libraries(ping-pong-solution MPI::MPI_C)
