add_executable(nonblocking-exercise exercise.c)
target_link_libraries(nonblocking-exercise MPI::MPI_C)

add_executable(nonblocking-solution solution.c)
target_link_libraries(nonblocking-solution MPI::MPI_C)
