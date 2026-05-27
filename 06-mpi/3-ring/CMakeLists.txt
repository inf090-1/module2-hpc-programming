add_executable(ring-exercise exercise.c)
target_link_libraries(ring-exercise MPI::MPI_C)

add_executable(ring-solution solution.c)
target_link_libraries(ring-solution MPI::MPI_C)
