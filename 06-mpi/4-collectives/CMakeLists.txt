add_executable(collectives-exercise exercise.c)
target_link_libraries(collectives-exercise MPI::MPI_C)

add_executable(collectives-solution solution.c)
target_link_libraries(collectives-solution MPI::MPI_C)
