add_executable(send-recv-exercise exercise.c)
target_link_libraries(send-recv-exercise MPI::MPI_C)

add_executable(send-recv-solution solution.c)
target_link_libraries(send-recv-solution MPI::MPI_C)
