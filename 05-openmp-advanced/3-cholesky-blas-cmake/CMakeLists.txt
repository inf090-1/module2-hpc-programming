cmake_minimum_required(VERSION 3.18)
project(cholesky_blas C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)


# Try to locate a BLAS implementation (OpenBLAS, MKL, etc.)
# We rely on classic BLAS symbols (dtrsm_/dsyrk_) called from C.
find_package(BLAS REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(cholesky_blas_common OBJECT cholesky_blas_util.c)

target_compile_options(cholesky_blas_common PRIVATE -O3 -Wall -Wextra)

find_package(OpenMP REQUIRED)

add_executable(cholesky_blas cholesky_blas.c)
target_link_libraries(cholesky_blas PRIVATE OpenMP::OpenMP_C ${BLAS_LIBRARIES} m)
target_include_directories(cholesky_blas PRIVATE ${BLAS_INCLUDE_DIRS})

# Link the shared utility object file

target_link_libraries(cholesky_blas PRIVATE $<TARGET_OBJECTS:cholesky_blas_common>)

target_compile_options(cholesky_blas PRIVATE -O3 -Wall -Wextra)

