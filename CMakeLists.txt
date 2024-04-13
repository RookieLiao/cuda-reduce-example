cmake_minimum_required(VERSION 3.10)

# set the project name
project(cuda_reduction LANGUAGES CXX CUDA VERSION 0.1.0)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define a shared library
add_library(reduce_impl SHARED cpu_reduce.cpp cuda_reduce_v0.cu)

enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(profile_reduce profile_reduce.cpp)
target_link_libraries(profile_reduce ${CUDA_LIBRARIES} reduce_impl)

# Specify the CUDA architecture (replace with your GPU architecture)
set_target_properties(reduce_impl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"  # Adjust according to your GPU
)
