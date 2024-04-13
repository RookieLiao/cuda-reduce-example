cmake_minimum_required(VERSION 3.10)

# set the project name
project(cuda_reduction VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add_library(picojson picojson.c)
# add_executable(picojson_test test.c)
# target_link_libraries(picojson_test picojson)

# add the executable
add_executable(cpu_reduce cpu_reduce.cpp)
# add_executable(cuda_reduce_v1 cuda_reduce_v1.cu)
