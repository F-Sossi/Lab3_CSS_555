cmake_minimum_required(VERSION 3.0)

project(MyProject CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
add_executable(lab3 src/kernel.cu)

set_property(TARGET lab3 PROPERTY CUDA_ARCHITECTURES 86)

target_link_libraries(lab3 PRIVATE cublas)
