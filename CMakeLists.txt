cmake_minimum_required(VERSION 3.12)
project(CUDA_Project)

find_package(CUDA REQUIRED)

file(GLOB CUDA_SOURCES *.cu)
list(REMOVE_ITEM CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common.cu)

cuda_add_library(common common.cu)

foreach(source_file ${CUDA_SOURCES})
  get_filename_component(executable_name ${source_file} NAME_WE)
  get_filename_component(kernel_file ${source_file} NAME)
  set(kernel_file ${CMAKE_SOURCE_DIR}/kernels/${kernel_file})
  cuda_add_executable(${executable_name} ${source_file} ${kernel_file})
  target_link_libraries(${executable_name} common)
endforeach()
