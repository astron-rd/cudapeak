cmake_minimum_required(VERSION 3.12)
project(CUDA_Project)

find_package(CUDA REQUIRED)

option(BUILD_WITH_PMT "Build with Power Measurement Toolkit support" OFF)

if(BUILD_WITH_PMT)
  find_package(pmt)
  add_compile_definitions("HAVE_PMT")
endif()

file(GLOB CUDA_SOURCES *.cu)
list(REMOVE_ITEM CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/common.cu)

foreach(source_file ${CUDA_SOURCES})
  get_filename_component(executable_name ${source_file} NAME_WE)
  get_filename_component(kernel_file ${source_file} NAME)
  set(KERNEL_FILE "${CMAKE_SOURCE_DIR}/kernels/${kernel_file}")
  cuda_add_executable(${executable_name} ${source_file} ${KERNEL_FILE}
                      common.cu)
  if(BUILD_WITH_PMT)
    target_link_libraries(${executable_name} pmt)
  endif()
endforeach()
