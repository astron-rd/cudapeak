cmake_minimum_required(VERSION 3.17)
project(cudapeak LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CUDA_ARCHITECTURES
    "86"
    CACHE STRING "CUDA architecture")

include(FetchContent)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.0)
FetchContent_MakeAvailable(cxxopts)

option(BUILD_WITH_PMT "Build with Power Measurement Toolkit support" OFF)
option(BUILD_WITH_FMT "Build with Frequency Measurement Toolkit support" OFF)
option(BUILD_KERNEL_DEBUG "Build kernels in debug mode")
option(BUILD_WITH_HIP "Build with HIP")

if(BUILD_WITH_HIP)
  set(CUDAWRAPPERS_BACKEND "HIP")
  enable_language(HIP)
else()
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
endif()

include(FetchContent)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG main)
FetchContent_MakeAvailable(cudawrappers)

if(BUILD_WITH_PMT)
  if(BUILD_WITH_HIP)
    set(PMT_BUILD_ROCM ON)
  else()
    set(PMT_BUILD_NVML ON)
    set(PMT_BUILD_TEGRA ON)
  endif()
  FetchContent_Declare(pmt GIT_REPOSITORY https://git.astron.nl/RD/pmt)
  FetchContent_MakeAvailable(pmt)
  add_compile_definitions("HAVE_PMT")
endif()

if(BUILD_WITH_FMT)
  if(BUILD_WITH_HIP)
    set(FMT_BUILD_AMDSMI ON)
  else()
    set(FMT_BUILD_NVIDIA ON)
  endif()
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://git.astron.nl/RD/fmt
    GIT_TAG main)
  FetchContent_MakeAvailable(fmt)
  add_compile_definitions("HAVE_FMT")
endif()

file(GLOB CUDA_SOURCES *.cu)
list(REMOVE_ITEM CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Benchmark.cu)

if(BUILD_WITH_HIP)
  set(CUDA_SOURCES dmem.cu fp16.cu fp32.cu mma.cu stream.cu)
endif()

add_library(common OBJECT Benchmark.cu common.cpp Measurement.cpp)
target_link_libraries(common cxxopts cudawrappers::cu)
if(BUILD_WITH_PMT)
  target_link_libraries(common pmt)
endif()
if(BUILD_WITH_FMT)
  target_link_libraries(common fmt)
endif()
if(BUILD_WITH_PMT OR BUILD_WITH_FMT)
  find_package(Threads)
  target_link_libraries(common Threads::Threads)
endif()
if(BUILD_WITH_HIP)
  set_source_files_properties(Benchmark.cu PROPERTIES LANGUAGE HIP)
  set_source_files_properties(common.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(Measurement.cpp PROPERTIES LANGUAGE HIP)
endif()

foreach(source_file ${CUDA_SOURCES})
  get_filename_component(executable_name ${source_file} NAME_WE)
  get_filename_component(kernel_file ${source_file} NAME)
  set(KERNEL_FILE "${CMAKE_SOURCE_DIR}/kernels/${kernel_file}")
  add_executable(${executable_name} ${source_file} ${KERNEL_FILE})
  target_link_libraries(${executable_name} common)
  if(NOT BUILD_WITH_HIP AND BUILD_KERNEL_DEBUG)
    target_compile_options(
      ${executable_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g
                                 --generate-line-info >)
  endif()
  if(BUILD_WITH_HIP)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(${KERNEL_FILE} PROPERTIES LANGUAGE HIP)
  endif()
endforeach()
