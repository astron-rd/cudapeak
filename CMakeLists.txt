cmake_minimum_required(VERSION 3.17)
project(cudapeak LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

include(FetchContent)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.0)
FetchContent_MakeAvailable(cxxopts)

option(BUILD_WITH_PMT "Build with Power Measurement Toolkit support" OFF)
option(BUILD_WITH_FMT "Build with Frequency Measurement Toolkit support" OFF)
option(BUILD_WITH_HIP "Build with HIP")

if(BUILD_WITH_HIP)
  set(CUDAWRAPPERS_BACKEND "HIP")
  enable_language(HIP)
  add_compile_definitions("__HIP__")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
  find_package(CUDAToolkit REQUIRED)
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

add_subdirectory(common)
add_subdirectory(benchmarks)
