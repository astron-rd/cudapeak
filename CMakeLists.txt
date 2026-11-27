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

# Unless BUILD_WITH_HIP is given explicitly, default to building for AMD (HIP)
# when no CUDA toolkit is found and a HIP compiler is available.
set(BUILD_WITH_HIP_DEFAULT OFF)
if(NOT DEFINED BUILD_WITH_HIP)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      message(
        STATUS "CUDA not found, but HIP is: defaulting BUILD_WITH_HIP to ON")
      set(BUILD_WITH_HIP_DEFAULT ON)
    endif()
  endif()
endif()

option(BUILD_WITH_HIP "Build with HIP (default: auto-detect)"
       ${BUILD_WITH_HIP_DEFAULT})
option(BUILD_WITH_PMT "Build with Power Measurement Toolkit support" OFF)
option(BUILD_WITH_FMT "Build with Frequency Measurement Toolkit support" OFF)

if(BUILD_WITH_HIP)
  set(CUDAWRAPPERS_BACKEND "HIP")
  enable_language(HIP)
  add_compile_definitions("__HIP__")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
  find_package(CUDAToolkit REQUIRED)
endif()

include(FetchContent)
set(CUDAWRAPPERS_COMPONENTS nvml;nvrtc)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG main)
FetchContent_MakeAvailable(cudawrappers)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.12.0)
FetchContent_MakeAvailable(json)

if(BUILD_WITH_PMT)
  if(BUILD_WITH_HIP)
    # Use PMT's AMDSMI backend rather than ROCM: FMT already links libamd_smi,
    # and loading librocm_smi64 alongside it segfaults .
    set(PMT_BUILD_AMDSMI ON)
  else()
    set(PMT_BUILD_NVML ON)
    set(PMT_BUILD_TEGRA ON)
  endif()
  set(PMT_CUDAWRAPPERS_API ON)
  FetchContent_Declare(
    pmt
    GIT_REPOSITORY https://git.astron.nl/RD/pmt
    GIT_TAG main)
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

enable_testing()

add_subdirectory(common)
add_subdirectory(benchmarks)
